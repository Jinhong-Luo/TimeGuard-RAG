"""仅在相同开发集和完整复核结果上选择思考强度。"""

import argparse
import copy
import math
from pathlib import Path
import statistics

from .common import digest, read_json, read_jsonl, write_json
from . import judge


MODEL_EFFORTS = {"low", "medium", "high", "xhigh", "max", "ultra"}


def validate_model_reviews(record, ids):
    """验证离线裁判与该次生成文件的绑定；跨档只比较固定裁判配置。"""
    manifest = record.get("judge_manifest") or {}
    identity = manifest.get("identity") or {}
    signature = manifest.get("signature")
    required = {"version", "judge_type", "model", "effort", "transport", "expected_ids",
                "generation_manifest_sha256", "generation_results_sha256", "gold_sha256",
                "prompt_sha256", "code_sha256", "schema", "concurrency", "request_timeout_seconds"}
    if (not required.issubset(identity) or digest(identity) != signature
            or identity["expected_ids"] != ids
            or identity["judge_type"] != "model" or identity["version"] != judge.JUDGE_VERSION
            or identity["model"] != judge.MODEL or identity["effort"] != judge.EFFORT
            or identity["transport"] != judge.TRANSPORT or identity["schema"] != judge.SCHEMA
            or type(identity["concurrency"]) is not int or identity["concurrency"] < 1
            or type(identity["request_timeout_seconds"]) not in (int, float)
            or not math.isfinite(identity["request_timeout_seconds"]) or identity["request_timeout_seconds"] <= 0
            or identity["prompt_sha256"] != digest(judge.INSTRUCTIONS)
            or identity["code_sha256"] != digest(Path(judge.__file__).read_bytes())):
        raise ValueError("缺少有效的当前固定模型裁判清单，或模型、强度、提示、代码与评分口径不符")
    if any(not isinstance(identity[key], str) or len(identity[key]) != 64
           or any(char not in "0123456789abcdef" for char in identity[key])
           for key in ("gold_sha256", "generation_manifest_sha256", "generation_results_sha256")):
        raise ValueError("裁判的金标或生成文件摘要无效")
    sources = record.get("source_hashes") or {}
    if any(not isinstance(sources.get(key), str) or len(sources[key]) != 64
           or any(char not in "0123456789abcdef" for char in sources[key])
           for key in ("model_reviews_manifest", "model_reviews")):
        raise ValueError("选择报告必须绑定本次裁判清单和判断文件的原始字节摘要")
    if (identity["generation_manifest_sha256"] != sources.get("manifest")
            or identity["generation_results_sha256"] != sources.get("results")):
        raise ValueError("模型裁判并非针对这份生成清单和结果，拒绝复用其他运行判断")
    rows = record.get("model_reviews") or []
    if len(rows) != len(ids) or {r.get("id") for r in rows} != set(ids):
        raise ValueError("模型判断缺题、重复或超出完整开发集")
    for row in rows:
        metadata = row.get("bridge_metadata") or {}
        if (row.get("judge_type") != "model" or row.get("judge_status") != "ok"
                or row.get("judge_run_signature") != signature
                or row.get("judge_model") != judge.MODEL or row.get("judge_effort") != judge.EFFORT
                or type(row.get("answer_correct")) is not bool
                or type(row.get("has_factual_claims")) is not bool
                or any(key not in row or (row[key] is not None and type(row[key]) is not bool)
                       for key in ("numeric_and_scope_correct", "citation_support"))
                or not row.get("reviewer") or not row.get("judgement_reason") or not row.get("raw_judgement")
                or metadata.get("transport") != judge.TRANSPORT or metadata.get("api_equivalent") is not False
                or metadata.get("actual_model") != judge.MODEL or metadata.get("reasoning_effort") != judge.EFFORT):
            raise ValueError("每题须有成功、完整且具实际模型/强度证据的固定模型答案判断")
    generated = record.get("generated_rows") or []
    if (len(generated) != len(ids) or {r.get("id") for r in generated} != set(ids)
            or any(r.get("status") != "ok" or not isinstance(r.get("answer"), str) or not r["answer"]
                   or type(r.get("elapsed_seconds")) not in (int, float)
                   or not math.isfinite(r["elapsed_seconds"]) or r["elapsed_seconds"] <= 0 for r in generated)):
        raise ValueError("模型选优需要39条完整成功生成及每题真实耗时，缺失不能当零成本")
    generated_by_id = {row["id"]: row for row in generated}
    for row in rows:
        parsed, _ = judge.parse_judgement({"model": row["judge_model"], "bridge_metadata": row["bridge_metadata"],
                                          "choices": [{"message": {"content": row["raw_judgement"]}}]})
        context = judge.citation_context(generated_by_id[row["id"]])
        mapping = {key: value for key, value in context.items() if key != "citations"}
        expected = judge.apply_citation_policy(parsed, context)
        if (row.get("citation_mapping") != mapping
                or any(row[key] != expected[key] for key in (*judge.FIELDS, "has_factual_claims"))):
            raise ValueError("裁判最终字段与原始模型判断或该答案引用映射不符")
    duration = statistics.mean(r["elapsed_seconds"] for r in generated)
    summary_duration = record["summary"].get("mean_seconds")
    if type(summary_duration) not in (int, float) or not math.isclose(duration, summary_duration, rel_tol=1e-9):
        raise ValueError("汇总耗时与逐题原始结果不符")
    config = {key: value for key, value in identity.items()
              if key not in ("expected_ids", "generation_manifest_sha256", "generation_results_sha256")}
    metrics = {"model_answer_accuracy": sum(r["answer_correct"] for r in rows) / len(ids),
               "model_reviewed": len(rows), "model_judge_status_counts": {"ok": len(rows)}}
    for field, target in (("numeric_and_scope_correct", "model_numeric_and_scope_accuracy"),
                          ("citation_support", "model_citation_support_rate")):
        values = [r[field] for r in rows if type(r[field]) is bool]
        metrics[target] = sum(values) / len(ids) if len(values) == len(ids) else None
        metrics[target + "_known_count"] = len(values)
        metrics[target + "_unknown_count"] = len(ids) - len(values)
        metrics[target + "_coverage"] = len(values) / len(ids)
    return config, metrics


def comparable_identity(identity):
    """只去除明确允许变化的档位和其传输证据，保留其他模型配置。"""
    current = copy.deepcopy({k: v for k, v in identity.items() if k not in ("effort", "transport_proof")})
    model = current["config"]["model"]
    parts = model.split("@")
    if parts[0] == "gpt-6.1-sol" and len(parts) in (2, 4):
        alias_effort = parts[1]
        if (alias_effort not in {"low", "medium", "high", "xhigh", "max", "ultra"}
                or alias_effort != identity["effort"] or any(not part for part in parts)):
            raise ValueError("模型别名必须严格包含当前思考强度及完整实例/提供方标识")
        # RAGFlow 完整标识为 模型名@实例名@提供方；只去掉模型名内部的已验证档位。
        current["config"]["model"] = "@".join([parts[0], *parts[2:]])
    elif parts[0] == "gpt-6.1-sol" and len(parts) not in (1, 3):
        raise ValueError("模型别名结构无效")
    return current


def choose(records, tolerance=0.02, review_source="human"):
    """只接收完整开发集、完整候选集合及冻结索引的运行记录。"""
    if not math.isfinite(tolerance) or not 0 <= tolerance <= 1:
        raise ValueError("质量容差必须是 0 到 1 之间的有限数")
    if review_source not in ("human", "model"):
        raise ValueError("复核来源只能为 human 或 model")
    if len(records) < 2:
        raise ValueError("至少需要两个不同思考强度，不能单档自称最优")
    candidates, shared, shared_snapshot, efforts, shared_judge = [], None, None, set(), None
    for record in records:
        complete_manifest, summary = record["manifest"], dict(record["summary"])
        identity = complete_manifest["identity"]
        ids = identity["query_ids"]
        if identity["split"] != "dev" or identity["retrieval_only"]:
            raise ValueError("只能使用开发集完整问答运行选强度，不能使用保留测试集")
        if (identity.get("scope") != "完整分组评测" or not ids
                or len(ids) != len(set(ids)) or len(ids) != identity.get("full_split_count")
                or digest(ids) != identity.get("full_split_query_ids_sha256")):
            raise ValueError("运行未覆盖完整开发集或缺少范围证明；连通测试不能参与选优")
        if summary["completed"] != len(ids) or summary["denominator"] != len(ids):
            raise ValueError("每档运行必须全部完成并保留完整分母")
        quality_key = "answer_accuracy"
        if review_source == "model":
            if len(ids) != 39 or identity["config"]["model"].split("@")[0] != judge.MODEL:
                raise ValueError("模型暂定选优仅接受当前固定的完整39题开发集")
            if summary.get("status_counts") != {"ok": len(ids)}:
                raise ValueError("模型暂定选优需要完整成功生成的汇总证明")
            judge_config, metrics = validate_model_reviews(record, ids)
            if shared_judge is not None and judge_config != shared_judge:
                raise ValueError("候选的裁判模型、强度、金标、提示、代码或schema不一致")
            shared_judge = judge_config
            summary.update(metrics)
            quality_key = "model_answer_accuracy"
        elif summary.get("answer_accuracy") is None or summary.get("reviewed") != len(ids):
            raise ValueError("每档人工答案复核必须全部完成；模型模式需显式选择")
        if any(count for status, count in summary.get("status_counts", {}).items() if status != "ok"):
            raise ValueError("候选存在请求、生成或缺失错误，不能据此比较模型思考强度")
        duration, accuracy = summary.get("mean_seconds"), summary[quality_key]
        if (not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0
                or not isinstance(accuracy, (int, float)) or not math.isfinite(accuracy) or not 0 <= accuracy <= 1):
            raise ValueError("候选耗时或正确率无效")
        effort = identity["effort"]
        if effort in efforts:
            raise ValueError("候选含重复思考强度，不能挑选同档最有利运行")
        efforts.add(effort)
        current = comparable_identity(identity)
        snapshot = complete_manifest.get("index_snapshot")
        if not snapshot:
            raise ValueError("缺少冻结索引快照，无法确认候选使用相同分块")
        if shared is not None and (current != shared or snapshot != shared_snapshot):
            raise ValueError("候选的问题、模型、Embedding、文档、索引或检索配置不一致")
        shared, shared_snapshot = current, snapshot
        candidates.append({"run": record["run"], "effort": effort, **summary})
    declared = shared["config"].get("efforts_to_probe", [])
    if len(declared) < 2 or len(declared) != len(set(declared)) or set(declared) != efforts:
        raise ValueError("必须提供配置中声明的全部不同候选档位，不能遗漏失败或较慢档位")
    if review_source == "model" and efforts != MODEL_EFFORTS:
        raise ValueError("模型暂定选优必须包含 low、medium、high、xhigh、max、ultra 六档")
    best_quality = max(c[quality_key] for c in candidates)
    eligible = [c for c in candidates if c[quality_key] >= best_quality - tolerance]
    chosen = min(eligible, key=lambda x: x["mean_seconds"])
    result = {"chosen_effort": chosen["effort"], "review_source": review_source,
            "rule": "完整开发集" + ("模型裁判" if review_source == "model" else "人工") + "正确率距最优不超过设定容差时选择平均耗时最短者",
            "quality_tolerance": tolerance, "candidates": candidates,
            "说明": "开发集选择为暂定结论；冻结配置后在保留测试集报告一次最终效果，费用未知时不能宣称最便宜。"}
    result["source_runs"] = [{"run": record["run"], "effort": record["manifest"]["identity"]["effort"],
                              **record.get("source_hashes", {})} for record in records]
    if review_source == "model":
        result.update(selection_status="模型裁判暂定推荐，未完成人工抽查", human_spot_check=None,
                      judge_configuration=shared_judge, judge_configuration_sha256=digest(shared_judge))
        result["说明"] += "本次只按完整模型答案正确率选优，不等于人工准确率；数值口径或引用支持的未知判断仍保留null，不把未知算成功。"
    return result


def main():
    parser = argparse.ArgumentParser(description="质量相差不超过 2 个百分点时优先选择更快的强度")
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--tolerance", type=float, default=0.02)
    parser.add_argument("--output", default="reports/effort_selection.json")
    parser.add_argument("--review-source", choices=("human", "model"), default="human",
                        help="默认要求完整人工复核；model仅产生六档完整39题的模型暂定推荐")
    args = parser.parse_args()
    records = []
    for run in args.runs:
        root = Path(run)
        manifest = read_json(root / "manifest.json")
        summary = read_json(root / "summary.json")
        record = {"run": run, "manifest": manifest, "summary": summary,
                  "source_hashes": {"manifest": digest((root / "manifest.json").read_bytes()),
                                    "results": digest((root / "results.jsonl").read_bytes())}}
        if args.review_source == "model":
            record.update(judge_manifest=read_json(root / "model_reviews.manifest.json"),
                          model_reviews=read_jsonl(root / "model_reviews.jsonl"),
                          generated_rows=read_jsonl(root / "results.jsonl"),
                          source_hashes={"manifest": digest((root / "manifest.json").read_bytes()),
                                         "results": digest((root / "results.jsonl").read_bytes()),
                                         "model_reviews_manifest": digest((root / "model_reviews.manifest.json").read_bytes()),
                                         "model_reviews": digest((root / "model_reviews.jsonl").read_bytes())})
        records.append(record)
    result = choose(records, args.tolerance, args.review_source)
    write_json(args.output, result)
    print(f"开发集暂选强度：{result['chosen_effort']}，结果已保存至 {args.output}")


if __name__ == "__main__":
    main()
