"""仅在相同开发集和完整复核结果上选择思考强度。"""

import argparse
import copy
import math
from pathlib import Path

from .common import digest, read_json, write_json


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


def choose(records, tolerance=0.02):
    """只接收完整开发集、完整候选集合及冻结索引的运行记录。"""
    if not math.isfinite(tolerance) or not 0 <= tolerance <= 1:
        raise ValueError("质量容差必须是 0 到 1 之间的有限数")
    if len(records) < 2:
        raise ValueError("至少需要两个不同思考强度，不能单档自称最优")
    candidates, shared, shared_snapshot, efforts = [], None, None, set()
    for record in records:
        complete_manifest, summary = record["manifest"], record["summary"]
        identity = complete_manifest["identity"]
        ids = identity["query_ids"]
        if identity["split"] != "dev" or identity["retrieval_only"]:
            raise ValueError("只能使用开发集完整问答运行选强度，不能使用保留测试集")
        if (identity.get("scope") != "完整分组评测" or not ids
                or len(ids) != len(set(ids)) or len(ids) != identity.get("full_split_count")
                or digest(ids) != identity.get("full_split_query_ids_sha256")):
            raise ValueError("运行未覆盖完整开发集或缺少范围证明；连通测试不能参与选优")
        if (summary["completed"] != len(ids) or summary["denominator"] != len(ids)
                or summary["answer_accuracy"] is None or summary.get("reviewed") != len(ids)):
            raise ValueError("每档运行和人工答案复核必须全部完成")
        if any(count for status, count in summary.get("status_counts", {}).items() if status != "ok"):
            raise ValueError("候选存在请求、生成或缺失错误，不能据此比较模型思考强度")
        duration, accuracy = summary.get("mean_seconds"), summary["answer_accuracy"]
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
    best_quality = max(c["answer_accuracy"] for c in candidates)
    eligible = [c for c in candidates if c["answer_accuracy"] >= best_quality - tolerance]
    chosen = min(eligible, key=lambda x: x["mean_seconds"])
    return {"chosen_effort": chosen["effort"], "rule": "完整开发集人工正确率距最优不超过设定容差时选择平均耗时最短者",
            "quality_tolerance": tolerance, "candidates": candidates,
            "说明": "开发集选择为暂定结论；冻结配置后在保留测试集报告一次最终效果，费用未知时不能宣称最便宜。"}


def main():
    parser = argparse.ArgumentParser(description="质量相差不超过 2 个百分点时优先选择更快的强度")
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--tolerance", type=float, default=0.02)
    parser.add_argument("--output", default="reports/effort_selection.json")
    args = parser.parse_args()
    records = []
    for run in args.runs:
        root = Path(run)
        manifest = read_json(root / "manifest.json")
        summary = read_json(root / "summary.json")
        records.append({"run": run, "manifest": manifest, "summary": summary})
    result = choose(records, args.tolerance)
    write_json(args.output, result)
    print(f"开发集暂选强度：{result['chosen_effort']}，结果已保存至 {args.output}")


if __name__ == "__main__":
    main()
