"""正式运行结束后离线汇总判断覆盖率、调用成本与复用来源；不调用模型。"""

import argparse
from collections import Counter
from pathlib import Path

from .common import digest, read_json, read_jsonl, write_json


def bound_reason(row, annotation):
    """整理摘要只适用于理由原文摘要完全匹配的这次裁判。"""
    source_hash = digest(row.get("judgement_reason"))
    matched = bool(annotation and annotation[0] == source_hash)
    return {"judgement_reason_sha256": source_hash, "summary_source_matched": matched,
            "reason_summary": annotation[1] if matched else "模型未能确定答案正确性，须查看私有运行记录人工复核。"}


def counts(rows, field):
    """逐字段统计原始判断；未知不能计为正确，也不从计划分母消失。"""
    values = [row.get(field) for row in rows]
    if any(value is not None and type(value) is not bool for value in values):
        raise ValueError("判断值必须为布尔或null")
    correct = sum(value is True for value in values)
    incorrect = sum(value is False for value in values)
    known = correct + incorrect
    return {"denominator": len(values), "true_count": correct, "false_count": incorrect,
            "null_count": len(values) - known, "known_count": known,
            "coverage": known / len(values),
            "known_subset_accuracy": correct / known if known else None,
            "full_denominator_accuracy": correct / len(values) if known == len(values) else None,
            "unknown_outcome_bounds": {"lower": correct / len(values),
                                       "upper": (correct + len(values) - known) / len(values)}
            if field == "answer_correct" else None}


def main():
    parser = argparse.ArgumentParser(description="仅对完成的正式R0结果进行离线汇总")
    parser.add_argument("--output", default="reports/r0-audit-summary.json")
    args = parser.parse_args()
    selection_path = Path("reports/effort_selection.json")
    selection = read_json(selection_path)
    effort = selection["chosen_effort"]
    run = Path("runs/all-selected") / effort
    source = Path("runs/dev-complete") / effort
    manifest = read_json(run / "manifest.json")
    judge_manifest = read_json(run / "model_reviews.manifest.json")
    final = read_json("reports/r0-final-summary.json")
    generated = read_jsonl(run / "results.jsonl")
    reviews = read_jsonl(run / "model_reviews.jsonl")
    queries = read_jsonl("data/financebench/queries.jsonl")
    query_ids = {row["id"] for row in queries}
    if (len(queries) != 150 or len(query_ids) != 150 or len(generated) != 150 or len(reviews) != 150
            or {row["id"] for row in generated} != query_ids or {row["id"] for row in reviews} != query_ids
            or any(row.get("status") != "ok" for row in generated)
            or any(row.get("judge_type") != "model" or row.get("judge_status") != "ok" for row in reviews)):
        raise ValueError("仅接受全150题完整生成与成功裁判；未知答案判断仍允许保留null")
    origin = manifest["identity"]["reuse"]
    judge_origin = judge_manifest["identity"]["reuse"]
    source_checks = {
        "generation_manifest": origin["manifest_sha256"] == digest((source / "manifest.json").read_bytes()),
        "generation_results": origin["results_sha256"] == digest((source / "results.jsonl").read_bytes()),
        "selection": origin["selection_sha256"] == digest(selection_path.read_bytes()),
        "judge_manifest": judge_origin["manifest_sha256"] == digest((source / "model_reviews.manifest.json").read_bytes()),
        "judge_results": judge_origin["reviews_sha256"] == digest((source / "model_reviews.jsonl").read_bytes()),
    }
    if not all(source_checks.values()) or any(sum("reuse_origin" in row for row in rows) != 39
                                            for rows in (generated, reviews)):
        raise ValueError("原39题生成或裁判来源被改动，或复用数不符")
    ledger_path = Path("runs/matrix-execution.jsonl")
    batches = read_jsonl(ledger_path)
    expected = Counter((stage, f"runs/dev-complete/{level}", 39)
                       for level in ("low", "medium", "high", "xhigh", "max", "ultra")
                       for stage in ("generation", "judge"))
    expected.update((stage, f"runs/all-selected/{effort}", 111) for stage in ("generation", "judge"))
    actual = Counter((row["stage"], Path(row["run"]).as_posix(), row["new_result_count"]) for row in batches)
    if actual != expected or any(row.get("verified") is not True or row.get("returncode") != 0 for row in batches):
        raise ValueError("正式14批审计不完整或未通过")
    by_id = {row["id"]: row for row in reviews}
    groups = {}
    for split, size in (("dev", 39), ("tune", 88), ("test", 23), ("all", 150)):
        ids = [row["id"] for row in queries if split == "all" or row["split"] == split]
        if len(ids) != size:
            raise ValueError("固定分组数量改变")
        group = [by_id[ident] for ident in ids]
        groups[split] = {"denominator": size,
                         "raw_field_judgements": {field: counts(group, field) for field in
                                                  ("answer_correct", "numeric_and_scope_correct", "citation_support")},
                         "original_score_summary": final["split_summaries"][split]}
    all_batch_tokens = sum(row["usage_total_tokens"] for row in batches)
    selected = [row for row in batches if row["effort"] == effort]
    # 只摘录模型所称的冲突类型，不公开答案/证据长文，也不认定金标确实错误。
    unknown_reasons = {
        "financebench_id_00540": ("5e9ada512f52613ac7cf53ef68720e4b07feae4542b00ccd81ab62b7a950b4b7", "模型认为存货周转率采用期末存货或平均存货的口径不一致，待人工复核。"),
        "financebench_id_05718": ("43800e1335407407ed4396c9199de7f4a9224d67226819b0d721aa9d1103d8c0", "模型认为标准答案与证据换算后的数值不一致；可能涉及近似或舍入，待人工复核。"),
        "financebench_id_04854": ("4d250c6fcd3e729cdd02e403cabe43ca2abdea010bcffcbd5995e4e9d1fe0f3b", "模型认为由证据计算的差额与标准答案不一致；可能涉及舍入精度，待人工复核。"),
        "financebench_id_00080": ("dba5632129212c154ff178466bd5cd441e67b6901a736fbe502489f32a2cca3c", "模型认为营运资本按标准证据计算与标准答案不一致，待人工复核。"),
        "financebench_id_00283": ("c9b0f6fd77cc730fbd079220b3c02352255bd778bd6acf6f04ea9db657fbf611", "模型认为重组剩余成本按比例计算与标准答案不一致，待人工复核。"),
    }
    summary = {
        "scope": "固定R0公开150题；含选档开发39题，独立保留测试为23题",
        "chosen_effort": effort, "selection_status": selection["selection_status"],
        "judge_type": "model", "human_spot_check": None,
        "answer_unknown_pending_review": [
            {"id": row["id"], **bound_reason(row, unknown_reasons.get(row["id"])),
             "reason_source": "固定模型裁判，不是已确认的数据错误", "human_review_completed": False}
            for row in reviews if row.get("answer_correct") is None],
        "groups": groups,
        "说明": ["逐字段计数来自成功裁判原始布尔/null；null可能是非数值、不适用、证据不足或标准材料冲突。",
                 "known_subset_accuracy仅限已判定子集，须同时读取known_count和coverage；不是全量准确率。",
                 "unknown_outcome_bounds仅给未知项最终全错至全对的算术界限，不是置信区间，也没有强行赋值。",
                 "原score的数值/引用覆盖率还要求该题answer_correct为布尔；original_score_summary保留该既有门槛，逐字段原始计数不更改评分或选档。",
                 "页命中与标准文本匹配均为独立诊断检索的代理，不是语义证据召回或问答实际上下文召回。",
                 "模型批次时间/调用量不含部署、烟测、编程代理或人工整理成本；金额未知。"],
        "formal_model_calls": {
            "batches": len(batches), "planned_successful_calls": 690,
            "actual_attempts": sum(row["actual_attempts"] for row in batches),
            "successful_calls": sum(row["successful_model_calls"] for row in batches),
            "failed_attempts": sum(row["failed_attempts"] for row in batches),
            "total_tokens_including_failures": all_batch_tokens,
            "failed_attempt_tokens": sum((record.get("usage") or {}).get("totalTokens", 0)
                                          for row in batches for record in row["records"] if record.get("success") is not True),
            "selected_150_including_source_39_calls": sum(row["actual_attempts"] for row in selected),
            "selected_150_including_source_39_tokens": sum(row["usage_total_tokens"] for row in selected),
            "cost_usd": None},
        "wall_seconds": {key: value for key, value in final.items() if key.endswith("wall_seconds")},
        "source_39_unchanged": source_checks,
        "source_files": {str(path): digest(path.read_bytes()) for path in
                         (selection_path, run / "manifest.json", run / "results.jsonl",
                          run / "model_reviews.manifest.json", run / "model_reviews.jsonl",
                          source / "manifest.json", source / "results.jsonl",
                          source / "model_reviews.manifest.json", source / "model_reviews.jsonl",
                          ledger_path, Path("reports/r0-final-summary.json"), Path(__file__))},
    }
    write_json(args.output, summary)
    print("正式R0离线汇总已保存：" + args.output)


if __name__ == "__main__":
    main()
