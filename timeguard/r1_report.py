"""核对计划、逐题结果和模型桥字节摘要，生成可公开的对照汇总。"""
import argparse
from collections import Counter
from pathlib import Path
from statistics import mean, median

from .common import digest, read_json, read_jsonl, write_json
from .judge import citation_context, unique_rows
from .scope import YEAR
from .score import doc_name


def counts(values):
    return dict(Counter("是" if x is True else "否" if x is False else "未知" for x in values))


def retrieval_detail(path, variants):
    gold = unique_rows(read_jsonl("data/financebench/gold.jsonl"), "离线金标")
    for i, variant in enumerate(variants):
        file = path / f"{i}.jsonl"
        if digest(file.read_bytes()) != variant["rows_sha256"]:
            raise ValueError("检索结果与汇总摘要不一致")
        counts_by_period = Counter()
        first_page = first_doc = 0
        for row in read_jsonl(file):
            evidence = gold[row["id"]]["evidence"]
            chunks = row["retrieved_chunks"]
            if chunks:
                name = doc_name(chunks[0])
                matching = [e for e in evidence if e["doc_name"].casefold() == name]
                first_doc += bool(matching)
                pages = {p[0]-1 for p in chunks[0].get("positions", []) if p}
                first_page += any(e["evidence_page_num"] in pages for e in matching)
            years = set(row["scope"].get("years", []))
            if years:
                for chunk in chunks:
                    seen = set(map(int, YEAR.findall(chunk["content"])))
                    label = "含目标年份" if seen & years else "仅含其他年份" if seen else "无明示年份"
                    counts_by_period[label] += 1
        variant.update(top1_document_hits=first_doc, top1_evidence_page_hits=first_page,
                       year_mentions=dict(counts_by_period))


def summarize_run(path):
    manifest = read_json(path / "manifest.json")
    ids = manifest["identity"]["query_ids"]
    rows = unique_rows(read_jsonl(path / "results.jsonl"), "回答")
    reviews = unique_rows(read_jsonl(path / "model_reviews.jsonl"), "裁判")
    batches = read_jsonl(path / "audit.jsonl")
    if set(rows) != set(ids) or set(reviews) != set(ids) or len(set(ids)) != len(ids):
        raise ValueError("运行未覆盖计划")
    if len(batches) != 2 or any(b["stage"] != stage or not b["verified"] or b["returncode"] or
            b["results_sha256"] != digest((path / file).read_bytes()) or b["new_result_count"] != len(ids)
            for b, stage, file in zip(batches, ("generation", "judge"), ("results.jsonl", "model_reviews.jsonl"))):
        raise ValueError("生成或裁判审计与当前结果文件不符")
    elapsed = [r["elapsed_seconds"] for r in rows.values()]
    mapping = [citation_context(r) for r in rows.values()]
    correct = {i: reviews[i]["answer_correct"] for i in ids}
    return {"run": path.name, "planned": len(ids), "answers": counts(correct.values()),
            "citations_supported": counts(r["citation_support"] for r in reviews.values()),
            "mapping": dict(Counter(c["mapping_status"] for c in mapping)),
            "citation_origins": dict(Counter(c.get("citation_origin") for c in mapping)),
            "mean_seconds": mean(elapsed), "median_seconds": median(elapsed), "max_seconds": max(elapsed),
            "prompt_tokens_estimate_mean": mean(r["prompt_tokens_estimate"] for r in rows.values()
                       if r.get("prompt_tokens_estimate") is not None) if any(r.get("prompt_tokens_estimate") for r in rows.values()) else None,
            "actual_attempts": sum(b["actual_attempts"] for b in batches), "failed_attempts": sum(b["failed_attempts"] for b in batches),
            "tokens": sum(b["usage_total_tokens"] for b in batches), "manifest_sha256": digest((path/"manifest.json").read_bytes()),
            "results_sha256": batches[0]["results_sha256"], "reviews_sha256": batches[1]["results_sha256"]}, correct


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--native", type=Path, nargs="+", required=True)
    p.add_argument("--oracle", type=Path)
    p.add_argument("--r0", type=Path, default=Path("runs/dev-complete/low"))
    p.add_argument("--fixed-pool", type=Path)
    p.add_argument("--filing", type=Path)
    p.add_argument("--table-answers", type=Path, nargs=2)
    p.add_argument("--app-check", type=Path)
    p.add_argument("--extra-retrieval", type=Path, nargs="*", default=[])
    p.add_argument("--incident-runs", type=Path, nargs="*", default=[])
    p.add_argument("--tune", type=Path, required=True)
    p.add_argument("--table", type=Path, required=True)
    p.add_argument("--review", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("请使用新的汇总文件")
    report = {"native": [], "paired_to_first": {}, "tune": read_json(args.tune/"summary.json"),
              "table": read_json(args.table/"summary.json"), "historical_review": read_json(args.review/"summary.json"),
              "human_reviews": 0, "cost_usd": None}
    retrieval_detail(args.tune, report["tune"]["variants"])
    report["extra_retrieval"] = {}
    for run in args.extra_retrieval:
        summary = read_json(run / "summary.json")
        retrieval_detail(run, summary["variants"])
        report["extra_retrieval"][run.name] = summary
    report["incident_runs"] = {}
    for run in args.incident_runs:
        audit = run / "audit.jsonl"
        report["incident_runs"][run.name] = {
            "audit_sha256": digest(audit.read_bytes()),
            "planned_questions": len(read_json(run/"manifest.json")["identity"]["query_ids"]),
            "recorded_answers": len(read_jsonl(run/"results.jsonl")),
            "stages": [{k: b[k] for k in ("stage", "verified", "actual_attempts", "successful_model_calls",
                       "failed_attempts", "usage_total_tokens", "returncode")} for b in read_jsonl(audit)]}
    baseline = None
    for path in args.native:
        stats, current = summarize_run(path)
        report["native"].append(stats)
        if baseline is None:
            baseline = current
        else:
            if current.keys() != baseline.keys():
                raise ValueError("开发集题目不同，不可配对")
            report["paired_to_first"][path.name] = {
                "wrong_to_correct": [i for i in baseline if baseline[i] is False and current[i] is True],
                "correct_to_wrong": [i for i in baseline if baseline[i] is True and current[i] is False],
                "unknown_in_either": [i for i in baseline if baseline[i] is None or current[i] is None]}
    if args.oracle:
        report["oracle_only"] = summarize_run(args.oracle)[0]
    for field, root in (("fixed_pool", args.fixed_pool), ("filing_dates", args.filing), ("app_check", args.app_check)):
        if root:
            report[field] = read_json(root / "summary.json")
    if args.table_answers:
        report["table_answers_only"] = [summarize_run(root)[0] for root in args.table_answers]
    original = unique_rows(read_jsonl(args.r0 / "model_reviews.jsonl"), "旧开发集裁判")
    if original.keys() != baseline.keys():
        raise ValueError("旧开发集与修正实验的题目不同")
    report["historical_dev"] = {"planned": len(original), "answers": counts(r["answer_correct"] for r in original.values()),
        "mapping": dict(Counter(r["citation_mapping"]["mapping_status"] for r in original.values())),
        "reviews_sha256": digest((args.r0 / "model_reviews.jsonl").read_bytes())}
    write_json(args.output, report)


if __name__ == "__main__":
    main()

