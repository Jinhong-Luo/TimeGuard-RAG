"""冻结历史评测，选取可重现的分层复核样本；两种旧编号解释均只作排查线索。"""
import argparse
from collections import Counter
from pathlib import Path

from .common import digest, read_jsonl, write_json, write_jsonl
from .judge import citation_context, unique_rows


def sampled(rows, reviews, size):
    chosen = []
    groups = {}
    for row in sorted(rows, key=lambda r: digest(r["id"])):
        review = reviews[row["id"]]
        if review["answer_correct"] is None:
            chosen.append(row)
        else:
            key = (review["answer_correct"], bool(citation_context(row)["unmapped_indexes"]))
            groups.setdefault(key, []).append(row)
    if len(chosen) > size or len(rows) < size:
        raise ValueError("样本数无法覆盖未知题或超出题数")
    while len(chosen) < size:
        for group in groups.values():
            if group and len(chosen) < size:
                chosen.append(group.pop())
    return chosen


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("--gold", default="data/financebench/gold.jsonl")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--size", type=int, default=25)
    args = p.parse_args()
    source = args.run / "results.jsonl"
    reviews_path = args.run / "model_reviews.jsonl"
    rows = read_jsonl(source)
    reviews = unique_rows(read_jsonl(reviews_path), "历史裁判")
    gold = unique_rows(read_jsonl(args.gold), "金标")
    if unique_rows(rows, "历史回答").keys() != reviews.keys():
        raise ValueError("历史回答和裁判集合不同")
    args.output.mkdir(parents=True, exist_ok=False)
    packet = []
    for row in sampled(rows, reviews, args.size):
        ident = row["id"]
        chunks = row["reference"].get("chunks", [])
        nonempty = [c for c in chunks if c.get("content")]
        context = citation_context(row)
        alternatives = []
        for i in context["marker_indexes"]:
            alternatives.append({"marker": i, "旧裁判位置": chunks[i]["id"] if i < len(chunks) else None,
                                 "假定显式且未裁剪": nonempty[i-1]["id"] if 1 <= i <= len(nonempty) else None})
        packet.append({"id": ident, "question": row["question"], "answer": row["answer"],
                       "reference_candidates": chunks, "historical_judge": reviews[ident],
                       "reference_answer": gold[ident]["answer"], "reference_evidence": gold[ident]["evidence"],
                       "mapping_hypotheses": alternatives, "actual_prompt": None, "citation_origin": "未知",
                       "human_reviewer": None, "gold_sufficient": None, "answer_correct": None,
                       "citation_support": None, "judge_reasonable": None, "notes": None})
    write_jsonl(args.output / "review.jsonl", packet)
    write_json(args.output / "summary.json", {"source_results_sha256": digest(source.read_bytes()),
        "source_reviews_sha256": digest(reviews_path.read_bytes()), "gold_sha256": digest(Path(args.gold).read_bytes()),
        "denominator": len(rows), "legacy_status": dict(Counter(citation_context(r)["mapping_status"] for r in rows)),
        "sample_size": len(packet), "unknown_in_sample": sum(r["historical_judge"]["answer_correct"] is None for r in packet),
        "sample_ids": [r["id"] for r in packet], "human_reviews": 0,
        "说明": "旧记录缺少模型原始输出及最终提示块轨迹，备选映射不构成已纠正的分数。"})


if __name__ == "__main__":
    main()

