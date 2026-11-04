"""离线评分：失败和零召回留在分母中；语义正确性由独立人工标注。"""

import argparse
import collections
import re
import statistics
from pathlib import Path

from .common import read_json, read_jsonl, write_json, write_jsonl


def normal(value):
    return "".join(re.findall(r"\w+", str(value).casefold()))


def doc_name(chunk):
    name = chunk.get("document_name", chunk.get("docnm_kwd", ""))
    return Path(name).stem.casefold()


def evidence_metrics(gold, chunks, k):
    chunks = chunks[:k]
    evidence = gold.get("evidence", [])
    if not evidence:
        return {"page_hit_recall": None, "strict_text_recall": None}
    page_hits = text_hits = 0
    for e in evidence:
        same_doc = [c for c in chunks if doc_name(c) == e["doc_name"].casefold()]
        # FinanceBench 页号从 0 开始，RAGFlow positions 页号从 1 开始。
        pages = {int(p[0]) - 1 for c in same_doc for p in c.get("positions", []) if isinstance(p, list) and p}
        page_hits += e["evidence_page_num"] in pages
        content = normal(" ".join(c.get("content", c.get("content_with_weight", "")) for c in same_doc))
        expected = normal(e["evidence_text"])
        text_hits += bool(expected) and expected in content
    return {"page_hit_recall": page_hits / len(evidence), "strict_text_recall": text_hits / len(evidence)}


def summarize(expected_ids, gold, rows, k=6, reviews=None):
    if not expected_ids:
        raise ValueError("没有待评测题目")
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("结果含重复题号；不得隐式选用最后一次成功结果")
    unexpected = {r["id"] for r in rows} - set(expected_ids)
    if unexpected:
        raise ValueError(f"结果存在清单以外的题目：{sorted(unexpected)}")
    by_id = {r["id"]: r for r in rows}
    scored, review_rows = [], []
    for ident in expected_ids:
        row = by_id.get(ident, {"id": ident, "status": "missing", "retrieved_chunks": []})
        metrics = evidence_metrics(gold[ident], row.get("retrieved_chunks", []), k)
        scored.append({"id": ident, "status": row["status"], **metrics})
        review_rows.append({"id": ident, "question": gold[ident]["question"], "answer": row.get("answer"),
                            "gold_answer": gold[ident]["answer"], "reference": row.get("reference", {}),
                            "answer_correct": None, "numeric_and_scope_correct": None,
                            "citation_support": None, "failure_category": None, "reviewer": None})
    n = len(expected_ids)
    report = {"denominator": n, "completed": len(rows), "status_counts": dict(collections.Counter(r["status"] for r in scored)),
              "zero_retrieval_count": sum(not by_id.get(i, {}).get("retrieved_chunks") for i in expected_ids),
              "page_hit_recall_at_k_proxy": sum(r["page_hit_recall"] or 0 for r in scored) / n,
              "strict_text_recall_at_k_proxy": sum(r["strict_text_recall"] or 0 for r in scored) / n,
              "k": k, "cost_usd": None, "answer_accuracy": None, "numeric_and_scope_accuracy": None,
              "citation_support_rate": None, "reviewed": 0,
              "说明": "页命中与标准证据全文归一化匹配均为代理指标，不能等同于语义证据支持率；费用未返回时保留为空。"}
    durations = [r["elapsed_seconds"] for r in rows if r.get("elapsed_seconds") is not None]
    report["mean_seconds"] = statistics.mean(durations) if durations else None
    report["median_seconds"] = statistics.median(durations) if durations else None
    if reviews is not None:
        if len({r["id"] for r in reviews}) != len(reviews) or {r["id"] for r in reviews} - set(expected_ids):
            raise ValueError("人工复核题号重复或超出运行集合")
        review_map = {r["id"]: r for r in reviews}
        valid = [review_map[i] for i in expected_ids if i in review_map and
                 isinstance(review_map[i].get("answer_correct"), bool) and review_map[i].get("reviewer")]
        report["reviewed"] = len(valid)
        # 部分复核不能伪装成完整准确率。
        if len(valid) == n:
            report["answer_accuracy"] = sum(r["answer_correct"] for r in valid) / n
            for source, target in [("numeric_and_scope_correct", "numeric_and_scope_accuracy"), ("citation_support", "citation_support_rate")]:
                values = [r[source] for r in valid if isinstance(r.get(source), bool)]
                if len(values) == n:
                    report[target] = sum(values) / n
    return report, scored, review_rows


def main():
    parser = argparse.ArgumentParser(description="在运行结束后单独读取金标并评分")
    parser.add_argument("run")
    parser.add_argument("--gold", default="data/financebench/gold.jsonl")
    parser.add_argument("--reviews")
    args = parser.parse_args()
    root = Path(args.run)
    manifest = read_json(root / "manifest.json")["identity"]
    gold = {r["id"]: r for r in read_jsonl(args.gold)}
    report, scored, review_rows = summarize(manifest["query_ids"], gold, read_jsonl(root / "results.jsonl"),
                                           manifest["config"]["retrieval"]["page_size"],
                                           read_jsonl(args.reviews) if args.reviews else None)
    write_json(root / "summary.json", report)
    write_jsonl(root / "scored.jsonl", scored)
    review_template = root / "review_template.jsonl"
    if not review_template.exists():
        write_jsonl(review_template, review_rows)
    print(__import__("json").dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
