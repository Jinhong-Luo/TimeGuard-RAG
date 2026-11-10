"""在已冻结的组合组候选中配对重排，并实查零检索题的原范围。"""
import argparse
import json
from pathlib import Path
import time
import urllib.request

from .common import Client, digest, read_json, read_jsonl, write_json
from .score import evidence_metrics


def rerank(question, chunks, url):
    data = {"query": question, "texts": [c["content"] for c in chunks],
            "truncate": True, "raw_scores": False}
    request = urllib.request.Request(url, json.dumps(data).encode(),
                                     headers={"Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=120) as response:
        scores = json.load(response)
    if sorted(s["index"] for s in scores) != list(range(len(chunks))):
        raise ValueError("重排结果与候选不对应")
    if not all(0 <= s["score"] <= 1 for s in scores):
        raise ValueError("概率分数超出区间")
    return sorted(scores, key=lambda s: (-s["score"], s["index"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rerank-url", required=True)
    args = parser.parse_args()
    report = read_json("reports/native-baseline-20261109.json")
    source = args.source / "results.jsonl"
    expected = next(r for r in report["core_runs"] if r["run"] == "combined-core-dev")
    if digest(source.read_bytes()) != expected["results_sha256"]:
        raise ValueError("来源与组合组冻结摘要不符")
    args.output.mkdir(parents=True, exist_ok=False)
    rows = read_jsonl(source)
    config = read_json("configs/r0-low.json")
    dataset = read_json("data/ingestion.json")["dataset_id"]
    client = Client(config["base_url"])
    gold = {r["id"]: r for r in read_jsonl("data/financebench/gold.jsonl")}
    records = []
    for row in rows:
        trace = row["native_trace"]["events"]
        allowed = next(e["selected_doc_ids"] for e in trace if e["kind"] == "metadata_filter")
        item = {"id": row["id"], "fallback_calls": 0, "rerank_calls": 0}
        records.append(item)
        chunks = (row["reference"].get("chunks") or [])[:12]
        item["pool"] = len(chunks)
        try:
            if allowed and allowed != ["-999"] and not any(e["kind"] == "retrieval" for e in trace) and not chunks:
                item["fallback_calls"] = 1
                found = client.request("POST", "/retrieval", {**config["retrieval"], "question": row["question"],
                    "document_ids": allowed, "dataset_ids": [dataset]})["chunks"]
                if any(c["document_id"] not in allowed for c in found):
                    raise ValueError("补查越界")
                item.update(fallback_chunks=len(found), fallback_recall=evidence_metrics(gold[row["id"]], found, 6)["page_hit_recall"])
            if chunks:
                started = time.monotonic()
                item["rerank_calls"] = 1
                scores = rerank(row["question"], chunks, args.rerank_url)
                item.update(max_score=scores[0]["score"], min_score=scores[-1]["score"],
                            seconds=time.monotonic()-started)
                ranked = [chunks[s["index"]] for s in scores]
            else:
                ranked = []
            item.update(original=evidence_metrics(gold[row["id"]], chunks, 6)["page_hit_recall"],
                        reranked=evidence_metrics(gold[row["id"]], ranked, 6)["page_hit_recall"])
        except Exception as exc:
            item["error"] = type(exc).__name__ + ": " + str(exc)[:200]
        write_json(args.output / "summary.json", {"source_sha256": expected["results_sha256"], "planned": len(rows),
            "说明": "以返回引用候选的前十二块为固定池，只改变排序，取前六块计算金标页召回；补查单列。", "records": records})


if __name__ == "__main__":
    main()

