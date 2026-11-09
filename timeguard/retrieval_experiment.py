"""在冻结索引上运行独立检索对照，再离线读取金标评分。"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from pathlib import Path
import time

from .common import Client, digest, read_json, read_jsonl, write_json, write_jsonl
from .scope import scope, search
from .score import doc_name, evidence_metrics, normal


def catalog(ledger, state):
    return [{**d, "id": state["documents"][d["doc_name"]]["id"]} for d in ledger]


def summarize(rows, gold, docs):
    n = len(rows)
    if not n or len({r["id"] for r in rows}) != n:
        raise ValueError("缺少实验题或题号重复")
    totals = Counter()
    for r in rows:
        g = gold[r["id"]]
        chunks = r["retrieved_chunks"]
        names = [doc_name(c) for c in chunks]
        totals["document_hits"] += bool(set(names) & {e["doc_name"].casefold() for e in g["evidence"]})
        totals["company_hits"] += any(docs.get(name, {}).get("company") == g["company"] for name in names)
        totals["wrong_company_chunks"] += sum(docs.get(name, {}).get("company") != g["company"] for name in names)
        contents = [normal(c.get("content", "")) for c in chunks]
        totals["duplicate_chunks"] += len(contents) - len(set(contents))
        totals["chunks"] += len(chunks)
        totals["empty"] += not chunks
        totals["errors"] += r["status"] != "ok"
        metrics = evidence_metrics(g, chunks, len(chunks))
        totals["page_recall_sum"] += metrics["page_hit_recall"] or 0
        totals["text_recall_sum"] += metrics["strict_text_recall"] or 0
        totals["seconds"] += r["elapsed_seconds"]
    return {"denominator": n, **totals, "page_recall": totals["page_recall_sum"] / n,
            "strict_text_recall": totals["text_recall_sum"] / n, "mean_seconds": totals["seconds"] / n}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--config", default="configs/r0-low.json")
    p.add_argument("--queries", default="data/financebench/queries.jsonl")
    p.add_argument("--documents", default="data/financebench/documents.jsonl")
    p.add_argument("--ingestion", default="data/ingestion.json")
    p.add_argument("--gold", default="data/financebench/gold.jsonl")
    p.add_argument("--split", choices=("tune", "dev"), default="tune")
    p.add_argument("--reranker", help="已实际注册的原生重排模型")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = read_json(args.config)
    state = read_json(args.ingestion)
    config["dataset_ids"] = [state["dataset_id"]]
    documents = catalog(read_jsonl(args.documents), state)
    docs = {d["doc_name"].casefold(): d for d in documents}
    queries = [q for q in read_jsonl(args.queries) if q["split"] == args.split]
    client = Client(config["base_url"])
    live = client.request("GET", f"/datasets/{state['dataset_id']}/documents?page=1&page_size=100")["docs"]
    if {d["id"] for d in live} != {d["id"] for d in documents} or any(
            d["run"] not in ("DONE", "3", 3) or not d["chunk_count"] for d in live):
        raise ValueError("索引未就绪或语料目录不一致")
    variants = [(a, 6, .3, "") for a in "ABCD"]
    variants.extend(("B", k, w, "") for k, w in [(12, .3), (24, .3), (6, .5), (6, .7)])
    if args.reranker:
        variants.append(("B", 6, .3, args.reranker))
    identity = {"split": args.split, "query_ids": [q["id"] for q in queries], "config": config,
                "query_sha256": digest(Path(args.queries).read_bytes()), "catalog_sha256": digest(documents),
                "corpus_sha256": state["corpus_sha256"], "index_snapshot_sha256": digest(sorted(
                    [{k: d.get(k) for k in ("id", "name", "size", "chunk_count", "update_time")} for d in live],
                    key=lambda d: d["id"])), "variants": variants,
                "source_sha256": {p: digest(Path(p).read_bytes()) for p in
                                  ("timeguard/scope.py", "timeguard/retrieval_experiment.py")}}
    write_json(args.output / "manifest.json", identity)
    for i, (arm, k, w, reranker) in enumerate(variants):
        def one(q):
            start = time.perf_counter()
            plan = scope(q["question"], documents)
            row = {"id": q["id"], "scope": plan, "retrieved_chunks": [], "status": "ok"}
            try:
                row["retrieved_chunks"], row["path"] = search(client, config, q["question"], plan, arm, docs, k, w, reranker)
            except Exception as exc:
                row.update(status="request_error", error=str(exc)[:400])
            row["elapsed_seconds"] = time.perf_counter() - start
            return row
        with ThreadPoolExecutor(max_workers=2) as pool:
            rows = list(pool.map(one, queries))
        write_jsonl(args.output / f"{i}.jsonl", rows)
        print(i, arm, k, w, Counter(r["status"] for r in rows), flush=True)
    gold = {g["id"]: g for g in read_jsonl(args.gold)}
    reports = []
    for i, (arm, k, w, reranker) in enumerate(variants):
        path = args.output / f"{i}.jsonl"
        reports.append({"arm": arm, "k": k, "weight": w, "reranker": reranker,
                        "rows_sha256": digest(path.read_bytes()), **summarize(read_jsonl(path), gold, docs)})
    write_json(args.output / "summary.json", {"manifest_sha256": digest(identity), "variants": reports})


if __name__ == "__main__":
    main()

