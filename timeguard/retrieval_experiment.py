"""在冻结索引上运行独立检索对照，再离线读取金标评分。"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import json
from pathlib import Path
import time

from .common import Client, digest, read_json, read_jsonl, write_json
from .scope import scope, search
from .score import doc_name, evidence_metrics, normal


def catalog(ledger, state):
    dates = read_json(Path(__file__).resolve().parents[1] / "configs/filing_dates.json")
    docs = []
    for doc in ledger:
        filing = dates.get(doc["doc_name"], {})
        if filing:
            source = state["documents"][doc["doc_name"]]
            matched = (filing["pdf_sha256"] == source.get("sha256") if filing.get("pdf_sha256") else
                       bool(filing.get("accession")) and filing["accession"] in doc["doc_link"])
            if not matched:
                raise ValueError("披露日期与原始 PDF 指纹或申报登记号不匹配")
        docs.append({**doc, **filing, "id": state["documents"][doc["doc_name"]]["id"]})
    return docs


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
    p.add_argument("--arm", choices=list("ABCD"), help="只复验指定组的 6 候选、0.3 权重配置")
    p.add_argument("--native-pool", type=int, help="原生检索内部排序的候选数；对照两组须一致")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = read_json(args.config)
    if args.native_pool is not None:
        if args.native_pool < (6 if args.arm in ("A", "B", "C") else 24):
            p.error("内部候选池小于所选方案需要的返回数")
        config["retrieval"]["rerank_candidates_count"] = args.native_pool
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
    if args.arm:
        variants = [(args.arm, 6, .3, args.reranker or "")]
    elif args.reranker:
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
            row = {"id": q["id"], "scope": {}, "retrieved_chunks": [], "status": "ok"}
            try:
                plan = row["scope"] = scope(q["question"], documents)
                row["retrieved_chunks"], row["path"] = search(client, config, q["question"], plan, arm, docs, k, w, reranker)
            except Exception as exc:
                row.update(status="request_error", error=str(exc)[:400])
            row["elapsed_seconds"] = time.perf_counter() - start
            return row
        statuses = Counter()
        with (args.output / f"{i}.jsonl").open("x", encoding="utf-8") as output, ThreadPoolExecutor(max_workers=2) as pool:
            for row in pool.map(one, queries):
                output.write(json.dumps(row, ensure_ascii=False)+"\n")
                output.flush()
                statuses[row["status"]] += 1
        print(i, arm, k, w, statuses, flush=True)
    gold = {g["id"]: g for g in read_jsonl(args.gold)}
    reports = []
    for i, (arm, k, w, reranker) in enumerate(variants):
        path = args.output / f"{i}.jsonl"
        reports.append({"arm": arm, "k": k, "weight": w, "reranker": reranker,
                        "rows_sha256": digest(path.read_bytes()), **summarize(read_jsonl(path), gold, docs)})
    write_json(args.output / "summary.json", {"manifest_sha256": digest(identity), "variants": reports})


if __name__ == "__main__":
    main()

