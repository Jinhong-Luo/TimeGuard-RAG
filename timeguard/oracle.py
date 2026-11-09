"""标准证据上限诊断，单独调用模型，无检索，结果禁止计入正常 RAG 分数。"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import time

from .common import digest, read_json, read_jsonl, write_json
from .judge import MODEL, post_completion
from .matrix import run_batch


def one(g, endpoint):
    start = time.monotonic()
    chunks = g["chunks"]
    blocks = "\n\n".join(f"ID: {i}\nTitle: {c['document_name']}\n{c['content']}" for i, c in enumerate(chunks))
    row = {"id": g["id"], "split": g["split"], "question": g["question"], "reference": {"chunks": chunks},
           "retrieved_chunks": [], "status": "ok", "cost_usd": None}
    try:
        response = post_completion(endpoint, {"model": MODEL + "@low", "reasoning_effort": "low",
            "messages": [{"role": "system", "content": "用问题的语言回答，逐项引用所用资料 [ID:i]。给出计算过程，缺信息时说明。\n" + blocks},
                         {"role": "user", "content": g["question"]}]}, os.environ["CODEX_BRIDGE_TOKEN"], 360)
        answer = response["choices"][0]["message"]["content"]
        if not answer:
            raise ValueError("上限诊断模型返回空答案")
        row.update(answer=answer, usage=response.get("usage"), citation_trace={"version": 1, "origin": "model",
            "raw_answer": answer, "visible": {str(i): {"chunk_id": c["id"], "document_id": c["document_id"],
            "content_sha256": digest(c["content"])} for i, c in enumerate(chunks)}})
    except Exception as exc:
        row.update(status="request_error", error=str(exc)[:400])
    row["elapsed_seconds"] = time.monotonic()-start
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--worker", action="store_true")
    p.add_argument("--table-trial", type=Path, help="改用真实表格检索的父块做独立回答诊断")
    p.add_argument("--table-arm", choices=("whole", "rows"))
    args = p.parse_args()
    if bool(args.table_trial) != bool(args.table_arm):
        p.error("表格检索目录与实验组须同时指定")
    if args.table_trial:
        source = args.table_trial / (args.table_arm + ".jsonl")
        qs = {q["id"]: q for q in read_jsonl("data/financebench/queries.jsonl")}
        parents = {c["id"]: c for c in read_jsonl(args.table_trial / "parents.jsonl")}
        selected = [{**qs[row["id"]], "chunks": [{**parents[i], "document_name": parents[i]["docnm_kwd"]}
                     for i in row["parents"]]} for row in read_jsonl(source)]
        mode = "真实表格检索父块的模型诊断"
        worker_options = ["--table-trial", args.table_trial, "--table-arm", args.table_arm]
    else:
        source = Path("data/financebench/gold.jsonl")
        selected = [{k: g[k] for k in ("id", "split", "question")} | {"chunks": [
            {"id": f"oracle-{g['id']}-{i}", "document_id": e["doc_name"], "document_name": e["doc_name"],
             "content": e["evidence_text"], "positions": [[e["evidence_page_num"]+1]]}
            for i, e in enumerate(g["evidence"])]} for g in read_jsonl(source) if g["split"] == "dev"]
        mode = "标准证据上限诊断，无原生检索"
        worker_options = []
    out = args.output
    endpoint = os.environ.get("CODEX_BRIDGE_BASE_URL", "http://172.18.0.1:18864/v1")
    if args.worker:
        with (out / "results.jsonl").open("x") as handle, ThreadPoolExecutor(max_workers=2) as pool:
            for i in range(0, len(selected), 2):
                batch = list(pool.map(lambda g: one(g, endpoint), selected[i:i+2]))
                for row in batch:
                    handle.write(json.dumps(row, ensure_ascii=False)+"\n")
                handle.flush()
                if any(r["status"] != "ok" for r in batch):
                    raise SystemExit(1)
        return
    out.mkdir(parents=True, exist_ok=False)
    identity = {"scope": mode, "query_ids": [g["id"] for g in selected], "material_sha256": digest(selected),
                "input_sha256": digest(source.read_bytes()), "source_sha256": digest(Path(__file__).read_bytes())}
    write_json(out / "manifest.json", {"identity": identity})
    audit, ledger = Path(".cache/bridge-audit.jsonl"), out / "audit.jsonl"
    run_batch("timeguard.oracle", ["--output", out, "--worker", *worker_options], out, "generation", "low", audit, ledger, identity)
    run_batch("timeguard.judge", [out, "--concurrency", "2", "--base-url", endpoint, "--execute"], out,
              "judge", "medium", audit, ledger, identity)


if __name__ == "__main__":
    main()

