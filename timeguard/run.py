"""R0 原生检索与问答运行器；不读取标准答案或标准证据。"""

import argparse
import datetime
import json
import time
from pathlib import Path

from .common import Client, digest, read_json, read_jsonl, write_json


def retrieval_body(config, query):
    # 只允许问题文本进入检索；禁止用金标文档、期间或证据反向约束 R0。
    return {**config["retrieval"], "dataset_ids": config["dataset_ids"], "question": query["question"]}


def chat_body(config, effort, name):
    setting = dict(config.get("llm_setting", {}))
    if effort != "default":
        setting["reasoning_effort"] = effort
    r = config["retrieval"]
    return {"name": name, "dataset_ids": config["dataset_ids"], "llm_id": config["model"], "llm_setting": setting,
            "similarity_threshold": r["similarity_threshold"], "vector_similarity_weight": r["vector_similarity_weight"],
            "top_n": r["page_size"], "top_k": r["top_k"], "rerank_id": r.get("rerank_id", ""),
            "prompt_config": {"quote": True, "reasoning": False, "use_kg": False,
                              "toc_enhance": False, "refine_multiturn": False}}


def chunk_list(value):
    if isinstance(value, dict):
        return list(value.values())
    return value or []


def main():
    parser = argparse.ArgumentParser(description="运行 RAGFlow 原生基线，每题使用独立会话")
    parser.add_argument("--config", default="configs/r0.json")
    parser.add_argument("--queries", default="data/financebench/queries.jsonl")
    parser.add_argument("--ingestion", default="data/ingestion.json")
    parser.add_argument("--split", choices=["tune", "dev", "test", "all"], default="dev")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--effort", default="default")
    parser.add_argument("--output", required=True)
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--transport-proof", help="模型网关/服务日志的强度传递确认 JSON，见文档")
    args = parser.parse_args()
    config = read_json(args.config)
    if not config["dataset_ids"]:
        config["dataset_ids"] = [read_json(args.ingestion)["dataset_id"]]
    ingestion = read_json(args.ingestion)
    if config["dataset_ids"] != [ingestion["dataset_id"]]:
        raise ValueError("基线只接受当前导入记录对应的固定知识库")
    proof = read_json(args.transport_proof) if args.transport_proof else None
    if args.effort != "default" and not args.retrieval_only:
        if not proof or proof.get("model") != config["model"] or args.effort not in proof.get("verified_efforts", []):
            raise ValueError("尚无端到端 reasoning_effort 传递证据；配置保存成功不等于参数生效，见 README")
    queries = read_jsonl(args.queries)
    if len({q["id"] for q in queries}) != len(queries):
        raise ValueError("查询题号重复")
    queries = [q for q in queries if args.split == "all" or q["split"] == args.split]
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit 必须大于 0")
        queries = queries[:args.limit]
    if not queries:
        raise ValueError("所选集合没有题目")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    identity = {"config": config, "queries_sha256": digest(Path(args.queries).read_bytes()),
                "query_ids": [q["id"] for q in queries], "split": args.split,
                "effort": args.effort, "retrieval_only": args.retrieval_only,
                "corpus_sha256": ingestion["corpus_sha256"], "embedding_model": ingestion["embedding_model"],
                "upstream": read_json("upstream.lock.json"), "transport_proof": proof}
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest["identity"] != identity:
            raise ValueError("运行目录与当前配置不一致；请使用新的 --output")
    else:
        manifest = {"identity": identity, "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "chat_id": None}
        write_json(manifest_path, manifest)
    client = Client(config["base_url"], timeout=config.get("request_timeout_seconds", 300))
    # 未完成解析的知识库不进入基线，避免把部署问题混成算法结果。
    docs = client.request("GET", f"/datasets/{ingestion['dataset_id']}/documents?page=1&page_size=1000")
    docs = docs.get("docs", []) if isinstance(docs, dict) else docs
    by_id = {d["id"]: d for d in docs}
    expected_doc_ids = {d["id"] for d in ingestion["documents"].values()}
    if set(by_id) != expected_doc_ids:
        raise ValueError("知识库实际文档与冻结导入清单不一致，请检查缺失或重复上传")
    for doc in ingestion["documents"].values():
        actual = by_id.get(doc["id"], {})
        if actual.get("run") not in ("DONE", "3", 3) or not actual.get("chunk_count"):
            raise ValueError(f"文档尚未完成解析或没有有效分块：{doc['id']}")
    snapshot = sorted([{k: d.get(k) for k in ("id", "name", "chunk_count", "size", "update_time")}
                       for d in docs], key=lambda d: d["id"])
    if manifest.get("index_snapshot") is not None and manifest["index_snapshot"] != snapshot:
        raise ValueError("索引文档或分块已变化，不能混入同一运行；请新建实验目录")
    manifest["index_snapshot"] = snapshot
    write_json(manifest_path, manifest)
    if not args.retrieval_only and not manifest["chat_id"]:
        payload = chat_body(config, args.effort, "TimeGuard-" + out.name + "-" + digest(identity)[:10])
        chat = client.request("POST", "/chats", payload)
        manifest["chat_id"] = chat["id"]
        manifest["chat_config"] = chat
        write_json(manifest_path, manifest)
        if args.effort != "default" and chat.get("llm_setting", {}).get("reasoning_effort") != args.effort:
            raise ValueError("RAGFlow 未保存 reasoning_effort，停止实验")
    results_path = out / "results.jsonl"
    previous = read_jsonl(results_path) if results_path.exists() else []
    done = {r["id"] for r in previous}
    # 失败请求同样作为已完成样本保存；如需重试须新建实验目录，不能挑选有利结果。
    with results_path.open("a", encoding="utf-8") as handle:
        for query in queries:
            if query["id"] in done:
                continue
            row = {"id": query["id"], "split": query["split"], "question": query["question"],
                   "status": "ok", "retrieved_chunks": [], "answer": None, "reference": {},
                   "usage": None, "cost_usd": None, "retrieval_seconds": None, "generation_seconds": None}
            start = time.perf_counter()
            try:
                r = client.request("POST", "/retrieval", retrieval_body(config, query))
                row["retrieved_chunks"] = chunk_list(r.get("chunks", []))
                row["retrieval_seconds"] = time.perf_counter() - start
                if not args.retrieval_only:
                    generation_start = time.perf_counter()
                    # 不传 session_id：RAGFlow 为每道独立题创建新会话，避免答案跨题污染。
                    answer = client.request("POST", "/chat/completions", {"chat_id": manifest["chat_id"],
                                             "question": query["question"], "stream": False})
                    row.update({"answer": answer.get("answer"), "reference": answer.get("reference", {}),
                                "session_id": answer.get("session_id"), "usage": answer.get("usage"),
                                "generation_seconds": time.perf_counter() - generation_start})
                    if not row["answer"] or "**ERROR**" in row["answer"]:
                        row["status"] = "generation_error"
            except Exception as exc:
                row["status"] = "request_error"
                row["error"] = f"{type(exc).__name__}: {exc}"[:500]
            row["elapsed_seconds"] = time.perf_counter() - start
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"{query['id']}：{row['status']}，{row['elapsed_seconds']:.2f} 秒", flush=True)


if __name__ == "__main__":
    main()
