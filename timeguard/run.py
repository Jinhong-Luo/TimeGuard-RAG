"""R0 原生检索与问答运行器；不读取标准答案或标准证据。"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime
import json
import math
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


def reuse_dev(source, selection_path, identity, queries):
    """只复用选择报告绑定的完整开发集；不读取裁判材料或标准答案。"""
    source = Path(source)
    source_manifest = read_json(source / "manifest.json")
    original = source_manifest["identity"]
    selection = read_json(selection_path)
    manifest_hash = digest((source / "manifest.json").read_bytes())
    result_hash = digest((source / "results.jsonl").read_bytes())
    dev_ids = [query["id"] for query in queries if query["split"] == "dev"]
    if (identity["split"] != "all" or identity["scope"] != "完整分组评测" or len(queries) != 150
            or original["split"] != "dev" or original["scope"] != "完整分组评测"
            or len(dev_ids) != 39 or original["query_ids"] != dev_ids or original["full_split_count"] != 39
            or original["full_split_query_ids_sha256"] != digest(dev_ids)):
        raise ValueError("复用只允许完整39题开发集进入同一数据源的完整150题报告")
    varying = {"split", "scope", "query_ids", "full_split_count", "full_split_query_ids_sha256"}
    if ({key: value for key, value in original.items() if key not in varying}
            != {key: value for key, value in identity.items() if key not in varying}):
        raise ValueError("复用源的模型、配置、并发、语料、查询源或运行源码与目标不一致")
    mode = selection.get("review_source")
    candidates = selection.get("candidates", [])
    quality = "model_answer_accuracy" if mode == "model" else "answer_accuracy"
    if (mode not in ("model", "human") or selection.get("chosen_effort") != identity["effort"]
            or (mode == "model" and selection.get("selection_status") != "模型裁判暂定推荐，未完成人工抽查")
            or len(candidates) != 6 or {row.get("effort") for row in candidates} !=
            {"low", "medium", "high", "xhigh", "max", "ultra"}
            or not any(row.get("manifest") == manifest_hash and row.get("results") == result_hash
                       and row.get("effort") == identity["effort"] for row in selection.get("source_runs", []))):
        raise ValueError("缺少绑定该开发集原始文件的有效六档选择报告")
    for candidate in candidates:
        if (candidate.get("denominator") != 39 or candidate.get("completed") != 39
                or candidate.get("status_counts") != {"ok": 39}
                or candidate.get("model_reviewed" if mode == "model" else "reviewed") != 39
                or type(candidate.get(quality)) not in (int, float) or not 0 <= candidate[quality] <= 1
                or type(candidate.get("mean_seconds")) not in (int, float)
                or not math.isfinite(candidate["mean_seconds"]) or candidate["mean_seconds"] <= 0):
            raise ValueError("选择报告的候选缺少完整有效评分或真实耗时")
    tolerance = selection.get("quality_tolerance")
    if type(tolerance) not in (int, float) or not 0 <= tolerance <= 1:
        raise ValueError("选择报告的质量容差无效")
    best = max(row[quality] for row in candidates)
    chosen = min((row for row in candidates if row[quality] >= best - tolerance), key=lambda row: row["mean_seconds"])
    if chosen["effort"] != identity["effort"]:
        raise ValueError("选择报告的声明档位与其原始候选评分不符")
    lines = [line for line in (source / "results.jsonl").read_bytes().splitlines() if line.strip()]
    rows = [json.loads(line) for line in lines]
    by_id = {row["id"]: row for row in rows}
    questions = {query["id"]: query["question"] for query in queries}
    if (len(rows) != 39 or set(by_id) != set(dev_ids)
            or any(row.get("status") != "ok" or row.get("split") != "dev"
                   or row.get("question") != questions[row["id"]] or not row.get("answer")
                   or type(row.get("elapsed_seconds")) not in (int, float)
                   or not math.isfinite(row["elapsed_seconds"]) or row["elapsed_seconds"] <= 0 for row in rows)):
        raise ValueError("复用源必须包含39个不同题号的完整成功答案及原始耗时")
    history_path = source / "execution_history.jsonl"
    history = read_jsonl(history_path)
    if not history or any(row.get("run_identity_sha256") != digest(original)
                          or type(row.get("wall_time_seconds")) not in (int, float)
                          or not math.isfinite(row["wall_time_seconds"]) or row["wall_time_seconds"] < 0 for row in history):
        raise ValueError("复用源缺少真实运行墙钟记录，不能把复用成本当零")
    proof = {"source_run": str(source), "manifest_sha256": manifest_hash, "results_sha256": result_hash,
             "selection_sha256": digest(Path(selection_path).read_bytes()), "selection_review_source": mode,
             # 仅保存选择报告的摘要元数据；生成端不读取裁判文件或金标。
             "selection_source_hashes": next(row for row in selection["source_runs"]
                                               if row.get("manifest") == manifest_hash and row.get("results") == result_hash
                                               and row.get("effort") == identity["effort"]),
             "execution_history_sha256": digest(history_path.read_bytes()),
             "source_wall_time_seconds": sum(row["wall_time_seconds"] for row in history),
             "reused_count": 39, "new_count": 111,
             "index_snapshot_sha256": digest(source_manifest["index_snapshot"])}
    for row, line in zip(rows, lines):
        row["reuse_origin"] = {"run": str(source), "results_sha256": result_hash, "row_sha256": digest(line)}
    return rows, proof


def run_one(query, config, chat_id, retrieval_only, client_factory=Client):
    """每题单独建立无会话状态的客户端，不接收金标或前题答案。"""
    row = {"id": query["id"], "split": query["split"], "question": query["question"],
           "status": "ok", "retrieved_chunks": [], "answer": None, "reference": {},
           "usage": None, "cost_usd": None, "retrieval_seconds": None, "generation_seconds": None}
    start = time.perf_counter()
    try:
        client = client_factory(config["base_url"], timeout=config.get("request_timeout_seconds", 300))
        r = client.request("POST", "/retrieval", retrieval_body(config, query))
        row["retrieved_chunks"] = chunk_list(r.get("chunks", []))
        row["retrieval_seconds"] = time.perf_counter() - start
        if not retrieval_only:
            generation_start = time.perf_counter()
            # 不传 session_id：RAGFlow 为每道独立题创建新会话，避免答案跨题污染。
            answer = client.request("POST", "/chat/completions", {"chat_id": chat_id,
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
    return row


def run_queries(queries, config, chat_id, results_path, retrieval_only=False, concurrency=1, client_factory=Client,
                reuse_rows=(), run_signature=None):
    """仅主线程追加结果；已记录失败也保留，不用续跑挑选更有利的答案。"""
    if type(concurrency) is not int or concurrency < 1:
        raise ValueError("并发数必须是正整数")
    ids = {query["id"] for query in queries}
    previous = read_jsonl(results_path) if results_path.exists() else []
    done = {row["id"] for row in previous}
    if len(ids) != len(queries) or len(done) != len(previous) or done - ids:
        raise ValueError("计划或已有结果题号重复，或结果超出计划，拒绝混用")
    previous_by_id = {row["id"]: row for row in previous}
    if any(row["id"] not in ids or (row["id"] in done and row != previous_by_id[row["id"]]) for row in reuse_rows):
        raise ValueError("已保存的复用题与冻结来源不符")
    with results_path.open("a", encoding="utf-8") as handle, ThreadPoolExecutor(max_workers=concurrency) as pool:
        for row in reuse_rows:
            if row["id"] not in done:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                done.add(row["id"])
        futures = [pool.submit(run_one, query, config, chat_id, retrieval_only, client_factory)
                   for query in queries if query["id"] not in done]
        for future in as_completed(futures):
            row = future.result()
            if run_signature:
                row["generation_run_signature"] = run_signature
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"{row['id']}：{row['status']}，{row['elapsed_seconds']:.2f} 秒", flush=True)


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
    parser.add_argument("--concurrency", type=int, default=1, help="每题独立会话的并发数；固定写入运行签名，默认1")
    parser.add_argument("--reuse-run", type=Path, help="仅全量150题：复用选择报告绑定的完整39题开发集")
    parser.add_argument("--selection", type=Path, help="复用必须提供已冻结的六档选择报告")
    parser.add_argument("--transport-proof", help="模型网关/服务日志的强度传递确认 JSON，见文档")
    args = parser.parse_args()
    if args.concurrency < 1:
        parser.error("--concurrency 必须为正整数")
    if bool(args.reuse_run) != bool(args.selection):
        parser.error("--reuse-run 与 --selection 必须同时提供")
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
    full_split_count = len(queries)
    full_split_query_ids_sha256 = digest([q["id"] for q in queries])
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
                "scope": "小样本连通测试" if args.limit is not None else "完整分组评测",
                "full_split_count": full_split_count,
                "full_split_query_ids_sha256": full_split_query_ids_sha256,
                "effort": args.effort, "retrieval_only": args.retrieval_only, "concurrency": args.concurrency,
                "corpus_sha256": ingestion["corpus_sha256"], "embedding_model": ingestion["embedding_model"],
                "parse_profile": ingestion.get("parse_profile"),
                "upstream": read_json("upstream.lock.json"), "transport_proof": proof,
                "runner_code_sha256": digest(Path(__file__).read_bytes())}
    reused_rows, reuse_proof = [], None
    if args.reuse_run:
        reused_rows, reuse_proof = reuse_dev(args.reuse_run, args.selection, identity, queries)
        identity["reuse"] = reuse_proof
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest["identity"] != identity:
            raise ValueError("运行目录与当前配置不一致；请使用新的 --output")
    else:
        manifest = {"identity": identity, "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "chat_id": None}
        write_json(manifest_path, manifest)
    manifest["signature"] = digest(identity)
    client = Client(config["base_url"], timeout=config.get("request_timeout_seconds", 300))
    # 未完成解析的知识库不进入基线，避免把部署问题混成算法结果。
    docs = client.request("GET", f"/datasets/{ingestion['dataset_id']}/documents?page=1&page_size=100")
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
    if reuse_proof and reuse_proof["index_snapshot_sha256"] != digest(snapshot):
        raise ValueError("当前索引文档快照与复用开发集不同，拒绝混用")
    manifest["index_snapshot"] = snapshot
    manifest["index_snapshot_sha256"] = digest(snapshot)
    write_json(manifest_path, manifest)
    if not args.retrieval_only and not manifest["chat_id"]:
        payload = chat_body(config, args.effort, "TimeGuard-" + out.name + "-" + digest(identity)[:10])
        chat = client.request("POST", "/chats", payload)
        if args.effort != "default" and chat.get("llm_setting", {}).get("reasoning_effort") != args.effort:
            raise ValueError("RAGFlow 未保存 reasoning_effort，停止实验")
        manifest["chat_id"] = chat["id"]
        manifest["chat_config"] = chat
        write_json(manifest_path, manifest)
    if (not args.retrieval_only and args.effort != "default"
            and manifest.get("chat_config", {}).get("llm_setting", {}).get("reasoning_effort") != args.effort):
        raise ValueError("保存的 chat 配置缺少正确思考强度，续跑不能绕过校验")
    results_path = out / "results.jsonl"
    before = read_jsonl(results_path) if results_path.exists() else []
    started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    started = time.perf_counter()
    status = "interrupted"
    try:
        run_queries(queries, config, manifest["chat_id"], results_path, args.retrieval_only, args.concurrency,
                    client_factory=Client, reuse_rows=reused_rows, run_signature=digest(identity))
        status = "completed"
    finally:
        after = read_jsonl(results_path) if results_path.exists() else []
        before_ids = {row["id"] for row in before}
        added = [row for row in after if row["id"] not in before_ids]
        execution = {"started_at_utc": started_at,
                     "run_identity_sha256": digest(identity),
                     "status": status, "wall_time_seconds": time.perf_counter() - started,
                     "new_result_count": sum("reuse_origin" not in row for row in added),
                     "reused_result_count": sum("reuse_origin" in row for row in added),
                     "planned_denominator": len(queries), "concurrency": args.concurrency,
                     "说明": "本次墙钟仅覆盖新增请求与写入；复用题原始墙钟见identity.reuse，不能将新增111题墙钟除以150。"}
        with (out / "execution_history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(execution, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
