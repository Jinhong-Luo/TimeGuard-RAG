"""固定六档评测驱动；先核对预热索引，再逐批核验真实模型调用。"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .common import Client, digest, read_json, read_jsonl, write_json
from .score import summarize


EFFORTS = ("low", "medium", "high", "xhigh", "max", "ultra")
AUDIT_FIELDS = ("request_id", "timestamp_utc", "requested_model", "requested_effort", "actual_model",
                "reasoning_effort", "elapsed_seconds", "success", "error_type", "http_status", "usage",
                "bridge_source_sha256", "instruction_source_sha256")


def audit_summary(raw, effort, expected_calls):
    """只核验独占桥窗口的调用计数；不臆造无法逐题匹配的请求ID。"""
    records = [json.loads(line) for line in raw.splitlines() if line.strip()]
    success = [row for row in records if row.get("success") is True]
    metadata_ok = all(row.get("actual_model") == "gpt-6.1-sol" and row.get("reasoning_effort") == effort
                      and (row.get("usage") or {}).get("totalTokens", 0) > 0 for row in success)
    requested_ok = all(row.get("requested_effort") == effort and row.get("requested_model") in
                       ("gpt-6.1-sol", "gpt-6.1-sol@" + effort) for row in records)
    unique = len({row.get("request_id") for row in records}) == len(records)
    def valid_hash(value):
        return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)
    fingerprints = [{key: row.get(key) for key in ("bridge_source_sha256", "instruction_source_sha256")} for row in success]
    fingerprints_ok = all(valid_hash(row["bridge_source_sha256"])
                          and isinstance(row["instruction_source_sha256"], dict)
                          and all(isinstance(path, str) and path and valid_hash(value)
                                  for path, value in row["instruction_source_sha256"].items()) for row in fingerprints)
    fingerprints_ok = fingerprints_ok and len({digest(row) for row in fingerprints}) <= 1
    return {"expected_new_model_calls": expected_calls, "actual_attempts": len(records),
            "successful_model_calls": len(success), "failed_attempts": len(records) - len(success),
            "verified": metadata_ok and requested_ok and unique and fingerprints_ok and len(success) == expected_calls,
            "bridge_fingerprint": fingerprints[0] if fingerprints and fingerprints_ok else None,
            "usage_total_tokens": sum((row.get("usage") or {}).get("totalTokens", 0) for row in records),
            "error_types": dict(Counter(row.get("error_type") for row in records if not row.get("success"))),
            "window_sha256": digest(raw),
            "records": [{key: row.get(key) for key in AUDIT_FIELDS} for row in records],
            "scope": "按独占批次窗口核对，不声称请求ID与具体题目一一对应；所有失败尝试计入成本。"}


def same_bridge_as_history(evidence, history):
    """冻结首个成功批次的桥源码和指令文件摘要，不允许中途换条件。"""
    known = {digest(row["bridge_fingerprint"]) for row in history
             if row.get("verified") and row.get("successful_model_calls", 0) > 0 and row.get("bridge_fingerprint")}
    current = evidence.get("bridge_fingerprint")
    return len(known) <= 1 and (not current or not known or digest(current) in known)


def frozen_index(ingestion_path, retrieval_run, query_path):
    """纯检索150题是预热前置条件；实时检查相同84文档元数据快照。"""
    ingestion = read_json(ingestion_path)
    manifest = read_json(retrieval_run / "manifest.json")
    identity = manifest["identity"]
    queries = read_jsonl(query_path)
    rows = read_jsonl(retrieval_run / "results.jsonl")
    ids = [row["id"] for row in queries]
    if (len(ids) != 150 or len(set(ids)) != 150 or len(ingestion["documents"]) != 84
            or identity.get("retrieval_only") is not True or identity.get("split") != "all"
            or identity.get("scope") != "完整分组评测" or identity.get("query_ids") != ids
            or identity.get("queries_sha256") != digest(query_path.read_bytes())
            or identity.get("corpus_sha256") != ingestion["corpus_sha256"]
            or identity["config"]["dataset_ids"] != [ingestion["dataset_id"]]
            or identity.get("parse_profile") != ingestion.get("parse_profile")
            or len(rows) != 150 or {row["id"] for row in rows} != set(ids)
            or any(row.get("status") != "ok" for row in rows)):
        raise ValueError("需要同一完整GPU导入记录和成功的150题纯检索预热记录")
    # GPU配置必须由部署记录明确说明；不能只凭知识库名字判断。
    if "gpu" not in json.dumps(ingestion.get("parse_profile") or {}, ensure_ascii=False).lower():
        raise ValueError("导入记录没有明确的GPU解析条件")
    config = read_json("configs/r0-low.json")
    if identity["config"]["retrieval"] != config["retrieval"]:
        raise ValueError("纯检索预热与正式矩阵的检索参数不同")
    docs = Client(config["base_url"], timeout=1200).request(
        "GET", f"/datasets/{ingestion['dataset_id']}/documents?page=1&page_size=100")
    docs = docs.get("docs", []) if isinstance(docs, dict) else docs
    if (len(docs) != 84 or {row["id"] for row in docs} != {row["id"] for row in ingestion["documents"].values()}
            or any(row.get("run") not in ("DONE", "3", 3) or not row.get("chunk_count") for row in docs)):
        raise ValueError("完整84文档尚未DONE，或文档集合与导入记录不同")
    snapshot = sorted([{key: row.get(key) for key in ("id", "name", "chunk_count", "size", "update_time")}
                       for row in docs], key=lambda row: row["id"])
    if snapshot != manifest.get("index_snapshot") or digest(snapshot) != manifest.get("index_snapshot_sha256"):
        raise ValueError("当前索引快照与预热记录不同，拒绝继续")
    return {"index_snapshot_sha256": digest(snapshot), "retrieval_manifest_sha256": digest((retrieval_run / "manifest.json").read_bytes()),
            "retrieval_results_sha256": digest((retrieval_run / "results.jsonl").read_bytes()),
            "ingestion_sha256": digest(ingestion_path.read_bytes()), "latency_scope": "已纯检索预热，非冷启动延迟"}


def run_batch(module, args, run, kind, effort, audit, ledger, freeze):
    """子进程结束后主进程保存批次证据；失败保留，禁止自动更换预算重试。"""
    result = run / ("results.jsonl" if kind == "generation" else "model_reviews.jsonl")
    before = {row["id"] for row in read_jsonl(result)} if result.exists() else set()
    if before:
        history = read_jsonl(ledger) if ledger.exists() else []
        if not any(row.get("stage") == kind and row.get("run") == str(run) and row.get("verified") is True
                   and row.get("results_sha256") == digest(result.read_bytes()) for row in history):
            raise ValueError("已有结果缺少绑定当前原始字节的成功桥审计；不能以空窗口冒充历史调用已验证")
    start = audit.stat().st_size
    record = {"stage": kind, "effort": effort, "run": str(run), "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "byte_start": start, "index_freeze": freeze, "driver_source_sha256": digest(Path(__file__).read_bytes())}
    began = time.perf_counter()
    code = subprocess.call([sys.executable, "-m", module, *map(str, args)])
    end = audit.stat().st_size
    with audit.open("rb") as handle:
        handle.seek(start)
        raw = handle.read(end - start)
    added = [row for row in read_jsonl(result) if row["id"] not in before and "reuse_origin" not in row] if result.exists() else []
    expected = len(added) if kind == "generation" else sum(row.get("model_call_attempted") is True for row in added)
    evidence = audit_summary(raw, effort if kind == "generation" else "medium", expected)
    history = read_jsonl(ledger) if ledger.exists() else []
    evidence["bridge_fingerprint_matches_history"] = same_bridge_as_history(evidence, history)
    evidence["verified"] = evidence["verified"] and evidence["bridge_fingerprint_matches_history"]
    record.update(evidence, byte_end=end, returncode=code, wall_time_seconds=time.perf_counter() - began,
                  new_result_count=len(added), results_sha256=digest(result.read_bytes()) if result.exists() else None)
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps({"stage": kind, "effort": effort, "verified": record["verified"],
                      "actual_attempts": record["actual_attempts"], "new_result_count": len(added)}, ensure_ascii=False), flush=True)
    if code or not record["verified"] or any(row.get("status" if kind == "generation" else "judge_status") != "ok" for row in added):
        raise RuntimeError("批次失败或实际模型调用证据不完整，已保存分母和错误；停止后续自动选择")


def main():
    parser = argparse.ArgumentParser(description="固定双并发六档开发集与所选全量驱动；不加--execute只显示计划")
    parser.add_argument("--stage", choices=("dev", "all"), required=True)
    parser.add_argument("--ingestion", type=Path, default=Path("data/ingestion.json"))
    parser.add_argument("--retrieval-run", type=Path, default=Path("runs/r0-gpu-retrieval-all"))
    parser.add_argument("--audit-log", type=Path, default=Path(".cache/bridge-audit.jsonl"))
    parser.add_argument("--judge-base-url", default=os.environ.get("CODEX_BRIDGE_BASE_URL", "http://172.18.0.1:18864/v1"))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    print(json.dumps({"stage": args.stage, "execute": args.execute, "generation_concurrency": 2, "judge_concurrency": 2,
                      "new_generation_calls": 234 if args.stage == "dev" else 111,
                      "new_judge_calls": 234 if args.stage == "dev" else 111,
                      "required_before_execute": "GPU84文档统一索引完成、纯检索150完成且Retail全50结束；RAG桥独占。"}, ensure_ascii=False))
    if not args.execute:
        return
    if not os.environ.get("RAGFLOW_API_KEY") or not os.environ.get("CODEX_BRIDGE_TOKEN"):
        raise ValueError("缺少进程环境认证；使用本机私有凭据加载器，不把凭据写入命令或报告")
    if not args.audit_log.is_file():
        raise ValueError("缺少桥审计文件")
    queries = Path("data/financebench/queries.jsonl")
    freeze = frozen_index(args.ingestion, args.retrieval_run, queries)
    root = Path("runs")
    ledger = root / "matrix-execution.jsonl"
    selection = Path("reports/effort_selection.json")
    efforts = EFFORTS if args.stage == "dev" else (read_json(selection)["chosen_effort"],)
    for effort in efforts:
        if effort not in EFFORTS:
            raise ValueError("选择报告档位不在冻结六档内")
        if frozen_index(args.ingestion, args.retrieval_run, queries) != freeze:
            raise ValueError("批次间索引冻结证据发生变化")
        run = root / ("dev-complete" if args.stage == "dev" else "all-selected") / effort
        source = root / "dev-complete" / effort
        generate = ["--config", f"configs/r0-{effort}.json", "--effort", effort, "--split", args.stage,
                    "--ingestion", args.ingestion, "--output", run, "--concurrency", "2",
                    "--transport-proof", f"reports/transport-{effort}.json"]
        judge = [run, "--concurrency", "2", "--timeout", "360", "--base-url", args.judge_base_url, "--execute"]
        if args.stage == "all":
            generate += ["--reuse-run", source, "--selection", selection]
            judge += ["--reuse-reviews", source / "model_reviews.jsonl"]
        run_batch("timeguard.run", generate, run, "generation", effort, args.audit_log, ledger, freeze)
        run_batch("timeguard.judge", judge, run, "judge", effort, args.audit_log, ledger, freeze)
        subprocess.run([sys.executable, "-m", "timeguard.score", str(run), "--reviews", str(run / "model_reviews.jsonl")], check=True)
    if args.stage == "dev":
        subprocess.run([sys.executable, "-m", "timeguard.select", "--review-source", "model",
                        *[str(root / "dev-complete" / effort) for effort in EFFORTS], "--output", str(selection)], check=True)
    else:
        generated = read_jsonl(run / "results.jsonl")
        reviews = read_jsonl(run / "model_reviews.jsonl")
        gold = {row["id"]: row for row in read_jsonl("data/financebench/gold.jsonl")}
        queries = read_jsonl(queries)
        reports = {}
        for split in ("dev", "tune", "test", "all"):
            ids = [row["id"] for row in queries if split == "all" or row["split"] == split]
            reports[split], _, _ = summarize(ids, gold, [row for row in generated if row["id"] in ids], 6,
                                             [row for row in reviews if row["id"] in ids])
        write_json(run / "split_summaries.json", reports)
        manifest = read_json(run / "manifest.json")
        judgement = read_json(run / "model_reviews.manifest.json")
        final = {"scope": "所选档公开150题，含参与选择的39题；不是独立泛化成绩",
                 "selection_status": read_json(selection)["selection_status"], "chosen_effort": effort,
                 "human_spot_check": None, "split_summaries": reports, "index_freeze": freeze,
                 "generation_source_39_wall_seconds": manifest["identity"]["reuse"]["source_wall_time_seconds"],
                 "generation_new_111_wall_seconds": sum(row["wall_time_seconds"] for row in read_jsonl(run / "execution_history.jsonl")),
                 "judge_source_39_wall_seconds": judgement["identity"]["reuse"]["source_wall_time_seconds"],
                 "judge_new_111_wall_seconds": sum(row["wall_time_seconds"] for row in read_jsonl(run / "model_reviews.execution.jsonl")),
                 "source_sha256": {name: digest((run / name).read_bytes()) for name in
                                   ("manifest.json", "results.jsonl", "model_reviews.manifest.json", "model_reviews.jsonl")},
                 "matrix_audit_ledger_sha256": digest(ledger.read_bytes()), "cost_usd": None}
        write_json(Path("reports/r0-final-summary.json"), final)


if __name__ == "__main__":
    main()
