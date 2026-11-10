"""从实际评测记录生成无财报原文、无凭据的交付摘要。"""
import argparse
from collections import Counter
from pathlib import Path
from .common import digest, read_json, read_jsonl, write_json


def summarize(path):
    identity = read_json(path/"manifest.json")["identity"]
    rows = read_jsonl(path/"results.jsonl")
    reviews = read_jsonl(path/"model_reviews.jsonl") if (path/"model_reviews.jsonl").exists() else []
    ids = identity["query_ids"]
    if any(len({r["id"] for r in group}) != len(group) or {r["id"] for r in group}-set(ids) for group in (rows,reviews)):
        raise ValueError("发现重复或计划外题号")
    execution = read_json(path/"execution.json")
    checks = read_json(path/"judge-audit.json") if (path/"judge-audit.json").exists() else {}
    return {
        "run":path.name, "planned":len(ids), "answers":len(rows), "reviews":len(reviews),
        "ok":sum(r["status"]=="ok" for r in rows),
        "correct":sum(r.get("answer_correct") is True for r in reviews),
        "incorrect":sum(r.get("answer_correct") is False for r in reviews),
        "unknown":sum(r.get("answer_correct") is None for r in reviews),
        "model_requests":execution["actual_calls"]+execution.get("reused_source_calls",0),
        "failed_requests":execution["failed_calls"], "returned_tokens":execution["tokens"]+execution.get("reused_source_tokens",0),
        "judge_requests":checks.get("actual_calls"), "judge_tokens":checks.get("tokens"),
        "visible_chunks":sum(len((r.get("citation_trace") or {}).get("visible",{})) for r in rows),
        "citation_mapping":dict(Counter((r.get("citation_mapping") or {}).get("mapping_status") for r in reviews)),
        "results_sha256":digest((path/"results.jsonl").read_bytes()),
        "reviews_sha256":digest((path/"model_reviews.jsonl").read_bytes()) if reviews else None,
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args=parser.parse_args()
    write_json(args.output,{"method":"模型暂评，失败、未知及缺失保留在计划分母。模型请求与裁判分列。",
                            "runs":[summarize(p) for p in args.runs]})


if __name__=="__main__":
    main()
