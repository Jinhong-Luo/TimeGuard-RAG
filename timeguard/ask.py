"""在现有项目服务器上回答单个真实问题，保存原生回答与可核对的实际引用。"""
import argparse
from datetime import date
from pathlib import Path
from uuid import uuid4

from .common import digest, read_json, read_jsonl, write_json
from .judge import citation_context
from .r1_dev import worker
from .retrieval_experiment import catalog
from .scope import scope


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("question")
    p.add_argument("--as-of", type=date.fromisoformat)
    p.add_argument("--k", choices=(6, 12, 24), type=int, default=6)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if not args.question.strip() or len(args.question) > 5000:
        p.error("请提供长度不超过 5000 字符的问题")
    state = read_json("data/ingestion.json")
    config = read_json("configs/r0-low.json")
    chat = read_json("runs/all-selected/low/manifest.json")["chat_id"]
    docs = catalog(read_jsonl("data/financebench/documents.jsonl"), state)
    cutoff = args.as_of.isoformat() if args.as_of else None
    scope(args.question, docs, cutoff)
    q = {"id": uuid4().hex, "question": args.question, "split": "interactive", "as_of": cutoff}
    spec = {"queries": [q], "documents": docs, "dataset_id": state["dataset_id"], "model": config["model"],
            "chat_id": chat, "k": args.k, "weight": .3, "arm": "D", "finance_prompt": True}
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/"spec.json", spec)
    worker(out)
    result, = read_jsonl(out/"results.jsonl")
    if result["status"] != "ok":
        raise RuntimeError("原生问答失败，详情见独立运行目录")
    citations = citation_context(result)
    write_json(out/"answer.json", {"answer": result["answer"], "scope": result["scope"],
               "citations": citations, "usage": result.get("usage"), "result_sha256": digest(result)})
    print(result["answer"], flush=True)


if __name__ == "__main__":
    main()

