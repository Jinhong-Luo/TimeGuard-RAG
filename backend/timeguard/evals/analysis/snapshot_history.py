"""用同题、同报告、同块和同正文指纹复原旧检索记录。"""
import argparse
from collections import Counter
from pathlib import Path

from timeguard.common import digest, read_jsonl, write_json


def recover(row):
    trace = row["native_trace"]["evidence_layers"]
    known = {}
    sources = list(row.get("retrieved_chunks") or []) + list(row.get("reference", {}).get("chunks") or [])
    for answer in trace["answers"]:
        for request in answer["requests"]:
            sources.extend(request.get("complete_chunks") or [])
    for chunk in sources:
        if isinstance(chunk.get("content"), str) and chunk.get("document_id") and chunk.get("id"):
            known[(chunk["document_id"], chunk["id"], digest(chunk["content"]))] = chunk["content"]
    recovered = []
    for call in trace["retrievals"]:
        for position, chunk in enumerate(call["chunks"] or []):
            key = (chunk["document_id"], chunk["id"], chunk["content_sha256"])
            recovered.append({"call": call["call"], "channel": call["channel"], "position": position,
                              **chunk, "content": known.get(key)})
    return recovered


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("private_output", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    rows = read_jsonl(args.run / "results.jsonl")
    counts = Counter()
    with args.private_output.open("x", encoding="utf-8") as out:
        import json
        for row in rows:
            matches = recover(row)
            counts["检索块出现次数（含嵌套调用）"] += len(matches)
            counts["正文指纹已匹配次数"] += sum(c["content"] is not None for c in matches)
            counts["已有完整入口快照题数"] += any(a.get("snapshot") for a in row["native_trace"]["evidence_layers"]["answers"])
            out.write(json.dumps({"id": row["id"], "retrievals": matches}, ensure_ascii=False) + "\n")
    write_json(args.report, {"历史开发题数": len(rows), **counts, "历史逐题结果指纹": digest((args.run/"results.jsonl").read_bytes())})


if __name__ == "__main__":
    main()
