"""离线复核冻结作答，仅输出引用映射与可见证据，不自动判定语义支持。"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

from timeguard.evals.judge import citation_context
from timeguard.evidence.citation_spans import recover, visible_sections


def audit(row, sources):
    old = citation_context(row)
    if not old["unmapped_indexes"]:
        return None
    trace = row["native_trace"]
    layer, = trace["evidence_layers"]["answers"]
    req, = [r for r in layer["requests"] if r["status"] == 200]
    sections = visible_sections(req["body"]["messages"])
    complete = {c["block_sha256"]: c for c in req["complete_chunks"]}
    fingerprints = defaultdict(set)
    for event in trace["evidence_layers"]["retrievals"]:
        for chunk in event.get("chunks") or []:
            fingerprints[chunk["id"]].add(chunk.get("content_sha256"))
    found = {}
    for n in old["unmapped_indexes"]:
        if not 0 < n <= len(layer["packet"] or []):
            continue
        cid = layer["packet"][n - 1]["id"]
        source = sources.get(cid)
        if source:
            c = recover(n, layer, sections.get(n), complete.get(layer["block_hashes"][n - 1]),
                        source, fingerprints[cid], source.get("rendered"))
            if c:
                found[n] = c
    req["verified_excerpts"] = list(found.values())
    new = citation_context(row)
    return {"position": row["position"], "recovered": {n: c["kind"] for n, c in found.items()},
            "old": old["mapping_status"], "new": new["mapping_status"],
            "missing": new["unmapped_indexes"], "answer": row["answer"], "citations": new["citations"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("sources", type=Path, help="按块标识保存的索引原文和原生表格视图")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    sources = json.loads(args.sources.read_text(encoding="utf-8"))
    seen = set()
    records = []
    for path in sorted(args.run.glob("generation/*/results.jsonl")):
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if row["position"] in seen:
                    raise ValueError("重复作答位置")
                seen.add(row["position"])
                record = audit(row, sources)
                if record:
                    records.append(record)
    if not seen:
        raise ValueError("未找到冻结作答")
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump({"positions": len(seen), "audited": records}, stream, ensure_ascii=False)


if __name__ == "__main__":
    main()
