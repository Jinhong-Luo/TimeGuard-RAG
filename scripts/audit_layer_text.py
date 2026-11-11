"""对三层诊断中已发送的文本再次回读索引，区分包裹完整原文、短摘录及其他改写。"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def classify(original, shown):
    if original == shown:
        return "逐字完整"
    if original in shown:
        return "含有完整原文"
    if " ".join(original.split()) in " ".join(shown.split()):
        return "忽略空白后包含全文"
    if shown and shown in original:
        return "连续短摘录"
    return "其他文本差异"


def worker():
    from api.apps import app
    from common import settings
    from api.db.services.knowledgebase_service import KnowledgebaseService
    from rag.nlp.search import index_name
    settings.init_settings()
    data = json.load(sys.stdin)
    found, kb = KnowledgebaseService.get_by_id(data["dataset_id"])
    if not found:
        raise ValueError("诊断知识库不存在")
    rows = []
    for c in data["chunks"]:
        raw = settings.docStoreConn.get(c["id"], index_name(kb.tenant_id), [kb.id])
        if not raw or raw.get("doc_id") != c["document_id"]:
            raise ValueError("无法验证来源块的文档编号")
        source = raw["content_with_weight"]
        checksum = hashlib.sha256(json.dumps(source, ensure_ascii=False).encode()).hexdigest()
        if checksum != c["source_sha256"]:
            raise ValueError("索引原块已变化")
        rows.append({k: c[k] for k in ("question_id", "document_id", "id")} | {
            "match": classify(source, c["content"]), "source_chars": len(source), "received_chars": len(c["content"])})
    print("TIMEGUARD_TEXT " + json.dumps(rows, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    spec = json.loads((args.run / "spec.json").read_text(encoding="utf-8"))
    chunks = []
    unknown = 0
    for line in (args.run / "results.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        for answer in row["native_trace"]["evidence_layers"]["answers"]:
            for req in answer["requests"]:
                for c in req["complete_chunks"] or []:
                    if c.get("source_sha256"):
                        chunks.append({**c, "question_id": row["id"]})
                    else:
                        unknown += 1
    script = Path(__file__).read_text(encoding="utf-8")
    completed = subprocess.run(["docker", "exec", "-i", "-w", "/ragflow", "timeguard_ljh-ragflow-cpu-1",
        "python", "-c", script, "--worker"], input=json.dumps({"chunks": chunks, "dataset_id": spec["dataset_id"]}),
        capture_output=True, text=True, check=True)
    rows, = [json.loads(line.partition(" ")[2]) for line in completed.stdout.splitlines() if line.startswith("TIMEGUARD_TEXT ")]
    result = {"counts": dict(Counter(r["match"] for r in rows)), "unverified": unknown, "rows": rows}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    if "--worker" in sys.argv:
        worker()
    else:
        main()
