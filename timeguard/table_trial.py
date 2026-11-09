"""同一财报、同一批表格建立两套独立索引，比较整表与带表头的行组。"""
import argparse
import re
from pathlib import Path
import time
from uuid import uuid4

from .common import Client, digest, read_json, read_jsonl, write_json, write_jsonl
from .score import evidence_metrics
from .scope import scope

ROW = re.compile(r"<tr\b[^>]*>.*?</tr>", re.I | re.S)
TABLE = re.compile(r"<table\b[^>]*>(.*?)</table>", re.I | re.S)
HEADER = re.compile(r"\b(?:19|20)\d{2}\b|\b(?:million|thousand|billion)s?\b|币种|单位", re.I)


def windows(text, size=6):
    """重复原始表题与连续表头。有跨行单元格或多表时保留整块，避免伪造列对应。"""
    tables = list(TABLE.finditer(text))
    if len(tables) != 1 or re.search(r"rowspan\s*=\s*['\"]?(?:[2-9]|\d{2,})", text, re.I):
        return [text]
    rows = list(ROW.finditer(tables[0][1]))
    first = 0
    while first < len(rows):
        html = rows[first].group()
        if re.search(r"<td\b", html, re.I) and not (first < 3 and HEADER.search(html)):
            break
        first += 1
    if len(rows) - first <= size:
        return [text]
    start = tables[0].start(1)
    prefix = text[:start + rows[first].start()]
    suffix = text[start + rows[-1].end():]
    return [prefix + "".join(r.group() for r in rows[i:i + size]) + suffix for i in range(first, len(rows), size)]


def load_parents(client, state, name):
    source = f"/datasets/{state['dataset_id']}/documents/{state['documents'][name]['id']}/chunks"
    parents = {}
    for page in range(1, 10000):
        response = client.request("GET", f"{source}?page={page}&page_size=100")
        for c in response["chunks"]:
            parents[c["id"]] = c
        if len(parents) == response["total"]:
            return [c for c in parents.values() if c.get("available") and TABLE.search(c["content"])]
        if not response["chunks"]:
            break
    raise ValueError("父块清单不完整")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--document", default="3M_2018_10K")
    args = p.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    ledger = {d["doc_name"]: d for d in read_jsonl("data/financebench/documents.jsonl")}
    doc = ledger[args.document]
    state = read_json("data/ingestion.json")
    pdf = Path("data/financebench") / doc["local_path"]
    if digest(pdf.read_bytes()) != state["documents"][args.document]["sha256"]:
        raise ValueError("财报 PDF 与导入记录不匹配")
    config = read_json("configs/r0-low.json")
    client = Client(config["base_url"])
    parents = load_parents(client, state, args.document)
    if not parents:
        raise ValueError("原生索引中没有可用表格")
    write_jsonl(out / "parents.jsonl", parents)
    queries = []
    for q in read_jsonl("data/financebench/queries.jsonl"):
        plan = scope(q["question"], [{**doc, "id": args.document}])
        if q["split"] == "tune" and plan["companies"] and doc["doc_period"] in plan["years"]:
            queries.append(q)
    if not queries:
        raise ValueError("当前财报没有匹配的调优问题")
    report = {"source_document": args.document, "source_pdf_sha256": digest(pdf.read_bytes()),
              "parent_count": len(parents), "parents_sha256": digest(parents), "source_dataset": state["dataset_id"],
              "embedding": state["embedding_model"], "source_sha256": digest(Path(__file__).read_bytes()), "arms": {}}
    for arm in ("whole", "rows"):
        target = client.request("POST", "/datasets", {"name": f"TimeGuard-R1-{arm}-{uuid4().hex[:12]}",
                 "embedding_model": state["embedding_model"], "permission": "me", "chunk_method": "naive"})
        dataset = target["id"]
        if dataset == state["dataset_id"]:
            raise ValueError("新索引与原知识库冲突")
        stats = report["arms"][arm] = {"dataset_id": dataset}
        write_json(out / "summary.json", report)
        target_doc = client.upload(dataset, pdf)[0]["id"]
        stats["document_id"] = target_doc
        endpoint = f"/datasets/{dataset}/documents/{target_doc}/chunks"
        parent_map = {}
        seen = {}
        for parent in parents:
            for content in windows(parent["content"]) if arm == "rows" else [parent["content"]]:
                # 保留源表标识，防止不同原表的相同行在新索引中碰撞。
                payload = "Source table " + parent["id"] + "\n" + content
                key = digest(payload)
                if key in seen:
                    continue
                child = client.request("POST", endpoint, {"content": payload})["chunk"]
                seen[key] = child["id"]
                parent_map[child["id"]] = parent
        write_json(out / f"{arm}-parents.json", {i: c["id"] for i, c in parent_map.items()})
        for attempt in range(8):
            count = client.request("GET", endpoint + "?page=1&page_size=1")["total"]
            if count == len(parent_map):
                break
            time.sleep(1)
        if count != len(parent_map):
            raise ValueError("衍生索引的实际数量与写入回执不一致")
        rows = []
        for q in queries:
            chunks = client.request("POST", "/retrieval", {**config["retrieval"], "dataset_ids": [dataset],
                                    "question": q["question"]})["chunks"]
            unique = list(dict.fromkeys(parent_map[c["id"]]["id"] for c in chunks))
            rows.append({"id": q["id"], "children": chunks, "parents": unique})
        write_jsonl(out / f"{arm}.jsonl", rows)
        stats.update(unique_chunks=count, questions=len(rows), results_sha256=digest(rows))
    gold = {g["id"]: g for g in read_jsonl("data/financebench/gold.jsonl")}
    by_id = {p["id"]: p for p in parents}
    for arm, stats in report["arms"].items():
        rows = read_jsonl(out / f"{arm}.jsonl")
        stats["parent_page_recall"] = sum(evidence_metrics(gold[r["id"]], [by_id[i] for i in r["parents"]], 6)["page_hit_recall"] or 0 for r in rows) / len(rows)
    write_json(out / "summary.json", report)


if __name__ == "__main__":
    main()

