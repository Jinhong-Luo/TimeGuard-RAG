"""从旧端到端判对且引用有支持的开发题中，预先固定跨公司的保护样本。"""
import argparse
import json
from pathlib import Path

from timeguard.common import digest, read_json, read_jsonl, write_json
from timeguard.retrieval.catalog import catalog
from timeguard.retrieval.scope import scope


def prepare(source, output, count):
    scores = {r["id"]:r for r in read_jsonl(source/"model_reviews.jsonl")}
    docs = catalog(read_jsonl("data/financebench/documents.jsonl"),read_json("data/ingestion.json"))
    seen = set()
    kept = []
    for row in sorted(read_jsonl(source/"results.jsonl"),key=lambda r:r["id"]):
        score = scores[row["id"]]
        if score["answer_correct"] is not True or score["citation_support"] is not True:
            continue
        ans = row["native_trace"].get("evidence_layers",{}).get("answers") or []
        companies = scope(row["question"],docs,row.get("as_of"))["companies"]
        if (len(companies)!=1 or companies[0] in seen or len(ans)!=1 or not ans[0].get("snapshot")
                or sum(r.get("status")==200 for r in ans[0]["requests"])!=1):
            continue
        kept.append(row)
        seen.add(companies[0])
        if len(kept)==count:
            break
    if len(kept)!=count:
        raise ValueError("符合旧评分及完整轨迹条件的公司不足")
    output.mkdir(exist_ok=False,parents=True)
    (output/"results.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in kept),encoding="utf-8")
    manifest = read_json(source/"manifest.json")
    manifest["identity"]["query_ids"] = [r["id"] for r in kept]
    manifest["旧结果指纹"] = digest((source/"results.jsonl").read_bytes())
    write_json(output/"manifest.json",manifest)


if __name__=="__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source",type=Path)
    p.add_argument("output",type=Path)
    p.add_argument("--count",type=int,default=3)
    a = p.parse_args()
    prepare(a.source,a.output,a.count)
