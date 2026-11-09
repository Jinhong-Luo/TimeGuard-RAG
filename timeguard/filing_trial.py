"""在既有索引上验证披露日前后可见文档变化，截止日按整日计算。"""
import argparse
from datetime import date, timedelta
from pathlib import Path
from .common import Client, digest, read_json, read_jsonl, write_json
from .retrieval_experiment import catalog
from .scope import scope, search


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    state = read_json("data/ingestion.json")
    config = read_json("configs/r0-low.json")
    config["dataset_ids"] = [state["dataset_id"]]
    docs = catalog(read_jsonl("data/financebench/documents.jsonl"), state)
    dated = [d for d in docs if d.get("published_at")]
    client = Client(config["base_url"])
    by_name = {d["doc_name"].casefold(): d for d in docs}
    checks = []
    for filing in dated:
        day = date.fromisoformat(filing["published_at"])
        cutoffs = {(day - timedelta(days=1)).isoformat(), day.isoformat()}
        if filing.get("event_date"):
            cutoffs.add(filing["event_date"])
        for cutoff in sorted(cutoffs):
            question = f"What were {filing['company']}'s FY{filing['doc_period']} sales as of {cutoff}?"
            plan = scope(question, docs)
            if (filing["id"] in plan["document_ids"]) != (cutoff >= day.isoformat()):
                raise ValueError("披露日边界验证失败")
            chunks, path = search(client, config, question, plan, "D", by_name, 6, .3)
            checks.append({"document": filing["doc_name"], "cutoff": cutoff, "published_at": filing["published_at"],
                           "date_source": filing["source"], "allowed_count": len(plan["document_ids"]),
                           "candidate_count": len(chunks), "path": path,
                           "found_documents": sorted({c["document_keyword"] for c in chunks})})
    write_json(args.output / "summary.json", {"catalog_sha256": digest(docs), "documents": len(docs),
               "dated_documents": len(dated), "checks": checks})


if __name__ == "__main__":
    main()

