"""下载固定版本公开数据，并在第一次调优前按公司冻结集合。"""

import argparse
import collections
from pathlib import Path
from urllib.parse import quote

from .common import digest, download, read_json, read_jsonl, write_json, write_jsonl


def prepare(source, docs, seed="timeguard-v1"):
    if len(source) != 150 or len({r["financebench_id"] for r in source}) != 150:
        raise ValueError("FinanceBench 公开单元应恰好有 150 个不同题号")
    companies = sorted({r["company"] for r in source}, key=lambda x: digest(seed + ":" + x))
    n = len(companies)
    allocation = {c: "tune" if i < int(n * 0.6) else "dev" if i < int(n * 0.8) else "test"
                  for i, c in enumerate(companies)}
    queries, gold = [], []
    for row in source:
        base = {"id": row["financebench_id"], "question": row["question"], "split": allocation[row["company"]]}
        queries.append(base)
        gold.append({**base, "company": row["company"], "doc_name": row["doc_name"],
                     "answer": row["answer"], "evidence": row["evidence"],
                     "question_type": row.get("question_type"), "question_reasoning": row.get("question_reasoning")})
    # 所有标准证据关联的文档仅用于形成全量语料，不用于逐题过滤或模型输入。
    required = {r["doc_name"] for r in source} | {e["doc_name"] for r in source for e in r["evidence"]}
    metadata = {r["doc_name"]: r for r in docs}
    missing = required - metadata.keys()
    if missing:
        raise ValueError(f"缺失文档元数据：{sorted(missing)}")
    document_splits = collections.defaultdict(set)
    for row in gold:
        for name in {row["doc_name"]} | {e["doc_name"] for e in row["evidence"]}:
            document_splits[name].add(row["split"])
    if any(len(splits) > 1 for splits in document_splits.values()):
        raise ValueError("存在跨集合共享文档，需先以公司与文档关联分组，不能直接评测")
    return queries, gold, [metadata[name] for name in sorted(required)], allocation


def main():
    parser = argparse.ArgumentParser(description="冻结 FinanceBench 数据和按公司分组的清单")
    parser.add_argument("--output", default="data/financebench")
    parser.add_argument("--pdfs", action="store_true", help="同时下载本单元关联的全部原始 PDF")
    args = parser.parse_args()
    root = Path(args.output)
    lock = read_json("upstream.lock.json")["financebench"]
    base = f"https://raw.githubusercontent.com/patronus-ai/financebench/{lock['commit']}"
    hashes = {}
    for filename, field in [("financebench_open_source.jsonl", "questions_git_blob"),
                            ("financebench_document_information.jsonl", "documents_git_blob")]:
        hashes[filename] = download(base + "/data/" + filename, root / "raw" / filename, lock[field])
    queries, gold, docs, allocation = prepare(read_jsonl(root / "raw/financebench_open_source.jsonl"),
                                               read_jsonl(root / "raw/financebench_document_information.jsonl"))
    write_jsonl(root / "queries.jsonl", queries)
    write_jsonl(root / "gold.jsonl", gold)
    write_json(root / "company_splits.json", allocation)
    ledger = []
    for doc in docs:
        entry = {**doc, "source_commit": lock["commit"], "pdf_sha256": None,
                 "download_url": base + "/pdfs/" + quote(doc["doc_name"] + ".pdf", safe=""),
                 "local_path": "pdfs/" + doc["doc_name"] + ".pdf"}
        # 再次生成清单时保留并校验已经落盘的PDF，不因省略--pdfs丢失摘要。
        if args.pdfs or (root / entry["local_path"]).exists():
            entry["pdf_sha256"] = download(entry["download_url"], root / entry["local_path"])
            if not (root / entry["local_path"]).read_bytes().startswith(b"%PDF"):
                raise ValueError(f"不是有效 PDF 文件：{doc['doc_name']}")
        ledger.append(entry)
    write_jsonl(root / "documents.jsonl", ledger)
    report = {"source_commit": lock["commit"], "questions": len(queries), "documents": len(docs),
              "companies": len(allocation), "split_counts": dict(collections.Counter(r["split"] for r in queries)),
              "source_sha256": hashes, "queries_sha256": digest((root / "queries.jsonl").read_bytes()),
              "pdfs_downloaded": sum(r["pdf_sha256"] is not None for r in ledger),
              "说明": "按公司隔离，测试答案和标准证据仅在离线评分使用；完整150题报告不等于独立泛化成绩。"}
    write_json(root / "manifest.json", report)
    print(__import__("json").dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
