"""完整84文档索引的只读审计；任务、分页分块和失败项均保留完整分母。"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from . import common, extend_corpus
from .common import Client, digest, read_json, read_jsonl
from .extend_corpus import parser_config, pdf_sha256


class AuditFailure(ValueError):
    """只保存预定义核验原因，不把服务异常响应正文写入公开报告。"""


def task_accounting(tasks, document_id, pages, metadata_count):
    """从原始任务记录重建任意长度PDF的页覆盖和ID集合。"""
    if not tasks or len({row.get("id") for row in tasks}) != len(tasks):
        raise AuditFailure("任务缺失或任务ID重复")
    unique_ids, raw_count, unique_sum, end, empty_tasks = set(), 0, 0, 0, 0
    for row in sorted(tasks, key=lambda item: item["from_page"]):
        if (not isinstance(row.get("id"), str) or not row["id"] or row.get("doc_id") != document_id
                or row.get("progress") != 1 or type(row.get("from_page")) is not int
                or type(row.get("to_page")) is not int or row["from_page"] != end
                or not end < row["to_page"] <= pages or type(row.get("retry_count")) is not int
                or row["retry_count"] < 1 or not isinstance(row.get("chunk_ids"), str)):
            raise AuditFailure("任务归属、完成状态或连续页段不符")
        ids = row["chunk_ids"].split()
        raw_count += len(ids)
        unique_sum += len(set(ids))
        empty_tasks += not ids
        unique_ids.update(ids)
        end = row["to_page"]
    if end != pages or unique_sum != metadata_count or not unique_ids:
        raise AuditFailure("页段覆盖不完整，或逐任务唯一计数之和不等于原生累计计数")
    return unique_ids, {"task_count": len(tasks), "covered_pages": end, "raw_task_chunk_id_count": raw_count,
                        "sum_per_task_unique_chunk_id_count": unique_sum, "task_unique_chunk_id_count": len(unique_ids),
                        "within_task_duplicate_entries": raw_count - unique_sum,
                        "cross_task_duplicate_entries": unique_sum - len(unique_ids), "empty_successful_tasks": empty_tasks,
                        "task_retry_counts": dict(Counter(str(row["retry_count"]) for row in tasks)),
                        "task_unique_ids_sha256": digest(sorted(unique_ids))}


def inspect_chunks(client, dataset_id, document_id, expected_ids):
    """枚举全部REST分块；仅在内存中处理正文，输出内容和图片位置的摘要。"""
    chunks = {}
    expected_count = len(expected_ids)
    for page in range(1, (expected_count + 99) // 100 + 1):
        result = client.request("GET", f"/datasets/{dataset_id}/documents/{document_id}/chunks?page={page}&page_size=100")
        if (result.get("total") != expected_count or result.get("doc", {}).get("id") != document_id
                or not isinstance(result.get("chunks"), list) or not result["chunks"]):
            raise AuditFailure("REST分页总数或文档归属与任务并集不符")
        for row in result["chunks"]:
            ident, positions = row.get("id"), row.get("positions")
            image = (row.get("doc_type_kwd") == "image" and row.get("image_id") == f"{dataset_id}-{ident}"
                     and isinstance(positions, list) and bool(positions)
                     and all(isinstance(position, list) and len(position) == 5
                             and all(type(value) in (int, float) and math.isfinite(value) for value in position)
                             for position in positions))
            if (not isinstance(ident, str) or not ident or ident in chunks or row.get("document_id") != document_id
                    or row.get("dataset_id") not in (dataset_id, [dataset_id]) or row.get("available") is not True
                    or not isinstance(row.get("content"), str) or (not row["content"].strip() and not image)):
                raise AuditFailure("REST块重复、归属错误、不可用，或空正文不满足原生图片规则")
            chunks[ident] = {key: row.get(key) for key in ("id", "document_id", "dataset_id", "content", "positions",
                                                         "available", "doc_type_kwd", "image_id")}
    if set(chunks) != expected_ids:
        raise AuditFailure("REST完整ID集合与任务并集存在缺失或多余")
    return {"unique_available_chunk_count": len(chunks),
            "text_chunk_count": sum(bool(row["content"].strip()) for row in chunks.values()),
            "nontext_image_chunk_count": sum(not row["content"].strip() for row in chunks.values()),
            "api_unique_ids_sha256": digest(sorted(chunks)),
            "chunk_content_positions_and_images_sha256": digest([chunks[key] for key in sorted(chunks)])}


def native_documents(client, dataset_id):
    result = client.request("GET", f"/datasets/{dataset_id}/documents?page=1&page_size=100")
    return result.get("docs", []) if isinstance(result, dict) else result


def snapshot(documents):
    return sorted([{key: row.get(key) for key in ("id", "name", "chunk_count", "size", "update_time")}
                   for row in documents], key=lambda row: row["id"])


def audit(state, ledger, page_report, task_snapshot, client, pdf_root):
    """无论单文档失败与否都保留该题材文档；全局不一致也不能产生通过结论。"""
    errors, rows = [], []
    expected_names = {row["doc_name"] for row in ledger}
    entries = state.get("documents") or {}
    dataset_id = state.get("dataset_id")
    pages_rows = page_report.get("document_pages") or []
    page_map = {row["name"]: row.get("pages") for row in pages_rows}
    groups = task_snapshot.get("documents") or []
    task_map = {row["document_id"]: row.get("tasks", []) for row in groups}
    expected_ids = {entry.get("id") for entry in entries.values()}
    task_ids = [task.get("id") for group in groups for task in group.get("tasks", [])]
    if (len(ledger) != 84 or len(expected_names) != 84 or len(entries) != 84 or set(entries) != expected_names
            or len(expected_ids) != 84 or state.get("corpus_sha256") != digest(ledger)):
        errors.append("冻结语料/导入记录不是相同的84份唯一文档")
    if (page_report.get("documents") != 84 or page_report.get("failed_documents") != 0 or len(pages_rows) != 84
            or len(page_map) != 84 or set(page_map) != {name + ".pdf" for name in expected_names}
            or any(type(value) is not int or value < 1 for value in page_map.values())
            or page_report.get("total_pages") != sum(value for value in page_map.values() if type(value) is int)):
        errors.append("完整84文档页数报告不一致或有失败")
    if (task_snapshot.get("dataset_id") != dataset_id or len(groups) != 84 or len(task_map) != 84
            or set(task_map) != expected_ids or len(task_ids) != len(set(task_ids))):
        errors.append("原始任务快照的知识库/文档集合或全局任务ID不一致")
    if (state.get("parse_profile", {}).get("device") != "gpu"
            or state.get("parse_profile", {}).get("table_auto_rotate") is not True):
        errors.append("正式解析条件缺少GPU和表格自动旋转true")
    documents, semantics = [], None
    try:
        dataset = client.request("GET", f"/datasets/{dataset_id}")
        semantics = parser_config(state["dataset_config"])
        if (dataset.get("id") != dataset_id or dataset.get("embedding_model", dataset.get("embd_id")) != state.get("embedding_model")
                or parser_config(dataset) != semantics):
            raise AuditFailure("原生知识库的Embedding或DeepDOC256解析配置改变")
        documents = native_documents(client, dataset_id)
        if (len(documents) != 84 or len({row["id"] for row in documents}) != 84
                or {row["id"] for row in documents} != expected_ids):
            errors.append("原生知识库不是导入台账对应的84份唯一文档")
    except Exception as error:
        errors.append(str(error) if isinstance(error, AuditFailure) else "原生知识库读取失败：" + type(error).__name__)
    by_id = {row["id"]: row for row in documents}
    initial_snapshot = snapshot(documents)
    for entry in ledger:
        name = entry["doc_name"]
        saved = entries.get(name) or {}
        ident = saved.get("id")
        row = {"doc_name": name, "document_id": ident, "expected_pages": page_map.get(name + ".pdf"),
               "status": "failed", "metadata_chunk_count": None, "unique_available_chunk_count": None,
               "text_chunk_count": None, "nontext_image_chunk_count": None, "cross_task_duplicate_entries": None}
        rows.append(row)
        try:
            if errors:
                raise AuditFailure("全局来源或知识库校验失败，本文档不能计作通过")
            path = (pdf_root / entry["local_path"]).resolve()
            if (not path.is_relative_to(pdf_root.resolve()) or not entry.get("pdf_sha256")
                    or saved.get("sha256") != entry["pdf_sha256"] or saved.get("parse_requested") is not True
                    or pdf_sha256(path) != entry["pdf_sha256"]):
                raise AuditFailure("冻结PDF字节/导入摘要不符或未请求原生解析")
            document = by_id.get(ident) or {}
            row["metadata_chunk_count"] = document.get("chunk_count")
            if (document.get("name") != name + ".pdf" or document.get("run") not in ("DONE", "3", 3)
                    or type(row["metadata_chunk_count"]) is not int or row["metadata_chunk_count"] < 1
                    or parser_config(document) != semantics):
                raise AuditFailure("文档未DONE、没有有效累计块数或解析配置发生变化")
            ids, accounting = task_accounting(task_map.get(ident, []), ident, row["expected_pages"], row["metadata_chunk_count"])
            row.update(accounting)
            row.update(inspect_chunks(client, dataset_id, ident, ids))
            row["status"] = "ok"
        except Exception as error:
            row["error_type"] = type(error).__name__
            row["reason"] = str(error) if isinstance(error, AuditFailure) else "文件或原生数据读取失败：" + type(error).__name__
        print(json.dumps({"doc_name": name, "status": row["status"], "metadata": row["metadata_chunk_count"],
                          "unique": row["unique_available_chunk_count"]}, ensure_ascii=False), flush=True)
    try:
        final_docs = native_documents(client, dataset_id)
        if snapshot(final_docs) != initial_snapshot or any(row.get("run") not in ("DONE", "3", 3) for row in final_docs):
            errors.append("审计期间文档快照/解析状态发生变化")
    except Exception as error:
        errors.append("末尾原生文档快照读取失败：" + type(error).__name__)
    passed = [row for row in rows if row["status"] == "ok"]
    totals = {key: sum(row[key] for row in passed) for key in
              ("metadata_chunk_count", "unique_available_chunk_count", "text_chunk_count", "nontext_image_chunk_count",
               "within_task_duplicate_entries", "cross_task_duplicate_entries", "covered_pages")}
    return {"version": "timeguard-index-audit-v1", "dataset_id": dataset_id,
            "status": "ok" if len(passed) == 84 and not errors else "failed", "planned_denominator": 84,
            "listed_ledger_documents": len(ledger), "audited_documents": len(rows), "passed_documents": len(passed),
            "failed_documents": 84 - len(passed), "failed_listed_documents": len(rows) - len(passed),
            "global_errors": errors, "documents": rows,
            "verified_document_totals": totals, "totals_scope": "仅通过逐文档核验者之和；只有总体status=ok才可冻结完整84索引",
            "index_snapshot": initial_snapshot, "index_snapshot_sha256": digest(initial_snapshot),
            "all_document_content_positions_sha256": digest([{key: row.get(key) for key in
                ("document_id", "status", "chunk_content_positions_and_images_sha256")} for row in rows]),
            "vector_values_audited": False,
            "limitations": ["原生块ID由正文+文档ID产生，空正文图片会碰撞；ID集合一致不证明不同图片内容无损保存。",
                            "metadata是逐任务去重计数之和，unique是当前索引全局唯一数，两者分别报告。",
                            "retry_count默认0，首次领取即加1；1不是一次失败重试。",
                            "校验正文、图片标识和位置摘要，不读取或验证完整向量数值；不调用模型或修改索引。"]}


def main():
    parser = argparse.ArgumentParser(description="完整84份索引只读审计；默认计划，--execute才读REST和写新报告")
    parser.add_argument("--state", type=Path, default=Path("data/ingestion.json"))
    parser.add_argument("--ledger", type=Path, default=Path("data/financebench/documents.jsonl"))
    parser.add_argument("--pages", type=Path, default=Path("reports/corpus_pages.json"))
    parser.add_argument("--tasks", type=Path, required=True, help="保留全部原始任务ID的本机私有快照JSON")
    parser.add_argument("--output", type=Path, default=Path("reports/full_index_audit.json"))
    parser.add_argument("--base-url", default="http://127.0.0.1:19380")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    print(json.dumps({"planned_documents": 84, "execute": args.execute, "model_calls": 0,
                      "output": str(args.output)}, ensure_ascii=False))
    if not args.execute:
        return
    if args.output.exists():
        raise ValueError("报告已存在，拒绝覆盖；重新审计请使用新的输出文件")
    paths = {"ingestion": args.state, "ledger": args.ledger, "pages": args.pages, "tasks": args.tasks}
    sources = {key: {"path": str(path), "sha256": digest(path.read_bytes())} for key, path in paths.items()}
    started = datetime.now(timezone.utc).isoformat()
    result = audit(read_json(args.state), read_jsonl(args.ledger), read_json(args.pages), read_json(args.tasks),
                   Client(args.base_url), args.ledger.parent)
    for key, path in paths.items():
        if digest(path.read_bytes()) != sources[key]["sha256"]:
            result["global_errors"].append("审计期间源文件字节发生变化：" + key)
            result["status"] = "failed"
    result.update(source_files=sources, started_at_utc=started, finished_at_utc=datetime.now(timezone.utc).isoformat(),
                  code_sha256=digest(Path(__file__).read_bytes()),
                  helper_code_sha256={module.__name__: digest(Path(module.__file__).read_bytes()) for module in (common, extend_corpus)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "passed": result["passed_documents"], "failed": result["failed_documents"],
                      "planned_denominator": 84}, ensure_ascii=False))
    raise SystemExit(0 if result["status"] == "ok" else 1)


if __name__ == "__main__":
    main()
