"""审计已完成的一份原生文档，并在同一知识库扩充冻结84份语料。"""

import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

from .common import Client, digest, read_json, read_jsonl


RESOURCE_FIELDS = {"profile", "task_executors", "max_concurrent_tasks", "max_concurrent_chunk_builders",
                   "total_task_limit", "memory_gib"}
MODEL_FIELDS = {"image_config_digest", "onnxruntime_gpu_version", "torch_version", "torch_cuda", "gpu_uuid",
                "gpu_model", "wheel_lock_sha256", "constraints_sha256"}
FINANCEBENCH_COMMIT = "cc39aeb4afdf33909ee1412188bf89035950c2eb"


def pdf_sha256(path):
    """逐块校验公开PDF，不将整份长财报读入内存。"""
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        if handle.read(4) != b"%PDF":
            raise ValueError("输入不是PDF文件")
        handle.seek(0)
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def semantic_profile(profile):
    """资源差异须另行留痕；旧版未显式写旋转字段时采用固定上游默认true。"""
    current = copy.deepcopy(profile)
    current.setdefault("table_auto_rotate", True)
    expected = {"parser": "DeepDOC", "device": "gpu", "onnx_provider": "CUDAExecutionProvider",
                "visible_gpu_count": 1, "table_auto_rotate": True, "ocr_gpu_mem_limit_mb": 8192,
                "ocr_arena_extend_strategy": "kSameAsRequested", "ocr_gpumem_arena_shrinkage": True}
    if any(current.get(key) != value for key, value in expected.items()) or any(not current.get(key) for key in MODEL_FIELDS):
        raise ValueError("必须保持固定CUDA模型环境、原生DeepDOC、8192MiB arena和表格自动旋转true")
    return {key: value for key, value in current.items() if key not in RESOURCE_FIELDS}


def parser_config(dataset):
    config = dataset.get("parser_config") or {}
    if (dataset.get("chunk_method", dataset.get("parser_id")) != "naive"
            or config.get("chunk_token_num") != 256 or config.get("layout_recognize") != "DeepDOC"):
        raise ValueError("知识库必须使用原生naive/DeepDOC/256参数")
    return config


def validate_parser_artifacts(source_models, final_models, source_parser, final_parser, profile):
    """比较重建前后实际权重清单和解析器证据，不能用相同模型名代替内容一致。"""
    first, last = read_json(source_models), read_json(final_models)
    def model_files(report):
        rows = report.get("files") or []
        required = {"det.onnx", "rec.onnx", "layout.onnx", "tsr.onnx", "ocr.res", "updown_concat_xgb.model"}
        if (len(rows) != len({row.get("name") for row in rows}) or not required.issubset({row.get("name") for row in rows})
                or report.get("image_config_digest") != profile["image_config_digest"]
                or any(type(row.get("bytes")) is not int or row["bytes"] <= 0
                       or not isinstance(row.get("sha256"), str) or len(row["sha256"]) != 64
                       or any(char not in "0123456789abcdef" for char in row["sha256"]) for row in rows)):
            raise ValueError("CUDA模型清单缺失、重复、摘要无效或镜像不一致")
        return sorted([{key: row[key] for key in ("name", "bytes", "sha256")} for row in rows], key=lambda row: row["name"])
    fixed = model_files(first)
    if fixed != model_files(last):
        raise ValueError("重建后原生CUDA模型权重文件发生变化")
    before, after = read_json(source_parser), read_json(final_parser)
    for record in (before, after):
        if (record.get("container_image_config") != profile["image_config_digest"]
                or record.get("upstream_commit") != "8090ffd8ba6098813d7f7ec2c4b8774f36c8423a"
                or record.get("table_auto_rotate_effective") is not True
                or (record.get("table_auto_rotate_env") is None and record.get("default_verified_in_fixed_source") is not True)):
            raise ValueError("缺少固定镜像/源码下表格自动旋转true的真实证据")
    if (not before.get("pdf_parser_source_sha256") or before["pdf_parser_source_sha256"] != after.get("pdf_parser_source_sha256")
            or before.get("ocr_environment") != after.get("ocr_environment")
            or str(after.get("table_auto_rotate_env")).lower() not in ("true", "1")):
        raise ValueError("解析器源码/OCR环境不同，或最终服务未显式启用表格自动旋转")
    return {"model_files_count": len(fixed), "model_files_sha256": digest(fixed),
            "pdf_parser_source_sha256": before["pdf_parser_source_sha256"], "table_auto_rotate": True}


def prepare_inputs(source_state, source_ledger, full_ledger, final_profile, evidence_paths, artifact_paths, task_accounting=None):
    """只读文件校验；不接触问题清单、答案或金标。"""
    paths = {"source_state": source_state, "source_ledger": source_ledger, "full_ledger": full_ledger,
             "final_runtime_profile": final_profile,
             **{f"resource_evidence_{i}": path for i, path in enumerate(evidence_paths)}, **artifact_paths}
    if task_accounting is not None:
        paths["task_accounting"] = task_accounting
    files = {name: {"path": str(path), "sha256": digest(path.read_bytes())} for name, path in paths.items()}
    state, old, full, target = read_json(source_state), read_jsonl(source_ledger), read_jsonl(full_ledger), read_json(final_profile)
    if len(old) != 1 or len(full) != 84 or len({row["doc_name"] for row in full}) != 84 or len(state["documents"]) != 1:
        raise ValueError("扩充仅接受已完成1份文档进入冻结的84份语料")
    if state["corpus_sha256"] != digest(old):
        raise ValueError("原state与原始单文档清单不一致")
    source = old[0]
    matches = [row for row in full if row["doc_name"] == source["doc_name"]]
    if (len(matches) != 1 or {k: v for k, v in source.items() if k != "local_path"}
            != {k: v for k, v in matches[0].items() if k != "local_path"}
            or set(state["documents"]) != {source["doc_name"]}):
        raise ValueError("原文档并非冻结84份中的同一元数据/PDF版本")
    document = state["documents"][source["doc_name"]]
    if document.get("sha256") != source.get("pdf_sha256") or document.get("parse_requested") is not True:
        raise ValueError("原state缺少已请求解析的同一PDF摘要")
    full_root = full_ledger.parent.resolve()
    for row in full:
        path = (full_root / row["local_path"]).resolve()
        if (not path.is_relative_to(full_root) or row.get("source_commit") != FINANCEBENCH_COMMIT
                or not row.get("pdf_sha256") or pdf_sha256(path) != row["pdf_sha256"]):
            raise ValueError("完整84份PDF缺失、超出语料目录或摘要/固定版本不符")
    original_pdf = (source_ledger.parent / source["local_path"]).resolve()
    if (not any(original_pdf.is_relative_to(root) for root in (source_ledger.parent.resolve(), full_root))
            or pdf_sha256(original_pdf) != source["pdf_sha256"]):
        raise ValueError("原单文档PDF的实际字节与冻结全文语料不一致")
    before = state.get("parse_profile") or {}
    if target.get("table_auto_rotate") is not True or semantic_profile(before) != semantic_profile(target):
        raise ValueError("除明确资源调度差异外，原文档与后续语料的模型/解析语义参数必须完全一致")
    artifacts = validate_parser_artifacts(**artifact_paths, profile=target)
    parser_config(state["dataset_config"])
    if not state.get("embedding_model") or not evidence_paths:
        raise ValueError("缺少固定Embedding或资源阶段证据")
    stages = [{"path": str(path), "sha256": digest(path.read_bytes())} for path in evidence_paths]
    for stage, path in zip(stages, evidence_paths):
        value = read_json(path)
        if isinstance(value, list):
            stage["resource_stage_records"] = value
    return {"state": state, "full": full, "source": source, "target_profile": target, "files": files,
            "task_accounting": read_json(task_accounting) if task_accounting is not None else None,
            "stages": stages, "parser_semantics": parser_config(state["dataset_config"]), "parser_artifacts": artifacts,
            "resource_changes": {key: {"before": before.get(key), "after": target.get(key)} for key in sorted(RESOURCE_FIELDS)
                                 if before.get(key) != target.get(key)}}


def reconcile_tasks(accounting, dataset_id, document_id, metadata_count):
    """从原始任务ID重算记账，不信任汇总布尔值；仅适用于本次完整160页源文档。"""
    tasks = accounting.get("tasks") or []
    if (accounting.get("dataset_id") != dataset_id or accounting.get("document_id") != document_id
            or not tasks or len({row.get("id") for row in tasks}) != len(tasks)):
        raise ValueError("任务记账证明的知识库/文档或任务集合不符")
    ids, raw_count, unique_sum, end = set(), 0, 0, 0
    for row in sorted(tasks, key=lambda value: value["from_page"]):
        values = row.get("chunk_ids", "").split()
        if (row.get("doc_id") != document_id or row.get("progress") != 1
                or type(row.get("from_page")) is not int or type(row.get("to_page")) is not int
                or row["from_page"] != end or not end < row["to_page"] <= 160
                or type(row.get("retry_count")) is not int or row["retry_count"] < 1 or not values):
            raise ValueError("任务未全部成功，或页段未连续覆盖同一160页文档")
        end = row["to_page"]
        raw_count += len(values)
        unique_sum += len(set(values))
        ids.update(values)
    if end != 160 or unique_sum != metadata_count:
        raise ValueError("逐任务去重之和不能精确解释原生文档计数，或160页覆盖不完整")
    summary = {"task_count": len(tasks), "source_pages": end, "raw_task_chunk_id_count": raw_count,
               "sum_per_task_unique_chunk_id_count": unique_sum, "global_unique_chunk_id_count": len(ids),
               "within_task_duplicate_entries": raw_count - unique_sum, "cross_task_duplicate_entries": unique_sum - len(ids),
               "global_unique_ids_sha256": digest(sorted(ids)),
               "task_retry_counts": dict(Counter(str(row["retry_count"]) for row in tasks)),
               "retry_count_semantics": "原生字段初始0，每次领取任务即加1；1表示首次领取，不等于一次失败重试。"}
    return ids, summary


def verify_native(prepared, client):
    """仅GET核验现有知识库、文档和所有分块；不上传、不重解析。"""
    state = prepared["state"]
    dataset_id = state["dataset_id"]
    dataset = client.request("GET", f"/datasets/{dataset_id}")
    if (dataset.get("id") != dataset_id or dataset.get("embedding_model", dataset.get("embd_id")) != state["embedding_model"]
            or parser_config(dataset) != prepared["parser_semantics"]):
        raise ValueError("原生知识库的Embedding或完整parser_config已变化")
    doc_id = state["documents"][prepared["source"]["doc_name"]]["id"]

    def get_document():
        response = client.request("GET", f"/datasets/{dataset_id}/documents?page=1&page_size=100")
        docs = response.get("docs", []) if isinstance(response, dict) else response
        if (len(docs) != 1 or docs[0].get("id") != doc_id or docs[0].get("run") not in ("DONE", "3", 3)
                or type(docs[0].get("chunk_count")) is not int or docs[0]["chunk_count"] < 1
                or docs[0].get("name") != prepared["source"]["doc_name"] + ".pdf"):
            raise ValueError("同一知识库必须恰有原1份DONE且含有效分块的文档")
        document = docs[0]
        if "parser_config" in document and parser_config(document) != prepared["parser_semantics"]:
            raise ValueError("原文档解析语义参数与知识库不一致")
        return {key: document.get(key) for key in ("id", "name", "run", "chunk_count", "size", "update_time")}

    snapshot = get_document()
    count = snapshot["chunk_count"]
    task_ids, accounting = None, None
    if prepared.get("task_accounting") is not None:
        task_ids, accounting = reconcile_tasks(prepared["task_accounting"], dataset_id, doc_id, count)
        count = len(task_ids)
    chunks = {}
    for page in range(1, (count + 99) // 100 + 1):
        result = client.request("GET", f"/datasets/{dataset_id}/documents/{doc_id}/chunks?page={page}&page_size=100")
        if (result.get("total") != count or result.get("doc", {}).get("id") != doc_id
                or not result.get("chunks")):
            raise ValueError("原生分页分块总数、文档归属或页面完整性不符")
        for row in result["chunks"]:
            ident = row.get("id")
            positions = row.get("positions")
            valid_image = (row.get("doc_type_kwd") == "image" and row.get("image_id") == f"{dataset_id}-{ident}"
                           and isinstance(positions, list) and bool(positions)
                           and all(isinstance(position, list) and len(position) == 5
                                   and all(type(value) in (int, float) and math.isfinite(value) for value in position)
                                   for position in positions))
            if (not isinstance(ident, str) or not ident or ident in chunks or row.get("document_id") != doc_id
                    or row.get("dataset_id") not in (dataset_id, [dataset_id]) or row.get("available") is not True
                    or not isinstance(row.get("content"), str) or (not row["content"].strip() and not valid_image)):
                raise ValueError("原生分块重复、归属错误、不可用，或空文本块缺少原生图片类型/标识/位置")
            # 原文只用于摘要，不进入公开报告或模型请求。
            chunks[ident] = {key: row.get(key) for key in ("id", "document_id", "dataset_id", "content", "positions", "available",
                                                          "doc_type_kwd", "image_id")}
    if len(chunks) != count or get_document() != snapshot:
        raise ValueError("分块未覆盖完整总数，或核验过程中原文档发生变化")
    if task_ids is not None and set(chunks) != task_ids:
        raise ValueError("当前原生API唯一分块与成功任务ID并集不完全相同，不能以记账差异掩盖丢块")
    return {"dataset_id": dataset_id, "document_snapshot": snapshot, "chunk_count": count,
            "native_document_chunk_count": snapshot["chunk_count"], "unique_available_chunk_count": count,
            "text_chunk_count": sum(bool(row["content"].strip()) for row in chunks.values()),
            "nontext_image_chunk_count": sum(not row["content"].strip() for row in chunks.values()),
            "task_count_reconciliation": accounting,
            "unique_chunk_ids_sha256": digest(sorted(chunks)),
            "chunk_content_and_positions_sha256": digest([chunks[key] for key in sorted(chunks)]),
            "parser_config_sha256": digest(prepared["parser_semantics"]),
            "embedding_model": state["embedding_model"], "read_only_native_check": True}


def materialize(prepared, native, target_state, target_profile, report_path):
    """新state最后写入；所有目标必须不存在，绝不覆盖旧失败或源记录。"""
    targets = [Path(target_state), Path(target_profile), Path(report_path)]
    if len({path.resolve() for path in targets}) != 3 or any(path.exists() for path in targets):
        raise ValueError("扩充输出已存在或路径重叠，请先由操作者审计归档，拒绝覆盖")
    for entry in prepared["files"].values():
        if digest(Path(entry["path"]).read_bytes()) != entry["sha256"]:
            raise ValueError("核验后输入文件发生变化，停止写入")
    state = copy.deepcopy(prepared["state"])
    profile = copy.deepcopy(prepared["target_profile"])
    provenance = {"version": "timeguard-corpus-extension-v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
                  "source_files": prepared["files"], "reused_documents": 1, "new_documents": 83, "final_documents": 84,
                  "source_corpus_sha256": state["corpus_sha256"], "target_corpus_sha256": digest(prepared["full"]),
                  "source_document": prepared["source"]["doc_name"], "native_validation": native,
                  "resource_changes": prepared["resource_changes"], "table_auto_rotate": True,
                  "parser_artifacts": prepared["parser_artifacts"],
                  "table_rotation_source": "固定v0.27.3缺省为true；源字段缺失不代表关闭，后续显式true。",
                  "code_sha256": digest(Path(__file__).read_bytes())}
    profile["corpus_extension"] = provenance
    profile["resource_schedule"] = {"reused_document_original_profile": state["parse_profile"],
                                    "reused_document_stage_evidence": prepared["stages"],
                                    "remaining_83_runtime_profile": prepared["target_profile"],
                                    "说明": "保留160页文档真实资源阶段，不把不同进程/内存配置伪称全库统一资源吞吐。解析算法、模型与旋转保持一致。"}
    state.update(corpus_sha256=provenance["target_corpus_sha256"], parse_profile=profile, corpus_extension=provenance)
    profile_bytes = (json.dumps(profile, ensure_ascii=False, indent=2) + "\n").encode()
    state_bytes = (json.dumps(state, ensure_ascii=False, indent=2) + "\n").encode()
    report = {**provenance, "target_profile_sha256": digest(profile_bytes), "target_state_sha256": digest(state_bytes),
              "说明": "只扩充导入台账；既有1份parse_requested=true保持，后续原生ingest仅上传并解析83份。未运行问答或读取金标。"}
    for path, data in ((targets[1], profile), (targets[2], report), (targets[0], state)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description="审计已完成160页文档并扩充84份语料；默认只校验本地文件")
    parser.add_argument("--source-state", type=Path, default=Path("data/gpu_long_smoke_ingestion.json"))
    parser.add_argument("--source-ledger", type=Path, default=Path("data/long-smoke/documents.jsonl"))
    parser.add_argument("--full-ledger", type=Path, default=Path("data/financebench/documents.jsonl"))
    parser.add_argument("--final-profile", type=Path, default=Path("deploy/parse-profile.gpu.json"))
    parser.add_argument("--evidence", type=Path, action="append", required=True, help="资源阶段/旋转/完成证明JSON，可重复")
    parser.add_argument("--source-models", type=Path, default=Path("reports/deepdoc_model_files.json"))
    parser.add_argument("--final-models", type=Path, required=True)
    parser.add_argument("--source-parser", type=Path, default=Path("reports/gpu_long_smoke_parser_evidence.json"))
    parser.add_argument("--final-parser", "--final-parser-evidence", dest="final_parser", type=Path, required=True)
    parser.add_argument("--task-accounting", type=Path, help="可选原始成功任务ID证明；只有完整机械对账才接受跨任务重复计数")
    parser.add_argument("--state", type=Path, default=Path("data/ingestion.json"))
    parser.add_argument("--profile", type=Path, default=Path("deploy/parse-profile.gpu.corpus.json"))
    parser.add_argument("--report", type=Path, default=Path("reports/corpus_extension.json"))
    parser.add_argument("--base-url", default="http://127.0.0.1:19380")
    parser.add_argument("--execute", action="store_true", help="执行原生只读核验后写新台账，不自动启动解析")
    args = parser.parse_args()
    artifacts = {"source_models": args.source_models, "final_models": args.final_models,
                 "source_parser": args.source_parser, "final_parser": args.final_parser}
    prepared = prepare_inputs(args.source_state, args.source_ledger, args.full_ledger, args.final_profile, args.evidence, artifacts,
                              args.task_accounting)
    print(json.dumps({"source_documents": 1, "target_documents": 84, "new_documents": 83, "execute": args.execute,
                      "table_auto_rotate": True, "resource_changes": prepared["resource_changes"]}, ensure_ascii=False))
    if args.execute:
        native = verify_native(prepared, Client(args.base_url))
        materialize(prepared, native, args.state, args.profile, args.report)
        print("原生分块审计通过，新台账已保存；请使用增强profile交由原生ingest继续83份。")


if __name__ == "__main__":
    main()
