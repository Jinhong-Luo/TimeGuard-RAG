"""将原始 PDF 上传到 RAGFlow 原生知识库并启动原生解析。"""

import argparse
from pathlib import Path

from .common import Client, digest, read_json, read_jsonl, write_json


def main():
    parser = argparse.ArgumentParser(description="导入全量 FinanceBench 文档；使用原生解析和索引")
    parser.add_argument("--base-url", default="http://127.0.0.1:19380")
    parser.add_argument("--data", default="data/financebench")
    parser.add_argument("--embedding-model", required=True, help="已在 RAGFlow 配置的固定 Embedding 模型标识")
    parser.add_argument("--name", default="TimeGuard-FinanceBench-R0")
    parser.add_argument("--state", default="data/ingestion.json")
    parser.add_argument("--parse-profile", help="冻结本次原生解析的CPU/GPU与资源条件JSON")
    parser.add_argument("--status", action="store_true", help="只检查解析进度")
    args = parser.parse_args()
    client = Client(args.base_url)
    root = Path(args.data)
    ledger = read_jsonl(root / "documents.jsonl")
    state_path = Path(args.state)
    state = read_json(state_path) if state_path.exists() else None
    corpus_hash = digest(ledger)
    profile = read_json(args.parse_profile) if args.parse_profile else None
    if state and (state["corpus_sha256"] != corpus_hash or state["embedding_model"] != args.embedding_model):
        raise ValueError("已有导入记录与文档或 Embedding 配置不一致，请使用新的 --state 与知识库名称")
    if state and not args.status and state.get("parse_profile") != profile:
        raise ValueError("已有导入记录的CPU/GPU解析条件不同，必须使用新的知识库和导入记录")
    if args.status:
        if not state:
            raise ValueError("尚无导入记录")
        result = client.request("GET", f"/datasets/{state['dataset_id']}/documents?page=1&page_size=100")
        documents = result.get("docs", []) if isinstance(result, dict) else result
        status = [{"id": d["id"], "name": d["name"], "run": d.get("run"), "progress": d.get("progress"),
                   "chunk_count": d.get("chunk_count"), "progress_msg": d.get("progress_msg")} for d in documents]
        write_json(root / "parse_status.json", status)
        print(__import__("json").dumps(status, ensure_ascii=False, indent=2))
        return
    for doc in ledger:
        path = root / doc["local_path"]
        if not path.exists() or digest(path.read_bytes()) != doc["pdf_sha256"]:
            raise ValueError(f"PDF 缺失或校验失败：{path}，请先运行 data --pdfs")
    if not state:
        dataset = client.request("POST", "/datasets", {"name": args.name, "embedding_model": args.embedding_model,
                                  "permission": "me", "chunk_method": "naive",
                                  "parser_config": {"chunk_token_num": 256, "layout_recognize": "DeepDOC"}})
        state = {"dataset_id": dataset["id"], "embedding_model": args.embedding_model, "corpus_sha256": corpus_hash,
                 "parse_profile": profile,
                 "dataset_config": dataset, "documents": {}}
        write_json(state_path, state)
    for doc in ledger:
        if doc["doc_name"] in state["documents"]:
            continue
        uploaded = client.upload(state["dataset_id"], root / doc["local_path"])
        state["documents"][doc["doc_name"]] = {"id": uploaded[0]["id"], "sha256": doc["pdf_sha256"], "parse_requested": False}
        write_json(state_path, state)
    for name, document in state["documents"].items():
        if not document["parse_requested"]:
            client.request("POST", f"/datasets/{state['dataset_id']}/chunks", {"document_ids": [document["id"]]})
            document["parse_requested"] = True
            write_json(state_path, state)
    print(f"已上传并请求解析 {len(state['documents'])} 份文档，知识库：{state['dataset_id']}；请用 --status 确认解析完成")


if __name__ == "__main__":
    main()
