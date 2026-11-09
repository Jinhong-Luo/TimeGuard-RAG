"""在完全相同的 24 个原生候选内比较前六项排序，按重排模型分词器裁剪评分输入。"""
import argparse
import json
from pathlib import Path
import subprocess
import time
import urllib.request

from .common import digest, read_json, read_jsonl, write_json, write_jsonl
from .reranker import CONTAINER, REVISION
from .retrieval_experiment import summarize


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    manifest = read_json(args.source / "manifest.json")
    reports = read_json(args.source / "summary.json")
    if reports["manifest_sha256"] != digest(manifest):
        raise ValueError("调优来源的清单摘要不一致")
    idx = manifest["variants"].index(["B", 24, .3, ""])
    source = args.source / f"{idx}.jsonl"
    if digest(source.read_bytes()) != reports["variants"][idx]["rows_sha256"]:
        raise ValueError("调优候选文件来源不匹配")
    inspect = json.loads(subprocess.check_output(["docker", "inspect", CONTAINER]))[0]
    host = inspect["NetworkSettings"]["Networks"]["timeguard_ljh_ragflow"]["IPAddress"]
    model = read_json("reports/r1-reranker.json")
    if model["revision"] != REVISION or inspect["Image"] != model["image"]:
        raise ValueError("运行镜像或模型版本不匹配")
    write_json(out / "manifest.json", {"source_sha256": digest(source.read_bytes()), "model": model,
               "truncate": True, "candidate_k": 24, "return_k": 6, "code_sha256": digest(Path(__file__).read_bytes())})
    qs = {r["id"]: r for r in read_jsonl("data/financebench/queries.jsonl")}
    original, rescored = [], []
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with (out / "reranked.jsonl").open("x") as save:
        for r in read_jsonl(source):
            if r["status"] != "ok":
                raise ValueError("来源有检索请求失败，无法进行配对比较")
            chunks = r["retrieved_chunks"]
            original.append({**r, "retrieved_chunks": chunks[:6]})
            start = time.monotonic()
            fresh = {**r, "retrieved_chunks": []}
            try:
                if chunks:
                    data = json.dumps({"query": qs[r["id"]]["question"], "texts": [c["content"] for c in chunks],
                                       "truncate": True, "raw_scores": False}).encode()
                    req = urllib.request.Request(f"http://{host}/rerank", data, headers={"Content-Type": "application/json"})
                    with opener.open(req, timeout=180) as response:
                        scores = json.load(response)
                    if sorted(o["index"] for o in scores) != list(range(len(chunks))):
                        raise ValueError("重排服务没有逐项返回候选")
                    scores.sort(key=lambda s: (-s["score"], s["index"]))
                    fresh["retrieved_chunks"] = [chunks[s["index"]] for s in scores[:6]]
                fresh["elapsed_seconds"] += time.monotonic()-start
            except Exception as exc:
                fresh.update(status="request_error", error=str(exc)[:300], elapsed_seconds=r["elapsed_seconds"]+time.monotonic()-start)
            save.write(json.dumps(fresh, ensure_ascii=False)+"\n")
            save.flush()
            rescored.append(fresh)
    write_jsonl(out / "native.jsonl", original)
    docs = {r["doc_name"].casefold(): r for r in read_jsonl("data/financebench/documents.jsonl")}
    gold = {r["id"]: r for r in read_jsonl("data/financebench/gold.jsonl")}
    write_json(out / "summary.json", {"manifest_sha256": digest((out/"manifest.json").read_bytes()),
            "native": summarize(original, gold, docs), "reranked": summarize(rescored, gold, docs),
            "native_rows_sha256": digest(original), "reranked_rows_sha256": digest(rescored)})


if __name__ == "__main__":
    main()

