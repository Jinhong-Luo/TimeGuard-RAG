"""在冻结知识库运行修正引用的原生开发集问答与独立裁判。"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from .common import digest, read_json, read_jsonl, write_json
from .matrix import frozen_index, run_batch
from .native_trial import FINGERPRINTS
from .retrieval_experiment import catalog


def worker(output):
    spec = read_json(output / "spec.json")
    container = "timeguard_ljh-ragflow-cpu-1"
    dest = "/tmp/timeguard-" + digest(spec)[:20]
    with tempfile.TemporaryDirectory() as stage:
        src = Path(stage, "timeguard")
        src.mkdir()
        for name in ("__init__", "common", "score", "scope", "native_trial"):
            shutil.copyfile(Path("timeguard", name + ".py"), src / (name + ".py"))
        subprocess.run(["docker", "exec", container, "mkdir", "-p", dest], check=True)
        subprocess.run(["docker", "cp", str(src), container + ":" + dest + "/"], check=True)
    argv = ["docker", "exec", "-i", "-w", "/ragflow", "-e", "PYTHONPATH=" + dest + ":/ragflow", container,
            "python", "-m", "timeguard.native_trial"]
    with (output / "worker.log").open("w") as log, (output / "results.jsonl").open("x") as rows:
        with subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True) as proc:
            proc.stdin.write(json.dumps(spec))
            proc.stdin.close()
            for line in proc.stdout:
                if line.startswith("TIMEGUARD_RESULT "):
                    row = json.loads(line.partition(" ")[2])
                    rows.write(json.dumps(row, ensure_ascii=False) + "\n")
                    rows.flush()
                    print(row["id"], row["status"], flush=True)
                else:
                    log.write(line)
            if proc.wait():
                raise SystemExit(1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--arm", choices=list("ABCD"), default="A")
    p.add_argument("--finance-prompt", action="store_true")
    p.add_argument("--limit", type=int)
    p.add_argument("--k", type=int, choices=(6, 12, 24), default=6)
    p.add_argument("--weight", type=float, choices=(.3, .5, .7), default=.3)
    p.add_argument("--worker", action="store_true")
    args = p.parse_args()
    out = args.output
    if args.worker:
        worker(out)
        return
    queries_path = Path("data/financebench/queries.jsonl")
    state_path = Path("data/ingestion.json")
    freeze = frozen_index(state_path, Path("runs/r0-gpu-retrieval-all"), queries_path)
    baseline = read_json("runs/all-selected/low/manifest.json")
    config = baseline["identity"]["config"]
    state = read_json(state_path)
    docs = catalog(read_jsonl("data/financebench/documents.jsonl"), state)
    queries = [q for q in read_jsonl(queries_path) if q["split"] == "dev"]
    if args.limit is not None:
        if args.limit < 1:
            p.error("样本数必须为正数")
        queries = queries[:args.limit]
    spec = {"chat_id": baseline["chat_id"], "dataset_id": state["dataset_id"], "model": config["model"],
            "arm": args.arm, "k": args.k, "weight": args.weight, "finance_prompt": args.finance_prompt,
            "documents": docs, "queries": queries}
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "spec.json", spec)
    config["retrieval"].update(page_size=args.k, vector_similarity_weight=args.weight)
    write_json(out / "manifest.json", {"identity": {"query_ids": [q["id"] for q in queries],
               "config": config, "spec_sha256": digest(spec), "freeze": freeze, "upstream_files": FINGERPRINTS,
               "code_sha256": {p: digest(Path(p).read_bytes()) for p in
                               ("timeguard/native_trial.py", "timeguard/scope.py", "timeguard/r1_dev.py")}}})
    audit = Path(".cache/bridge-audit.jsonl")
    ledger = out / "audit.jsonl"
    run_batch("timeguard.r1_dev", ["--output", out, "--worker"], out, "generation", "low", audit, ledger, freeze)
    if len(read_jsonl(out / "results.jsonl")) != len(queries):
        raise ValueError("原生问答未覆盖计划题目")
    run_batch("timeguard.judge", [out, "--concurrency", "2", "--base-url",
              os.environ.get("CODEX_BRIDGE_BASE_URL", "http://172.18.0.1:18864/v1"), "--execute"],
              out, "judge", "medium", audit, ledger, freeze)


if __name__ == "__main__":
    main()

