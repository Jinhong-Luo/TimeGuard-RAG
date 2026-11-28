"""在同一候选快照上重新生成配对答案，逐次记录调用、耗时和实收。"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

from timeguard.worker_bundle import package_files, stage_package
from timeguard.common import digest,read_json,read_jsonl,write_json
from timeguard.evals.baselines.native_baseline import CONTAINER, private_runtime
from timeguard.evals.baselines.native_matrix import judge


def render(rows, arms, reviews=None, pages=None, guards=None, history=None, history_swap=False, fours=None, adaptive=False):
    files = package_files("evidence/final_snapshot.py", "evidence/evidence_layers.py",
                          "retrieval/page_select.py", "evidence/message_budget.py",
                          "evidence/evidence_review.py", "evals/trials/paired_render.py")
    with tempfile.TemporaryDirectory() as directory:
        pkg = Path(directory, "timeguard")
        hashes = stage_package(pkg, files)
        dest = "/tmp/timeguard-paired-" + digest(hashes)[:16]
        subprocess.run(["docker","exec",CONTAINER,"mkdir","-p",dest],check=True)
        subprocess.run(["docker","cp",str(pkg),CONTAINER+":"+dest+"/"],check=True)
    inputs = []
    for row in rows:
        a, = row["native_trace"]["evidence_layers"]["answers"]
        req, = [r for r in a["requests"] if r.get("status")==200]
        inputs.append({k:a[k] for k in ("snapshot","selected","block_hashes")} |
                      {"body":req["body"],"question":row["question"],"arms":arms})
    for item,row in zip(inputs,rows,strict=True):
        if fours is not None:
            item["fours"] = fours[row["id"]]
            item["adaptive"] = adaptive
        if history is not None:
            item["history"] = history[row["id"]]["chunks"]
            item["history_swap"] = history_swap
        if guards is not None:
            item["guard"] = guards[row["id"]]
        if pages is not None:
            item["page_result"] = pages[row["id"]]
        if reviews is not None:
            item["review"] = {k: reviews[row["id"]][k] for k in ("ids","fallback") if k in reviews[row["id"]]}
    cmd = ["docker","exec","-i","-w","/ragflow","-e","PYTHONPATH="+dest+":/ragflow",CONTAINER,
           "python","-m","timeguard.evals.trials.paired_render"]
    proc = subprocess.run(cmd,input=json.dumps(inputs),capture_output=True,text=True)
    if proc.returncode:
        raise RuntimeError(proc.stderr[-1800:])
    outputs = [json.loads(line.partition(" ")[2]) for line in proc.stdout.splitlines() if line.startswith("TIMEGUARD_RENDER ")]
    if len(outputs)!=len(rows):
        raise ValueError("渲染结果不完整")
    return outputs


def generate(rt, body):
    req = urllib.request.Request(rt["base_url"].rstrip("/")+"/chat/completions", json.dumps(body).encode(),
                                 headers={"Authorization":"Bearer "+rt["token"],"Content-Type":"application/json"})
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req,timeout=330) as response:
        frames = [json.loads(line[6:]) for line in response.read().decode().splitlines()
                  if line.startswith("data: ") and line[6:]!="[DONE]"]
    answer = "".join(c["delta"].get("content","") for f in frames for c in f["choices"])
    if not answer.strip() or not any(c.get("finish_reason")=="stop" for f in frames for c in f["choices"]):
        raise ValueError("专用桥没有返回完整回答")
    return answer,frames[0]["id"],next((f["usage"] for f in reversed(frames) if f.get("usage")),None)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source",type=Path)
    p.add_argument("output",type=Path)
    p.add_argument("--arms",nargs="+",default=["original","page_select"])
    p.add_argument("--query-ids",nargs="+")
    p.add_argument("--history-tables",action="store_true",help="允许将唯一匹配行名与年份的合并原表作为待核材料送达")
    p.add_argument("--history-swap",action="store_true",help="允许历史原块占用原证据的位置，记录原先实收块的减少数")
    args = p.parse_args()
    args.output.mkdir(exist_ok=False,parents=True)
    rows = read_jsonl(args.source/"results.jsonl")
    if args.query_ids:
        by_id = {row["id"]:row for row in rows}
        if len(by_id)!=len(rows) or len(set(args.query_ids))!=len(args.query_ids):
            raise ValueError("配对复跑题号重复")
        rows = [by_id[key] for key in args.query_ids]
    _,rt = private_runtime()
    reviews = None
    if "review" in args.arms:
        from timeguard.evidence.evidence_review import inspect_once
        reviews = {}
        for row in rows:
            a, = row["native_trace"]["evidence_layers"]["answers"]
            reviews[row["id"]] = inspect_once(rt,row["question"],a["snapshot"])
            write_json(args.output/"review-private.json",reviews)
    pages = None
    if set(args.arms) & {"page","supplement"}:
        from timeguard.common import Client
        from timeguard.retrieval.filing_pages import FilingPages
        from timeguard.retrieval.catalog import catalog
        from timeguard.evals.baselines.matrix import frozen_index
        freeze = frozen_index(Path("data/ingestion.json"),Path("runs/r0-gpu-retrieval-all"),Path("data/financebench/queries.jsonl"))
        if freeze != read_json(args.source/"manifest.json")["identity"]["freeze"]:
            raise ValueError("用于找页的索引与冻结快照不一致")
        state = read_json("data/ingestion.json")
        docs = catalog(read_jsonl("data/financebench/documents.jsonl"),state)
        index = FilingPages(Client(read_json("configs/r0-low.json")["base_url"]),state["dataset_id"],docs)
        pages = {}
        for row in rows:
            a, = row["native_trace"]["evidence_layers"]["answers"]
            pages[row["id"]] = index.retrieve(row["question"],a["snapshot"],row.get("as_of"))
            write_json(args.output/"pages-private.json",pages)
    guards = None
    if "formula" in args.arms:
        from timeguard.evidence.formula_guard import guard_row
        from timeguard.common import Client
        client=Client(read_json("configs/r0-low.json")["base_url"])
        guards={}
        for row in rows:
            guards[row["id"]]=guard_row(row,client)
            write_json(args.output/"guards-private.json",guards)
    history = None
    if "history_read" in args.arms:
        from timeguard.retrieval.history_read import history_reader
        if args.arms[0] != "original":
            raise ValueError("历史回读需先回放原版以核对完整实收块")
        reader = history_reader(args.source)
        history = {}
        for row in rows:
            history[row["id"]] = reader(row,args.history_tables)
            history[row["id"]]["audit"]["允许待核原表"] = args.history_tables
            write_json(args.output/"history-private.json",history)
    packets = render(rows,args.arms,reviews,pages,guards,history,args.history_swap)
    run_packets(args.source,args.output,args.arms,rows,packets,rt)


def run_packets(source,output,arms,rows,packets,rt,recovery_root=None):
    for arm in arms:
        dest = output/arm
        dest.mkdir()
        previous_file = recovery_root/arm/"results.jsonl" if recovery_root else None
        previous = {r["id"]:r for r in read_jsonl(previous_file)} if previous_file and previous_file.is_file() else {}
        write_json(dest/"manifest.json",{"identity":{"query_ids":[r["id"] for r in rows],"arm":arm,
                   "inputs":[v[arm]["snapshot"]["sha256"] for v in packets],"source":digest((source/"results.jsonl").read_bytes())}})
        def one(pair):
            src,variants = pair
            a = variants[arm]
            body = a.pop("body")
            existing = previous.get(src["id"])
            if existing is not None:
                old, = existing["native_trace"]["evidence_layers"]["answers"]
                old_request, = old["requests"]
                if (old["snapshot"]!=a["snapshot"] or old_request["body"]!=body
                        or old["selected"]!=a["selected"] or old["block_hashes"]!=a["block_hashes"]):
                    raise ValueError("拒绝跨输入复用已完成回答")
                if existing["status"]=="ok" and old_request.get("status")==200:
                    return {**existing,"恢复来源":recovery_root.name}
            visible = a.pop("visible")
            started = time.monotonic()
            req = {"body":body,"status":None}
            a["requests"].append(req)
            trace = {"citation_origin":"model","evidence_layers":{"answers":[a]}}
            row = {k:src[k] for k in ("id","question","split")}
            try:
                answer, request_id, usage = generate(rt, body)
                req.update(status=200,complete_chunks=visible,request_id=request_id)
                row.update(answer=answer,status="ok",usage=usage)
            except Exception as exc:
                row.update(answer=None,status="request_error",error=type(exc).__name__)
            row.update(native_trace=trace,elapsed_seconds=time.monotonic()-started)
            return row
        with ThreadPoolExecutor(max_workers=2) as pool, (dest/"results.jsonl").open("x") as f:
            for row in pool.map(one,zip(rows,packets,strict=True)):
                f.write(json.dumps(row,ensure_ascii=False)+"\n")
                f.flush()
        if judge(dest):
            raise RuntimeError("配对裁判未完成")


if __name__ == "__main__":
    main()
