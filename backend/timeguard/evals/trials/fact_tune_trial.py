"""沿用事先固定的三道调优题，筛查事实筛选和完整送达。"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

from timeguard.common import Client, digest, read_json, read_jsonl
from timeguard.evals.trials.fact_final_trial import ARMS, decide, render_packets, report, confirm
from timeguard.evals.trials.four_blocks import material
from timeguard.evals.trials.full_feedback import answer, cached, grade, packet_key, save
from timeguard.evals.baselines.matrix import frozen_index
from timeguard.evals.baselines.native_baseline import private_runtime
from timeguard.retrieval.catalog import catalog


def run(source,pools,frozen,out):
    rows = {r["id"]:r for r in read_jsonl(source/"results.jsonl")}
    all_queries = {q["id"]:q for q in read_jsonl("data/financebench/queries.jsonl")}
    queries = {q:all_queries[q] for q in rows}
    if len(queries)!=3 or any(q["split"]!="tune" for q in queries.values()):
        raise ValueError("必须使用预先封存的三道调优题")
    ledger = read_json("data/ingestion.json")
    freeze = frozen_index(Path("data/ingestion.json"),Path("runs/r0-gpu-retrieval-all"),Path("data/financebench/queries.jsonl"))
    if freeze!=read_json(source/"manifest.json")["identity"]["freeze"]:
        raise ValueError("调优索引发生变化")
    config = read_json("configs/r0-low.json")
    config["dataset_ids"]=[ledger["dataset_id"]]
    _,rt = private_runtime()
    exp = SimpleNamespace(source=source,root=out,queries=queries,rows=rows,rt=rt,config=config,
         client=Client(config["base_url"]),docs=catalog(read_jsonl("data/financebench/documents.jsonl"),ledger),adaptive=False)
    def parallel(action,ids):
        with ThreadPoolExecutor(max_workers=3) as workers:
            list(workers.map(action,ids))
    exp.parallel=parallel
    base=out/"frozen"
    previous=read_json(frozen/"four-private.json")
    audits=read_json(frozen/"decisions-private.json")
    stored=read_json(pools/"pools-private.json")
    trace={r["id"]:r for r in read_jsonl(frozen/"feedback/results.jsonl")}
    protocol={"source":digest((source/"results.jsonl").read_bytes()),"freeze":freeze,"pools":digest(stored),
              "choices":digest(previous),"version":1}
    if cached(out/"manifest.json",lambda:protocol)!=protocol:
        raise ValueError("调优续跑身份不符")
    for q in rows:
        pool=stored[q]
        if digest(pool["chunks"])!=pool["sha256"] or pool["dataset_id"]!=ledger["dataset_id"]:
            raise ValueError("调优首搜指纹不同")
        pool={"allowed":pool["allowed"],"chunks":[material(c) for c in pool["chunks"]]}
        save(base/q/"pool.json",pool)
        save(base/q/"selection.json",{"pool_sha256":digest(pool)})
        save(base/q/"feedback.json",{"choices":previous[q],"audit":audits[q]})
        a,=trace[q]["native_trace"]["evidence_layers"]["answers"]
        req,=a["requests"]
        p={**a,"body":req["body"],"visible":req["complete_chunks"]}
        save(base/q/"packets.json",{"feedback":p})
    parallel(lambda q:cached(out/q/"choices.json",lambda:decide(exp,base,q)),rows)
    render_packets(exp,base,list(rows))
    def generate(q):
        batch=read_json(out/q/"packets.json")
        for packet in batch.values():
            answer(exp,q,packet)
        save(out/q/"assignment.json",{k:packet_key(p) for k,p in batch.items()})
    parallel(generate,rows)
    grade(exp)
    summary=report(exp,list(rows))
    confirm(exp,summary)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ("source","pools","frozen","output"):
        p.add_argument(arg,type=Path)
    a=p.parse_args()
    run(a.source,a.pools,a.frozen,a.output)
