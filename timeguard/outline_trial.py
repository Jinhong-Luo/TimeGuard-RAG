"""冻结反馈、首搜和缺项，在同一题分别对照补搜方法与补搜后自主选块。"""
import argparse
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from .common import Client, digest, read_json, read_jsonl, write_json
from .four_blocks import identity, material
from .four_feedback import review, gaps
from .native_baseline import private_runtime
from .outline_search import OutlineSearch
from .matrix import frozen_index
from .paired_trial import render, run_packets
from .paired_report import compare
from .retrieval_experiment import catalog
from .scope import scope


def trial(source, pools_dir, frozen, out, mode):
    out.mkdir(parents=True,exist_ok=False)
    rows = read_jsonl(source/"results.jsonl")
    pools = read_json(pools_dir/"pools-private.json")
    past = read_json(frozen/"decisions-private.json")
    choices = read_json(frozen/"four-private.json")
    ledger = read_json("data/ingestion.json")
    index = frozen_index(Path("data/ingestion.json"),Path("runs/r0-gpu-retrieval-all"),Path("data/financebench/queries.jsonl"))
    if read_json(source/"manifest.json")["identity"]["freeze"] != index:
        raise ValueError("语料索引发生变化")
    docs = catalog(read_jsonl("data/financebench/documents.jsonl"),ledger)
    cfg = read_json("configs/r0-low.json")
    cfg["dataset_ids"] = [ledger["dataset_id"]]
    client = Client(cfg["base_url"])
    nav = OutlineSearch(client,ledger["dataset_id"],docs)
    _,rt = private_runtime()
    arms = ["native","outline" if mode=="outline" else "adaptive"]
    def select(row):
        qid = row["id"]
        pool,log = pools[qid],past[qid]["log"]
        permitted = scope(row["question"],docs,row.get("as_of"))
        snap, = row["native_trace"]["evidence_layers"]["answers"]
        allowed = set(pool["allowed"])
        if (len(permitted["companies"])!=1 or not allowed or
                not allowed.issubset(set(permitted["document_ids"]) & set(snap["snapshot"]["input"]["tools"]["doc_scope"])) or
                pool["dataset_id"]!=ledger["dataset_id"] or digest(pool["chunks"])!=pool["sha256"]):
            raise ValueError("冻结材料或公司、披露日期范围不一致")
        entries = [e for e in log if e["阶段"]=="定向补搜"]
        history = choices[qid]["history_only"]
        events, routing = [],{}
        before = choices[qid]["feedback"]
        after = history
        extra = []
        if entries:
            lookup, = entries
            q = lookup["request"]["query"]
            if set(lookup["request"]["allowed"])!=allowed:
                raise ValueError("冻结补搜范围不同")
            if mode=="outline":
                extra = nav.search(q,allowed,lambda _:lookup["response"],routing)
                if routing["路径"]=="原生补搜":
                    routing["复用冻结原生补搜"] = True
                    routing["原生补搜次数"] = 0
            else:
                extra = lookup["response"]
            initial = [material(c) for c in pool["chunks"]]
            known = {identity(c) for c in initial}
            candidates = initial+[material(c) for c in extra if identity(c) not in known]
            if any(c["document_id"] not in allowed for c in candidates):
                raise ValueError("候选越界")
            if len(candidates)>len(initial):
                missing = gaps(json.loads(next(e["response"] for e in log if e["阶段"]=="主模型反馈")))
                order = [next(i for i,c in enumerate(initial,1) if identity(c)==identity(v)) for v in history]
                try:
                    picked,_,_ = review(rt,row["question"],missing,candidates,order,events,"补搜后筛选",mode=="adaptive")
                    after = [candidates[i-1] for i in picked]
                except (ValueError,KeyError,TypeError,IndexError,OSError,RuntimeError) as exc:
                    routing["筛选回退"] = type(exc).__name__
        else:
            after = before
        selected = {arms[0]:before,arms[1]:after}
        if any(c["document_id"] not in allowed for cs in selected.values() for c in cs):
            raise ValueError("选择越界")
        return selected,{"目录":routing,"筛选调用":len(events),"筛选用量":[e["usage"] for e in events],
                         "筛选秒":sum(e["seconds"] for e in events),"返回块":len(extra),"触发补搜":bool(entries)}
    with ThreadPoolExecutor(max_workers=2) as pool:
        decisions = list(pool.map(select,rows))
    chosen = {r["id"]:c for r,(c,_) in zip(rows,decisions,strict=True)}
    write_json(out/"chosen-private.json",chosen)
    audits = {r["id"]:d for r,(_,d) in zip(rows,decisions,strict=True)}
    write_json(out/"audit-private.json",audits)
    packets = render(rows,arms,fours=chosen,adaptive=mode=="adaptive")
    for r,p in zip(rows,packets,strict=True):
        qid = r["id"]
        for arm,packet in p.items():
            if list(map(identity,packet["selected"]))!=list(map(identity,chosen[qid][arm])):
                raise ValueError("最终选块顺序不同")
    run_packets(source,out,arms,rows,packets,rt)
    report = compare(out,Path("data/financebench/gold.jsonl"),arms)
    for d in audits.values():
        nav_info = d["目录"]
        if "章节范围" in nav_info:
            nav_info["选用章节数"] = len(nav_info.pop("章节范围"))
    report["补搜与筛选"] = audits
    write_json(out/"summary.json",report)


if __name__=="__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("source","pools","frozen","output"):
        p.add_argument(arg,type=Path)
    p.add_argument("mode",choices=["outline","adaptive"])
    args = p.parse_args()
    trial(args.source,args.pools,args.frozen,args.output,args.mode)
