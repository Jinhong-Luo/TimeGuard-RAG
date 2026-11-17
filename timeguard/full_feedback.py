"""全开发集的补搜配对实验。每个已完成的外部步骤单独保存，允许原条件续跑。"""
import argparse
from copy import copy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time

from .common import Client, digest, read_json, read_jsonl, write_json
from .four_blocks import material, scoped_search, select_four
from .four_feedback import feedback
from .judge import judge_one
from .paired_trial import generate, render
from .matrix import frozen_index
from .native_baseline import private_runtime
from .retrieval_experiment import catalog
from .scope import scope


ARMS = ("no_feedback","history_only","feedback")


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    write_json(tmp,value)
    tmp.replace(path)


def cached(path, action):
    if path.is_file():
        return read_json(path)
    value = action()
    save(path,value)
    return value


class Experiment:
    def __init__(self, source, root, workers):
        self.source, self.root, self.workers = source, root, workers
        self.queries = {q["id"]:q for q in read_jsonl("data/financebench/queries.jsonl") if q["split"]=="dev"}
        self.rows = {r["id"]:r for r in read_jsonl(source/"results.jsonl")}
        if len(self.queries)!=39 or self.queries.keys()!=self.rows.keys():
            raise ValueError("开发集必须完整覆盖三十九道不同题")
        state = read_json("data/ingestion.json")
        self.docs = catalog(read_jsonl("data/financebench/documents.jsonl"),state)
        self.config = read_json("configs/r0-low.json")
        self.config["dataset_ids"] = [state["dataset_id"]]
        self.client = Client(self.config["base_url"])
        _,self.rt = private_runtime()
        freeze = frozen_index(Path("data/ingestion.json"),Path("runs/r0-gpu-retrieval-all"),Path("data/financebench/queries.jsonl"))
        if freeze!=read_json(source/"manifest.json")["identity"]["freeze"]:
            raise ValueError("原生索引与原始快照不一致")
        self.root.mkdir(exist_ok=True,parents=True)
        identity = {"source":digest((source/"results.jsonl").read_bytes()), "freeze":freeze,
                    "questions":digest(self.queries),"protocol":"开发集补搜配对第一版"}
        if cached(root/"manifest.json",lambda:identity)!=identity:
            raise ValueError("续跑来源或语料已变")

    def parallel(self, action, ids):
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            list(pool.map(action,ids))

    def select(self,qid):
        row = self.queries[qid]
        folder = self.root/qid
        plan = scope(row["question"],self.docs,row.get("as_of"))
        if len(plan["companies"])!=1 or not plan["document_ids"]:
            save(folder/"scope.json",{"allowed":[]})
            return
        a, = self.rows[qid]["native_trace"]["evidence_layers"]["answers"]
        allowed = sorted(set(plan["document_ids"]) & set(a["snapshot"]["input"]["tools"]["doc_scope"] or []))
        if not allowed:
            raise ValueError("合法范围与原生范围无交集")
        def retrieve():
            start = time.monotonic()
            chunks = scoped_search(self.client,self.config,row["question"],allowed)
            return {"allowed":allowed,"chunks":[material(c) for c in chunks],"seconds":time.monotonic()-start}
        p = cached(folder/"pool.json",retrieve)
        if p["allowed"]!=allowed or any(c["document_id"] not in allowed for c in p["chunks"]):
            raise ValueError("初搜范围不一致")
        def choose():
            log = []
            picks = select_four(self.rt,row["question"],p["chunks"],log)
            return {"ids":picks,"log":log,"pool_sha256":digest(p)}
        cached(folder/"selection.json",choose)


def packet_key(packet):
    return digest([packet["body"],packet["block_hashes"],packet["visible"]])


def packets(exp, ids, arms, chosen, filename):
    todo = [qid for qid in ids if not (exp.root/qid/filename).exists()]
    for offset in range(0,len(todo),6):
        batch = todo[offset:offset+6]
        items = render([exp.rows[qid] for qid in batch],arms,fours={qid:chosen(qid) for qid in batch})
        for qid,item in zip(batch,items,strict=True):
            expected = chosen(qid)
            for arm,packet in item.items():
                if [(c["document_id"],c["id"]) for c in expected[arm]] != [
                        (c["document_id"],c["id"]) for c in packet["selected"]]:
                    raise ValueError("提交给原生节点的块顺序有变")
            save(exp.root/qid/filename,item)


def answer(exp,qid,packet):
    key = packet_key(packet)
    folder = exp.root/qid/"answers"
    def call():
        start = time.monotonic()
        try:
            txt,request_id,usage = generate(exp.rt,packet["body"])
        except Exception as exc:
            save(folder/f"{key}.failure-{time.time_ns()}.json",{"error":type(exc).__name__,"seconds":time.monotonic()-start})
            raise
        a = {k:v for k,v in packet.items() if k not in ("body","visible","requests")}
        a["requests"] = [{"body":packet["body"],"status":200,"request_id":request_id,"complete_chunks":packet["visible"]}]
        return {"id":qid,"question":exp.queries[qid]["question"],"split":"dev","answer":txt,"status":"ok",
                "usage":usage,"elapsed_seconds":time.monotonic()-start,
                "native_trace":{"citation_origin":"model","evidence_layers":{"answers":[a]}}}
    return cached(folder/(key+".json"),call)


def trial(exp):
    ids = sorted(exp.queries)
    active = [qid for qid in ids if not (exp.root/qid/"scope.json").exists()]
    def start_choice(qid):
        pool = read_json(exp.root/qid/"pool.json")
        selected = read_json(exp.root/qid/"selection.json")
        if digest(pool)!=selected["pool_sha256"]:
            raise ValueError("共享初搜池指纹不一致")
        return {"no_feedback":[pool["chunks"][i-1] for i in selected["ids"]]}
    packets(exp,active,["no_feedback"],start_choice,"initial-packet.json")
    def research(qid):
        folder = exp.root/qid
        packet = read_json(folder/"initial-packet.json")["no_feedback"]
        base = answer(exp,qid,packet)
        req, = base["native_trace"]["evidence_layers"]["answers"][0]["requests"]
        def decide():
            pool = read_json(folder/"pool.json")
            ids = read_json(folder/"selection.json")["ids"]
            plan,record = feedback(exp.rt,exp.queries[qid],pool,ids,req["complete_chunks"],
                                   lambda q:scoped_search(exp.client,exp.config,q,pool["allowed"]))
            return {"choices":plan,"audit":record}
        cached(folder/"feedback.json",decide)
    exp.parallel(research,active)
    def final_choice(qid):
        return read_json(exp.root/qid/"feedback.json")["choices"]
    packets(exp,active,list(ARMS),final_choice,"packets.json")
    def complete(qid):
        folder = exp.root/qid
        if (folder/"scope.json").exists():
            row = {**exp.queries[qid],"answer":"请明确要查询的公司。","status":"ok","reference":{"chunks":[]}}
            save(folder/"answers"/"scope.json",row)
            save(folder/"assignment.json",{arm:"scope" for arm in ARMS})
            return
        final = read_json(folder/"packets.json")
        if packet_key(final["no_feedback"])!=packet_key(read_json(folder/"initial-packet.json")["no_feedback"]):
            raise ValueError("无反馈对照发生变化")
        for packet in final.values():
            answer(exp,qid,packet)
        save(folder/"assignment.json",{arm:packet_key(final[arm]) for arm in ARMS})
    exp.parallel(complete,ids)


def grade(exp,ids=None):
    labels = {r["id"]:r for r in read_jsonl("data/financebench/gold.jsonl")}
    work = [(qid,key) for qid in sorted(exp.queries if ids is None else ids)
            for key in sorted(set(read_json(exp.root/qid/"assignment.json").values()))]
    def one(pair):
        qid,key = pair
        folder = exp.root/qid
        def call():
            generated = read_json(folder/"answers"/(key+".json"))
            result = judge_one(labels[qid],generated,key,exp.rt["base_url"],exp.rt["token"],360)
            if result["judge_status"]!="ok":
                save(folder/"judges"/f"{key}.failure-{time.time_ns()}.json",result)
                raise RuntimeError("裁判失败，按相同答案续跑")
            return result
        cached(folder/"judges"/(key+".json"),call)
    exp.parallel(one,work)


def confirm(exp):
    from .full_feedback_report import make
    labels = {r["id"]:r for r in read_jsonl("data/financebench/gold.jsonl")}
    report = make(exp.root,labels)
    save(exp.root/"dev39-report.json",report)
    keys = set()
    for name in ("完整流程对无反馈","仅补搜对只回看"):
        change = report["全量"][name]
        keys.update(change["救回"]+change["损失"])
    replica = copy(exp)
    replica.root = exp.root/"confirmation"
    def repeat(qid):
        fixed = read_json(exp.root/qid/"packets.json")
        assignment = read_json(exp.root/qid/"assignment.json")
        for arm,p in fixed.items():
            if packet_key(p)!=assignment[arm]:
                raise ValueError("复验消息或引用映射与首次不同")
            answer(replica,qid,p)
        save(replica.root/qid/"assignment.json",assignment)
    exp.parallel(repeat,sorted(keys))
    grade(replica,keys)
    repeated = {}
    calls = 0
    for qid in sorted(keys):
        repeated[qid] = {}
        assignment = read_json(replica.root/qid/"assignment.json")
        calls += len(set(assignment.values()))
        for arm,key in assignment.items():
            score = read_json(replica.root/qid/"judges"/(key+".json"))
            repeated[qid][arm] = {"判对":score["answer_correct"],"引用支持":score["citation_support"],
                                  "达标":score["answer_correct"] is True and score["citation_support"] is True}
    report["联合指标翻转题复验"] = {"题数":len(keys),"实际回答":calls,"实际裁判":calls,"逐题":repeated}
    save(exp.root/"dev39-report.json",report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source",type=Path)
    parser.add_argument("root",type=Path)
    parser.add_argument("stage",choices=["select","trial","grade","confirm"])
    parser.add_argument("--workers",type=int,choices=range(1,5),default=4)
    args = parser.parse_args()
    exp = Experiment(args.source,args.root,args.workers)
    if args.stage=="select":
        exp.parallel(exp.select,sorted(exp.queries))
    elif args.stage=="trial":
        trial(exp)
    elif args.stage=="grade":
        grade(exp)
    else:
        confirm(exp)


if __name__=="__main__":
    main()
