"""冻结完整开发集已执行的搜索和反馈，只改变最后一次筛选的块数。"""
import argparse
import json
from copy import copy
from pathlib import Path

from timeguard.common import digest, read_json, read_jsonl
from timeguard.evals.trials.four_blocks import identity, material
from timeguard.evals.trials.four_feedback import gaps, review
from timeguard.evals.trials.full_feedback import Experiment, answer, cached, grade, packet_key, packets, save
from timeguard.evals.analysis.layer_report import page_set
from timeguard.retrieval.scope import scope


ARMS = ("fixed","adaptive")


def decisions(exp, base, qid):
    folder = base/qid
    pool = read_json(folder/"pool.json")
    previous = read_json(folder/"feedback.json")
    old = previous["choices"]["feedback"]
    history = previous["choices"]["history_only"]
    snap, = exp.rows[qid]["native_trace"]["evidence_layers"]["answers"]
    query = exp.queries[qid]
    plan = scope(query["question"],exp.docs,query.get("as_of"))
    allowed = set(pool["allowed"])
    if (len(plan["companies"])!=1 or not allowed or
            not allowed.issubset(set(plan["document_ids"]) & set(snap["snapshot"]["input"]["tools"]["doc_scope"])) or
            digest(pool)!=read_json(folder/"selection.json")["pool_sha256"]):
        raise ValueError("冻结候选范围或指纹有变")
    logs = previous["audit"]["log"]
    candidates = list(pool["chunks"])
    known = {identity(c) for c in candidates}
    extra = [e for e in logs if e["阶段"]=="定向补搜"]
    if extra:
        event, = extra
        if set(event["request"]["allowed"])!=allowed:
            raise ValueError("冻结补搜文档范围不同")
        candidates.extend(material(c) for c in event.get("response",[]) if identity(c) not in known)
    if any(c["document_id"] not in allowed for c in [*candidates,*old,*history]):
        raise ValueError("最终候选越界")
    log = []
    new = old
    reason = None
    if len(candidates)>len(known):
        missing = gaps(json.loads(next(e["response"] for e in logs if e["阶段"]=="主模型反馈")))
        ids = {identity(c):i for i,c in enumerate(candidates,1)}
        try:
            picks,_,_ = review(exp.rt,query["question"],missing,candidates,[ids[identity(c)] for c in history],
                               log,"补搜后筛选",adaptive=True)
            new = [candidates[i-1] for i in picks]
        except (OSError,KeyError,TypeError,ValueError,RuntimeError,IndexError) as exc:
            reason = type(exc).__name__
            new = history
    return {"choices":{"fixed":old,"adaptive":new},"log":log,"fallback":reason}


def run(source, base, out):
    exp = Experiment(source,out,4,adaptive=True)
    manifest = read_json(base/"manifest.json")
    for key in ("source","freeze","questions"):
        if manifest[key]!=read_json(out/"manifest.json")[key]:
            raise ValueError("原生对照与本轮身份不符")
    cached_manifest = cached(out/"pair.json",lambda:{"base":str(base.resolve()),"version":1})
    if cached_manifest!={"base":str(base.resolve()),"version":1}:
        raise ValueError("已启动的配对目录不可跨来源重用")
    active = []
    for qid in sorted(exp.queries):
        if (base/qid/"scope.json").exists():
            q = exp.queries[qid]
            plan = scope(q["question"],exp.docs,q.get("as_of"))
            if len(plan["companies"])==1 and plan["document_ids"]:
                raise ValueError("历史澄清与当前公司范围不符")
            save(out/qid/"answers/scope.json",{**q,"answer":"请明确要查询的公司。","status":"ok","reference":{"chunks":[]}})
            save(out/qid/"assignment.json",{k:"scope" for k in ARMS})
        else:
            active.append(qid)
    exp.parallel(lambda q:cached(out/q/"choices.json",lambda:decisions(exp,base,q)),active)
    packets(exp,active,ARMS,lambda q:read_json(out/q/"choices.json")["choices"],"packets.json")
    def generate(q):
        batch = read_json(out/q/"packets.json")
        historical = read_json(base/q/"packets.json")["feedback"]
        if packet_key(batch["fixed"])!=packet_key(historical):
            raise ValueError("固定四块对照与原请求不一致")
        for p in batch.values():
            answer(exp,q,p)
        save(out/q/"assignment.json",{k:packet_key(p) for k,p in batch.items()})
    exp.parallel(generate,active)
    grade(exp)
    labels = {r["id"]:r for r in read_jsonl("data/financebench/gold.jsonl")}
    rows = {}
    for q in exp.queries:
        d = rows[q] = {}
        assignment = read_json(out/q/"assignment.json")
        if q in active:
            selection = read_json(out/q/"choices.json")
            d.update(筛选用量=[e["usage"] for e in selection["log"]],筛选秒=sum(e["seconds"] for e in selection["log"]),
                     回退=selection["fallback"])
        for arm,key in assignment.items():
            p = read_json(out/q/"answers"/(key+".json"))
            score = read_json(out/q/"judges"/(key+".json"))
            a = p.get("native_trace",{}).get("evidence_layers",{}).get("answers",[{}])[0]
            received = a.get("requests",[{}])[0].get("complete_chunks",[])
            target = {(e["doc_name"].casefold(),e["evidence_page_num"]+1) for e in labels[q]["evidence"]}
            d[arm] = {"判对":score["answer_correct"],"引用支持":score["citation_support"],
                      "达标":score["answer_correct"] is True and score["citation_support"] is True,
                      "选中块":len(a.get("selected",[])),"实收块":len(received),
                      "实收题页":len(page_set(received)[0] & target),"回答用量":p.get("usage"),"回答秒":p.get("elapsed_seconds"),
                      "裁判用量":score.get("usage"),"裁判秒":score.get("elapsed_seconds")}
    gains = [q for q,r in rows.items() if r["adaptive"]["达标"] and not r["fixed"]["达标"]]
    losses = [q for q,r in rows.items() if r["fixed"]["达标"] and not r["adaptive"]["达标"]]
    replica = copy(exp)
    replica.root = out/"confirmation"
    def repeat(q):
        batch = read_json(out/q/"packets.json")
        for p in batch.values():
            answer(replica,q,p)
        save(replica.root/q/"assignment.json",{k:packet_key(p) for k,p in batch.items()})
    exp.parallel(repeat,gains+losses)
    grade(replica,gains+losses)
    repeated = {}
    for q in gains+losses:
        repeated[q] = {}
        for k,key in read_json(replica.root/q/"assignment.json").items():
            score = read_json(replica.root/q/"judges"/(key+".json"))
            repeated[q][k] = score["answer_correct"] is True and score["citation_support"] is True
    counts = {k:sum(v[k]["达标"] for v in rows.values()) for k in ARMS}
    assignments = [set(read_json(out/q/"assignment.json").values()) for q in rows]
    save(out/"summary.json",{"题数":len(rows),"联合达标":counts,"救回":gains,"损失":losses,
         "独立重答":repeated,"逐题":rows,"补搜新增请求":0,
         "首次实际回答":sum(len(keys-{"scope"}) for keys in assignments),
         "首次实际裁判":sum(map(len,assignments)),
         "失败次数":len(list(out.rglob("*.failure-*.json")))})


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source","base","output"):
        parser.add_argument(name,type=Path)
    args = parser.parse_args()
    run(args.source,args.base,args.output)
