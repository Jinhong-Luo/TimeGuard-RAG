"""完整三十九题：冻结搜索，分开对照事实筛选和事实完整送达。"""
import argparse
from copy import copy, deepcopy
import json
from pathlib import Path
import time
from urllib.parse import quote

from .common import digest, read_json, read_jsonl
from .fact_selection import prepare, received_facts, select
from .formula_guard import check as formula_check
from .four_blocks import identity, material
from .four_feedback import gaps
from .full_feedback import Experiment, answer, cached, grade, packet_key, packets, save
from .layer_report import page_set
from .scope import scope

ARMS = ("fixed", "fact_select", "fact_deliver")


def frozen(exp, base, qid):
    folder = base/qid
    pool = read_json(folder/"pool.json")
    prior = read_json(folder/"feedback.json")
    snap, = exp.rows[qid]["native_trace"]["evidence_layers"]["answers"]
    q = exp.queries[qid]
    plan = scope(q["question"],exp.docs,q.get("as_of"))
    allowed = set(pool["allowed"])
    if (len(plan["companies"])!=1 or not allowed or
        not allowed.issubset(set(plan["document_ids"]) & set(snap["snapshot"]["input"]["tools"]["doc_scope"])) or
        digest(pool)!=read_json(folder/"selection.json")["pool_sha256"]):
        raise ValueError("冻结材料身份或范围发生变化")
    candidates = list(pool["chunks"])
    known = {identity(c) for c in candidates}
    logs = prior["audit"]["log"]
    lookups = [e for e in logs if e["阶段"]=="定向补搜"]
    if lookups:
        event, = lookups
        if set(event["request"]["allowed"])!=allowed:
            raise ValueError("补搜范围变化")
        for c in event.get("response",[]):
            if identity(c) not in known:
                candidates.append(material(c))
                known.add(identity(c))
    old = prior["choices"]["feedback"]
    history = prior["choices"]["history_only"]
    if any(c["document_id"] not in allowed for c in [*candidates,*old,*history]):
        raise ValueError("材料超出公司或截止日范围")
    missing = gaps(json.loads(next(e["response"] for e in logs if e["阶段"]=="主模型反馈"))) if lookups else []
    return pool, candidates, old, history, missing, len(candidates)>len(pool["chunks"])


def decide(exp, base, qid):
    pool, candidates, old, history, missing, active = frozen(exp,base,qid)
    result = {"choices":{k:old for k in ARMS},"log":[],"reads":[],"facts":[],"fallback":None,
              "candidate_sha256":digest(candidates),"candidate_count":len(candidates),"unresolved":[]}
    if not active:
        return result
    ids = {identity(c):i for i,c in enumerate(candidates,1)}
    try:
        decision = select(exp.rt,exp.queries[qid]["question"],missing,candidates,
                          [ids[identity(c)] for c in history],result["log"])
        result.update({k:decision[k] for k in ("facts","unresolved","rejected")})
        chosen = [candidates[i-1] for i in decision["ids"]]
        originals = {}
        dataset = exp.config["dataset_ids"][0]
        for c in chosen:
            event = {"document_id":c["document_id"],"id":c["id"],"failed":False}
            result["reads"].append(event)
            started = time.monotonic()
            try:
                data = exp.client.request("GET",f"/datasets/{dataset}/documents/{quote(c['document_id'],safe='')}/chunks?id={quote(c['id'],safe='')}")
                raw, = data["chunks"]
                if identity(raw)!=identity(c) or digest(raw["content"])!=digest(c["content"]):
                    raise ValueError("原文回读与冻结候选不符")
                originals[identity(c)] = {**c, "content":raw["content"]}
            except Exception:
                event["failed"] = True
                raise
            finally:
                event["seconds"] = time.monotonic()-started
        compact,guard = prepare(exp.queries[qid]["question"],chosen,result["facts"],originals,pool["allowed"])
        result["guard_pre_delivery"] = guard
        result["choices"].update(fact_select=chosen,fact_deliver=compact)
    except (ValueError,TypeError,KeyError,IndexError,RuntimeError,OSError) as exc:
        result["fallback"] = {"type":type(exc).__name__,"reason":str(exc)[:400]}
    return result


def render_packets(exp, base, active):
    packets(exp,active,ARMS,lambda q:read_json(exp.root/q/"choices.json")["choices"],"rendered.json")
    for q in active:
        def finish():
            batch = read_json(exp.root/q/"rendered.json")
            old = read_json(base/q/"packets.json")["feedback"]
            if packet_key(batch["fixed"])!=packet_key(old):
                raise ValueError("原版最终消息无法原样复现")
            decision = read_json(exp.root/q/"choices.json")
            facts = decision["facts"] if not decision["fallback"] else []
            for arm in ARMS:
                batch[arm]["fact_delivery"] = {"required":len(facts),"received":received_facts(facts,batch[arm]["visible"])}
            delivered = batch["fact_deliver"]
            if (not facts or not delivered.get("delivery_complete") or
                    len(delivered["fact_delivery"]["received"])!=len(facts)):
                reason = "无新事实筛选" if not facts else "统一预算内未能完整装入事实，回退本题原版"
                batch["fact_deliver"] = deepcopy(batch["fixed"])
                batch["fact_deliver"]["delivery_fallback"] = reason
            return batch
        cached(exp.root/q/"packets.json",finish)


def report(exp, active):
    labels = {r["id"]:r for r in read_jsonl("data/financebench/gold.jsonl")}
    rows = {}
    for q in sorted(exp.queries):
        target = {(e["doc_name"].casefold(),e["evidence_page_num"]+1) for e in labels[q]["evidence"]}
        r = rows[q] = {"标准题页":len(target),"候选命中题页":0}
        if q in active:
            d = read_json(exp.root/q/"choices.json")
            base = Path(read_json(exp.root/"pair.json")["base"]) if (exp.root/"pair.json").exists() else exp.root/"frozen"
            _, candidates, _, _, _, _ = frozen(exp,base,q)
            r["候选命中题页"] = len(page_set(candidates)[0] & target)
            r.update(筛选用量=[e["usage"] for e in d["log"]],筛选秒=sum(e["seconds"] for e in d["log"]),
                     回读次数=len(d["reads"]),回读失败=sum(e["failed"] for e in d["reads"]),
                     回读秒=sum(e["seconds"] for e in d["reads"]),回退=d["fallback"],
                     事实数=len(d["facts"]),仍缺反馈数=len(d["unresolved"]),候选数=d["candidate_count"])
        for arm,key in read_json(exp.root/q/"assignment.json").items():
            p = read_json(exp.root/q/"answers"/(key+".json"))
            score = read_json(exp.root/q/"judges"/(key+".json"))
            a = p.get("native_trace",{}).get("evidence_layers",{}).get("answers",[{}])[0]
            received = a.get("requests",[{}])[0].get("complete_chunks",[])
            target = {(e["doc_name"].casefold(),e["evidence_page_num"]+1) for e in labels[q]["evidence"]}
            guard = formula_check(exp.queries[q]["question"],received,
                                  {c["document_id"] for c in received},lambda c:c)
            # 送达记录按各组渲染结果读取，避免同消息缓存共享掩盖方案的回退标签。
            packet = read_json(exp.root/q/"packets.json")[arm] if q in active else {}
            r[arm] = {"判对":score["answer_correct"],"引用支持":score["citation_support"],
                      "达标":score["answer_correct"] is True and score["citation_support"] is True,
                      "选中块":len(a.get("selected",[])),"实收块":len(received),
                      "选中题页":len(page_set(a.get("selected",[]))[0] & target),
                      "实收题页":len(page_set(received)[0] & target),
                      "事实应收":packet.get("fact_delivery",{}).get("required",0),
                      "事实实收":len(packet.get("fact_delivery",{}).get("received",[])),
                      "装入回退":packet.get("delivery_fallback"),
                      "操作数应有":guard.get("required"),"操作数核验":guard.get("verified"),
                      "回答用量":p.get("usage"),"回答秒":p.get("elapsed_seconds"),
                      "裁判用量":score.get("usage"),"裁判秒":score.get("elapsed_seconds")}
    changes = {}
    for arm in ARMS[1:]:
        changes[arm] = {"救回":[q for q,r in rows.items() if r[arm]["达标"] and not r["fixed"]["达标"]],
                        "损失":[q for q,r in rows.items() if r["fixed"]["达标"] and not r[arm]["达标"]]}
    summary = {"题数":len(rows),"联合达标":{k:sum(v[k]["达标"] for v in rows.values()) for k in ARMS},
               "变化":changes,"逐题":rows,"新增搜索":0,
               "首次实际回答":len([p for p in exp.root.glob("*/answers/*.json") if p.name!='scope.json' and '.failure-' not in p.name]),
               "首次实际裁判":len([p for p in exp.root.glob("*/judges/*.json") if '.failure-' not in p.name]),
               "失败次数":len(list(exp.root.rglob("*.failure-*.json")))}
    save(exp.root/"summary.json",summary)
    return summary


def confirm(exp, summary):
    ids = sorted({q for change in summary["变化"].values() for keys in change.values() for q in keys})
    replica = copy(exp)
    replica.root = exp.root/"confirmation"
    def one(q):
        batch = read_json(exp.root/q/"packets.json")
        for packet in batch.values():
            answer(replica,q,packet)
        save(replica.root/q/"assignment.json",{k:packet_key(p) for k,p in batch.items()})
    exp.parallel(one,ids)
    grade(replica,ids)
    repeated = {}
    for q in ids:
        repeated[q] = {}
        for arm,key in read_json(replica.root/q/"assignment.json").items():
            s = read_json(replica.root/q/"judges"/(key+".json"))
            repeated[q][arm] = {"判对":s["answer_correct"],"引用支持":s["citation_support"],
                               "达标":s["answer_correct"] is True and s["citation_support"] is True}
    summary["独立复验"] = repeated
    summary["失败次数"] = len(list(exp.root.rglob("*.failure-*.json")))
    save(exp.root/"summary.json",summary)


def run(source,base,out,stage):
    exp = Experiment(source,out,4)
    for key in ("source","freeze","questions"):
        if read_json(base/"manifest.json")[key]!=read_json(out/"manifest.json")[key]:
            raise ValueError("冻结源身份不符")
    protocol = {"base":str(base.resolve()),"version":1,"arms":list(ARMS),"max_chunks":4,"budget":8000}
    if cached(out/"pair.json",lambda:protocol)!=protocol:
        raise ValueError("实验不可跨协议恢复")
    active = []
    for q in sorted(exp.queries):
        if (base/q/"scope.json").exists():
            query = exp.queries[q]
            plan = scope(query["question"],exp.docs,query.get("as_of"))
            if len(plan["companies"])==1 and plan["document_ids"]:
                raise ValueError("历史澄清范围已经变化")
            save(out/q/"answers/scope.json",{**query,"answer":"请明确要查询的公司。","status":"ok","reference":{"chunks":[]}})
            save(out/q/"assignment.json",{k:"scope" for k in ARMS})
        else:
            active.append(q)
    if stage in ("all","decide"):
        exp.parallel(lambda q:cached(out/q/"choices.json",lambda:decide(exp,base,q)),active)
    if stage in ("all","render"):
        render_packets(exp,base,active)
    if stage in ("all","answer"):
        def generate(q):
            batch = read_json(out/q/"packets.json")
            for packet in batch.values():
                answer(exp,q,packet)
            save(out/q/"assignment.json",{k:packet_key(p) for k,p in batch.items()})
        exp.parallel(generate,active)
    if stage in ("all","grade"):
        grade(exp)
    if stage in ("all","report"):
        summary = report(exp,active)
        confirm(exp,summary)


if __name__=="__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("source","base","output"):
        p.add_argument(key,type=Path)
    p.add_argument("--stage",choices=("all","decide","render","answer","grade","report"),default="all")
    args = p.parse_args()
    run(args.source,args.base,args.output,args.stage)
