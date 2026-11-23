"""冻结上一轮事实筛选原块，仅比较原生八千和一万二预算，禁止重选和回退。"""
import argparse
from copy import copy
from pathlib import Path

from .common import digest, read_json, read_jsonl
from .fact_selection import received_facts
from .full_feedback import Experiment, answer, cached, grade, packet_key, packets, save
from .layer_report import page_set
from .scope import scope

ARMS = ("narrow", "wide")


def frozen_choice(exp,base,q):
    d=read_json(base/q/"choices.json")
    chosen=d["choices"]["fact_select"]
    query=exp.queries[q]
    plan=scope(query["question"],exp.docs,query.get("as_of"))
    snap,=exp.rows[q]["native_trace"]["evidence_layers"]["answers"]
    allowed=set(plan["document_ids"] or []) & set(snap["snapshot"]["input"]["tools"]["doc_scope"] or [])
    if len(plan["companies"])!=1 or len(chosen)!=4 or any(c["document_id"] not in allowed for c in chosen):
        raise ValueError("冻结四块超出允许范围")
    if any(c.get("fact_excerpt") for c in chosen):
        raise ValueError("预算对照禁止使用缩短材料")
    return {arm:chosen for arm in ARMS}


def totals(root,pattern):
    # 公司澄清占位由程序生成，没有发出模型请求。
    paths = [
        path for path in root.glob(pattern)
        if ".failure-" not in path.name and path.name != "scope.json"
    ]
    data=[read_json(p) for p in paths]
    return {"请求":len(data),"用量缺失":sum(not d.get("usage") for d in data),
            "累计秒":sum(d.get("elapsed_seconds",0) for d in data),
            **{k:sum((d.get("usage") or {}).get(k,0) for d in data) for k in ("prompt_tokens","completion_tokens","total_tokens")}}


def summarize(exp,base,active):
    gold={r["id"]:r for r in read_jsonl("data/financebench/gold.jsonl")}
    rows={}
    for q in sorted(exp.queries):
        targets={(e["doc_name"].casefold(),e["evidence_page_num"]+1) for e in gold[q]["evidence"]}
        decision=read_json(base/q/"choices.json") if q in active else {}
        facts=decision.get("facts",[]) if not decision.get("fallback") else []
        row=rows[q]={"标准题页":len(targets),"冻结事实数":len(facts)}
        for arm in ARMS:
            root, label = (base, "fact_select") if arm == "narrow" else ((exp.root, "wide") if q in active else (base, "fact_select"))
            key = read_json(root/q/"assignment.json")[label]
            a=read_json(root/q/"answers"/(key+".json"))
            j=read_json(root/q/"judges"/(key+".json"))
            p=read_json(exp.root/q/"packets.json")[arm] if q in active else {}
            received=p.get("visible",[])
            row[arm]={"判对":j["answer_correct"],"引用支持":j["citation_support"],
                "达标":j["answer_correct"] is True and j["citation_support"] is True,
                "选中块":len(p.get("selected",[])),"实收块":len(received),
                "四块送全":len(received)==4,"实收标准题页":len(page_set(received)[0]&targets),
                "事实实收":len(received_facts(facts,received)),
                "输入指纹":digest(p.get("body")),"回答用量":a.get("usage"),"回答秒":a.get("elapsed_seconds"),
                "裁判用量":j.get("usage"),"裁判秒":j.get("elapsed_seconds")}
    for row in rows.values():
        row["输入有变化"] = row["narrow"]["输入指纹"] != row["wide"]["输入指纹"]
    gain=[q for q,r in rows.items() if r["wide"]["达标"] and not r["narrow"]["达标"]]
    loss=[q for q,r in rows.items() if r["narrow"]["达标"] and not r["wide"]["达标"]]
    summary={"类型":"冻结原块及顺序，只改变原生预算；复用八千历史结果，仅一万二组重新回答和评分", "题数":len(rows),
         "判分来源":"模型裁判，未独立人工逐题裁定","救回":gain,"损失":loss,"逐题":rows,
         "新增搜索":0,"新增筛选":0,"新增原文回读":0,
         "联合达标":{arm:sum(r[arm]["达标"] for r in rows.values()) for arm in ARMS}}
    save(exp.root/"summary.json",summary)
    return summary


def run(source,base,out,stage):
    exp=Experiment(source,out,4)
    for k in ("source","freeze","questions"):
        if read_json(base/"manifest.json")[k]!=read_json(out/"manifest.json")[k]:
            raise ValueError("源数据和索引身份不一致")
    choices={q:digest((base/q/"choices.json").read_bytes()) for q in exp.queries if (base/q/"choices.json").exists()}
    protocol={"version":1,"base":str(base.resolve()),"choices":choices,"budgets":[8000,12000]}
    if cached(out/"pair.json",lambda:protocol)!=protocol:
        raise ValueError("续跑期间冻结选择发生变化")
    outcome = {"comparison": "historical_fact_select", "fresh": "wide_only"}
    if cached(out/"evaluation-mode.json",lambda:outcome)!=outcome:
        raise ValueError("同一运行目录不能混用不同的评测方案")
    active=[]
    for q in sorted(exp.queries):
        if q not in choices:
            query=exp.queries[q]
            plan=scope(query["question"],exp.docs,query.get("as_of"))
            if len(plan["companies"])==1 and plan["document_ids"]:
                raise ValueError("缺少有效题目的冻结选择")
            # 同一条澄清结果适用两档预算，沿用历史答案与评分。
            if read_json(base/q/"assignment.json")["fact_select"]!="scope":
                raise ValueError("历史澄清记录与当前范围不符")
            save(out/q/"assignment.json",{"wide":"scope"})
        else:
            active.append(q)
    if stage in ("all","render"):
        packets(exp,active,ARMS,lambda q:frozen_choice(exp,base,q),"packets.json")
        for q in active:
            p=read_json(out/q/"packets.json")
            if packet_key(p["narrow"])!=packet_key(read_json(base/q/"packets.json")["fact_select"]):
                raise ValueError("八千组未复现上一轮原始消息")
            if p["narrow"]["selected"]!=p["wide"]["selected"]:
                raise ValueError("预算两组所选块或顺序变化")
            originals=frozen_choice(exp,base,q)["narrow"]
            for arm in ARMS:
                for c in p[arm]["visible"]:
                    raw=next(v for v in originals if (v["id"],v["document_id"])==(c["id"],c["document_id"]))
                    if c["content"]!=raw["content"]:
                        raise ValueError("原文被缩短或改写")
    if stage in ("all","render","answer"):
        for q in active:
            assignment = read_json(base/q/"assignment.json")["fact_select"]
            pair = read_json(out/q/"packets.json")
            if packet_key(pair["narrow"]) != assignment:
                raise ValueError("历史答案未绑定到本轮核对过的八千请求")
    if stage in ("all","answer"):
        def one(q):
            p=read_json(out/q/"packets.json")
            answer(exp,q,p["wide"])
            save(out/q/"assignment.json",{"wide":packet_key(p["wide"])})
        exp.parallel(one,active)
    if stage in ("all","grade"):
        grade(exp,active)
    if stage in ("all","report"):
        s=summarize(exp,base,active)
        changed=sorted(q for q in set(s["救回"]+s["损失"]) if s["逐题"][q]["输入有变化"])
        replica=copy(exp); replica.root=out/"confirmation"
        def repeat(q):
            p=read_json(out/q/"packets.json")
            answer(replica,q,p["wide"])
            save(replica.root/q/"assignment.json",{"wide":packet_key(p["wide"])})
        exp.parallel(repeat,changed)
        grade(replica,changed)
        s["独立复验"]={}
        for q in changed:
            s["独立复验"][q]={}
            for arm,key in read_json(replica.root/q/"assignment.json").items():
                j=read_json(replica.root/q/"judges"/(key+".json"))
                s["独立复验"][q][arm]={"判对":j["answer_correct"],"引用支持":j["citation_support"],
                    "达标":j["answer_correct"] is True and j["citation_support"] is True}
        s["实际调用"]={name:totals(out,pattern) for name,pattern in
            (("首轮回答","*/answers/*.json"),("首轮裁判","*/judges/*.json"),
             ("复验回答","confirmation/*/answers/*.json"),("复验裁判","confirmation/*/judges/*.json"))}
        s["失败记录"] = {
            stage:len(list(out.glob(pattern)))
            for stage,pattern in (
                ("首轮回答","*/answers/*.failure-*.json"),("首轮裁判","*/judges/*.failure-*.json"),
                ("复验回答","confirmation/*/answers/*.failure-*.json"),
                ("复验裁判","confirmation/*/judges/*.failure-*.json"),
            )
        }
        save(out/"summary.json",s)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("source","base","output"):
        p.add_argument(k,type=Path)
    p.add_argument("--stage",choices=("all","render","answer","grade","report"),default="all")
    a=p.parse_args(); run(a.source,a.base,a.output,a.stage)
