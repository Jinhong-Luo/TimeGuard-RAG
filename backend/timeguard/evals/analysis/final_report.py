"""核对全流程评测的金标隔离、逐阶段成本、页覆盖与跨公司结果。"""
import argparse
from collections import Counter,defaultdict
from pathlib import Path
from timeguard.common import Client,digest,read_json,read_jsonl,write_json
from timeguard.evals.analysis.layer_report import indexed,layers,page_set,reviews
from timeguard.evidence.formula_guard import guard_row,requirements


def generation_cost(path):
    """沿恢复清单回溯每批请求，计入失败尝试并核对复用来源。"""
    totals=Counter()
    batches=[]
    visited=set()
    while path not in visited:
        visited.add(path)
        execution=read_json(path/"execution.json")
        fields=("actual_calls","failed_calls","tokens","wall_seconds")
        totals.update({key:execution[key] for key in fields})
        batch={key:execution.get(key) for key in (*fields,"model_identity_verified","completed","planned")}
        if execution.get("wall_note"):
            batch["时间口径"]=execution["wall_note"]
        if execution.get("interruption"):
            batch["中断"]=execution["interruption"]
        if execution.get("returncode")!=0 and (path/"results.jsonl").exists():
            batch["已落盘题目"]={}
            for row in read_jsonl(path/"results.jsonl"):
                if not row.get("id"):
                    continue
                events=(row.get("native_trace") or {}).get("events",[])
                attempts=[e for e in events if e.get("kind")=="bridge_request"]
                batch["已落盘题目"][row["id"]]={
                    "状态":row["status"],"生成秒":row["elapsed_seconds"],"已记录请求":len(attempts),
                    "失败":sum(e.get("http_status")!=200 for e in attempts),
                    "已知token":sum((e.get("usage") or {}).get("total_tokens",0) for e in attempts)}
        batches.append(batch)
        proof=execution.get("recovery")
        if not proof:
            return {**totals,"model_identity_verified":all(b["model_identity_verified"] for b in batches),
                    "独立执行批次":list(reversed(batches))}
        source=path.parent/proof["source_run"]
        if source.parent!=path.parent or digest((source/"results.jsonl").read_bytes())!=proof["results_sha256"]:
            raise ValueError("恢复成本来源与清单不符")
        path=source
    raise ValueError("恢复清单循环引用")


def summarize(path,old=None,check_operands=False):
    manifest=read_json(path/"manifest.json")["identity"]
    ids=manifest["query_ids"]
    gold=indexed(read_jsonl("data/financebench/gold.jsonl"))
    rows,scores,_=reviews(path,ids,gold)
    execution=generation_cost(path)
    judge=read_json(path/"judge-audit.json")
    by_company=defaultdict(Counter)
    counts=Counter()
    phase_totals=defaultdict(Counter)
    details={}
    historical={} if old is None else indexed(read_jsonl(old/"model_reviews.jsonl"))
    client=Client(read_json("configs/r0-low.json")["base_url"]) if check_operands else None
    for key in ids:
        row=rows[key]
        rv=scores[key]
        trace=row["native_trace"]
        answers=trace.get("evidence_layers",{}).get("answers",[])
        allowed=trace.get("allowed", (trace.get("plan") or {}).get("document_ids"))
        target={(e["doc_name"].casefold(),e["evidence_page_num"]+1) for e in gold[key]["evidence"]}
        result={True:"判对",False:"判错",None:"未知"}[rv["answer_correct"]]
        counts[result]+=1
        counts["判对且引用支持"]+=rv["answer_correct"] is True and rv["citation_support"] is True
        counts["引用支持"]+=rv["citation_support"] is True
        counts["引用不支持"]+=rv["citation_support"] is False
        counts["引用支持未知"]+=rv["citation_support"] is None
        counts["题页分母"]+=len(target)
        by_company[gold[key]["company"]][result]+=1
        stages=layers(row)
        stages["pool"]=page_set(answers[0].get("pool_locations") if len(answers)==1 else None)
        ruled_out=row["status"]=="ok" and not answers and (trace.get("plan") or {}).get("document_ids") in (None,[])
        if ruled_out:
            counts["范围门控直接回答"]+=1
            stages={k:(set(), True) for k in stages}
        hits={k:len(p&target) if known else None for k,(p,known) in stages.items()}
        for stage,h in hits.items():
            if h is None:counts[stage+"未知题数"]+=1
            else:counts[stage+"命中组合"]+=h
        fields=("actual_calls","failed_calls","tokens","wall_seconds","model_identity_verified")
        phases=defaultdict(Counter)
        for e in trace["events"]:
            if e.get("kind")!="bridge_request":
                continue
            phase=e.get("phase") or e.get("node") or "外层未标记"
            item=phases[phase]
            item["请求"]+=1
            item["失败"]+=e.get("http_status")!=200
            item["累计秒"]+=e.get("seconds",0)
            if e.get("usage") is None:item["用量未知"]+=1
            else:
                for name in ("prompt_tokens","completion_tokens","total_tokens"):
                    item[name]+=e["usage"].get(name,0)
        for phase,cost in phases.items():
            phase_totals[phase].update(cost)
        visible=[c for a in answers for req in a["requests"] if req.get("status")==200 for c in req.get("complete_chunks") or []]
        if allowed is not None and any(c["document_id"] not in allowed for c in visible):
            raise ValueError("实收材料越界")
        calls=trace.get("evidence_layers",{}).get("retrievals",[])
        if allowed is not None and any(c["document_id"] not in allowed for call in calls for c in call.get("chunks") or []):
            raise ValueError("检索材料越界")
        counts["回放节点"]+=len(answers)
        counts["回放三项一致"]+=sum(all(a.get("replay",{}).get(k) is True for k in ("selected_equal","packet_equal","request_equal")) for a in answers)
        item={"判定":result,"引用支持":rv["citation_support"],"引用映射":rv.get("citation_mapping",{}).get("mapping_status"),
              "题页分母":len(target),"命中":hits,"允许报告数":len(allowed) if allowed is not None else None,
              "生成秒":row["elapsed_seconds"],"生成阶段":phases,"裁判秒":rv["elapsed_seconds"],"裁判用量":rv.get("usage"),
              "裁判请求":int(rv.get("model_call_attempted") is True),"裁判失败":int(rv["judge_status"]=="failed"),
              "公司":gold[key]["company"],"分流":trace.get("route_decision"),"轻量查数":trace.get("light_check")}
        review=trace.get("review_attempt")
        if review:
            item["审查"]={k:review.get(k) for k in ("reason","usage","fallback","error","seconds","request_count")}
        if client and (plan:=requirements(row["question"])):
            metric,needs=plan
            counts["有操作数定义题数"]+=1
            counts["操作数分母"]+=len(needs)
            requests=[r for a in answers for r in a["requests"] if r.get("status")==200]
            op={"metric":metric,"required":len(needs),"verified":None,"状态":"实收记录不足"}
            if len(answers)==len(requests)==1 and requests[0].get("complete_chunks") is not None:
                result=guard_row(row,client)
                op={k:result.get(k) for k in ("metric","required","verified","reads","seconds")}
                op["状态"]="已核验"
                counts["严格核验操作数"]+=op["verified"]
                counts["操作数齐全题数"]+=op["verified"]==op["required"]
            else:
                counts["操作数核验未知题数"]+=1
            item["离线操作数"]=op
        if key in historical:
            prior=historical[key]["answer_correct"]
            counts["历史错转对"]+=prior is False and rv["answer_correct"] is True
            counts["历史对转错"]+=prior is True and rv["answer_correct"] is False
        details[key]=item
    return {"评测类型":"真实端到端，模型裁判","分组":manifest["split"],"计划题数":len(ids),"汇总":counts,
            "公司":by_company,"生成":execution,"裁判":{k:judge.get(k) for k in fields},
            "生成阶段累计":phase_totals,"逐题":details}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("run",type=Path)
    p.add_argument("output",type=Path)
    p.add_argument("--old",type=Path)
    p.add_argument("--check-operands",action="store_true")
    a=p.parse_args()
    write_json(a.output,summarize(a.run,a.old,a.check_operands))
