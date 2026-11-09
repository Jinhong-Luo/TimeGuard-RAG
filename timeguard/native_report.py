"""离线重算原生基线的安全公开摘要；不发送模型请求，不复制答案或原始证据。"""
from collections import Counter
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path

from .common import digest,read_json,read_jsonl,write_json
from .native_matrix import summarize_run

CORE=("plain","keyword","rerank","metadata_auto","metadata_all","agent_low")
LABELS={"plain":"普通原生问答","keyword":"原生关键词增强","rerank":"原生重排（分词截断兼容）",
        "metadata_auto":"公司字段半自动过滤","metadata_all":"原生全自动元数据过滤",
        "agent_low":"原生low模式","agent_medium":"原生medium模式","agent_high":"原生high模式",
        "agent_ultra":"原生ultra模式","combined":"原生过滤与Agentic组合配置"}


def judgement_counts(ids,rows):
    if len(set(ids))!=len(ids) or len({r["id"] for r in rows})!=len(rows) or {r["id"] for r in rows}-set(ids):
        raise ValueError("计划或裁判题号重复、超出范围")
    mapping={r["id"]:r for r in rows};counts=Counter()
    for ident in ids:
        value=mapping.get(ident,{}).get("answer_correct")
        counts["correct" if value is True else "incorrect" if value is False else "unknown"]+=1
    n=len(ids);known=counts["correct"]+counts["incorrect"]
    return {"planned":n,**{k:counts[k] for k in ("correct","incorrect","unknown")},"missing_reviews":len(set(ids)-mapping.keys()),"judged_coverage":known/n if n else 0,
        "known_subset_accuracy":counts["correct"]/known if known else None,
        "full_accuracy":counts["correct"]/n if n and known==n else None,
        "arithmetic_lower_bound":counts["correct"]/n if n else None,
        "arithmetic_upper_bound":(counts["correct"]+counts["unknown"])/n if n else None}


def window_records(raw,window):
    start,end=window["audit_byte_start"],window["audit_byte_end"]
    if not 0<=start<=end<=len(raw):raise ValueError("模型审计窗口越界")
    selected=raw[start:end]
    if digest(selected)!=window["audit_sha256"]:raise ValueError("模型审计窗口字节摘要不匹配")
    return [json.loads(line) for line in selected.splitlines() if line.strip()]


def audit_summary(root,audit):
    if not audit.exists():return {"available":False}
    raw=audit.read_bytes();all_rows=[json.loads(line) for line in raw.splitlines() if line.strip()]
    ids=[r["request_id"] for r in all_rows]
    if len(set(ids))!=len(ids):raise ValueError("桥审计包含重复请求编号")
    owned=set();windows=[]
    paths=[*root.glob("*/execution.json"),*root.glob("*/judge-audit.json"),root/"http-session-stream/summary.json"]
    for path in sorted(paths):
        if not path.exists():continue
        value=read_json(path)
        if "audit_byte_start" not in value:continue
        records=window_records(raw,value);group_ids={r["request_id"] for r in records}
        if owned&group_ids:raise ValueError("模型调用被两个实验窗口重复计数")
        owned.update(group_ids)
        windows.append({"file":str(path.relative_to(root)),"calls":len(records),
            "failed":sum(r.get("success") is not True for r in records),"audit_sha256":value["audit_sha256"]})
    fingerprints={digest({k:r.get(k) for k in ("bridge_source_sha256","instruction_source_sha256")})
                  for r in all_rows if r.get("success")}
    return {"available":True,"actual_calls":len(all_rows),"successful_calls":sum(r.get("success") is True for r in all_rows),
        "failed_calls":sum(r.get("success") is not True for r in all_rows),
        "known_tokens":sum((r.get("usage") or {}).get("totalTokens",0) for r in all_rows),
        "usage_unknown_calls":sum(not r.get("usage") for r in all_rows),"accounted_window_calls":len(owned),
        "outside_window_calls":len(set(ids)-owned),"window_count":len(windows),"windows":windows,
        "bridge_fingerprint_count":len(fingerprints),"source_fingerprint_stable":len(fingerprints)==1,
        "actual_models":dict(Counter(r.get("actual_model") for r in all_rows if r.get("success"))),
        "cost_usd":None,"audit_sha256":digest(raw),
        "说明":"已返回token合计；缺少用量的失败调用不当作零成本。窗口外包含启动校验和网络中断未完整落盘的阶段；不伪造逐题请求对应关系。"}


def run_details(path):
    value=summarize_run(path)
    rows=read_jsonl(path/"results.jsonl") if (path/"results.jsonl").exists() else []
    manifest=read_json(path/"manifest.json")
    if manifest["identity"].get("recovery"):
        value["recovery"]=manifest["identity"]["recovery"]
    ids=manifest["identity"]["query_ids"]
    reviews=read_jsonl(path/"model_reviews.jsonl") if (path/"model_reviews.jsonl").exists() else []
    value.update(judgement_counts(ids,reviews))
    value["label"]=LABELS[manifest["identity"]["arm"]["name"]]
    for filename,key in (("execution.json","generation"),("judge-audit.json","judge")):
        if (path/filename).exists():
            record=read_json(path/filename)
            value[key]={k:record.get(k) for k in ("returncode","actual_calls","failed_calls","tokens","wall_seconds","model_identity_verified",
                "new_count","reused_count","reused_source_calls","reused_source_tokens","reused_source_wall_seconds")}
            value[key]["attributable_calls_with_reuse"]=record.get("actual_calls",0)+record.get("reused_source_calls",0)
    value["complete"]=len(rows)==len(ids) and len(reviews)==len(ids) and all(r["status"]=="ok" for r in rows) and all(r.get("judge_status")=="ok" for r in reviews)
    value["citation_field_counts"]={field:dict(Counter("true" if r.get(field) is True else "false" if r.get(field) is False else "unknown" for r in reviews)) for field in ("numeric_and_scope_correct","citation_support")}
    return value


def validate_selection(root,selection,expected_arms=CORE):
    candidates=selection["candidates"]
    if len(candidates)!=len(expected_arms) or {c["arm"] for c in candidates}!=set(expected_arms):raise ValueError("核心选择缺少当前计划中的完整组")
    for candidate in candidates:
        path=root/candidate["run"]
        if not path.resolve().is_relative_to(root.resolve()):raise ValueError("选择来源超出实验目录")
        for name,key in (("manifest.json","manifest_sha256"),("results.jsonl","results_sha256"),("model_reviews.jsonl","reviews_sha256")):
            if digest((path/name).read_bytes())!=candidate[key]:raise ValueError("选择报告与原始来源字节不一致")
        rows=read_jsonl(path/"results.jsonl");reviews=read_jsonl(path/"model_reviews.jsonl")
        if len(rows)!=39 or len(reviews)!=39 or any(r["status"]!="ok" for r in rows) or any(r.get("judge_status")!="ok" for r in reviews):raise ValueError("不完整核心组不能选优")
        if (sum(r.get("answer_correct") is True for r in reviews)!=candidate["correct"]
                or sum(r.get("answer_correct") is None for r in reviews)!=candidate["unknown"]
                or sum(r["elapsed_seconds"] for r in rows)/39!=candidate["mean_seconds"]):raise ValueError("选择候选计数或耗时不匹配")
    expected=sorted(candidates,key=lambda c:(-c["correct"],c["unknown"],c["mean_seconds"]))[0]
    if selection["chosen"]!=expected:raise ValueError("所选配置与预先声明的规则不符")


def build(root,audit):
    scope=read_json(root/"scope-current.json") if (root/"scope-current.json").exists() else {"core_arms":list(CORE),"full150_requested":True}
    expected_arms=scope["core_arms"]
    if not expected_arms or len(set(expected_arms))!=len(expected_arms) or set(expected_arms)-set(CORE):
        raise ValueError("核心范围为空、重复或包含未允许的完整评测组")
    core=[]
    for arm in expected_arms:
        name=scope.get("run_paths",{}).get(arm,"plain-dev" if arm=="plain" else arm+"-core-dev")
        path=root/name
        if not path.resolve().is_relative_to(root.resolve()):raise ValueError("核心来源超出实验目录")
        if (path/"manifest.json").exists():core.append(run_details(path))
    interrupted=[]
    for row in core:
        if row.get("recovery"):
            path=root/row["recovery"]["source_run"]
            if not path.resolve().is_relative_to(root.resolve()):raise ValueError("中断来源超出实验目录")
            if digest((path/"results.jsonl").read_bytes())!=row["recovery"]["results_sha256"]:
                raise ValueError("恢复来源的原始结果已改变")
            interrupted.append(run_details(path))
    smokes=[]
    for path in sorted(root.glob("*-validation-smoke")):
        if not (path/"execution.json").exists():continue
        value=run_details(path);value.pop("citation_field_counts",None)
        smokes.append(value)
    selection=read_json(root/"selection-core.json") if (root/"selection-core.json").exists() else None
    if selection:validate_selection(root,selection,expected_arms)
    full=None;by_split={}
    if selection:
        path=root/(selection["chosen"]["arm"]+"-core-all")
        if (path/"manifest.json").exists():
            full=run_details(path);reviews=read_jsonl(path/"model_reviews.jsonl") if (path/"model_reviews.jsonl").exists() else []
            queries=read_json(path/"spec.json")["queries"]
            for split in ("tune","dev","test"):
                ids=[q["id"] for q in queries if q["split"]==split]
                by_split[split]=judgement_counts(ids,[r for r in reviews if r["id"] in ids])
    restore=read_json(root/"metadata-restore.json") if (root/"metadata-restore.json").exists() else None
    finish=read_json(root/"supervisor-finish.json") if (root/"supervisor-finish.json").exists() else None
    audits=audit_summary(root,audit)
    manifests=[read_json(root/r["run"]/"manifest.json")["identity"] for r in core]
    comparable={key:len({digest(m.get(key)) for m in manifests})==1 for key in ("source_sha256","queries_sha256","upstream","freeze")}
    common_ready=(len(core)==len(expected_arms) and all(r["complete"] for r in core)
           and bool(restore and restore.get("verified") and restore.get("documents")==84)
           and bool(finish and not finish.get("error")) and audits.get("available") and audits.get("source_fingerprint_stable")
           and all(comparable.values()))
    pending=scope.get("status")=="awaiting_scope_decision" or scope.get("full150_requested",True) is None
    ready=common_ready and not pending and (bool(full and full["complete"]) if scope.get("full150_requested",True) else True)
    status="completed" if ready else "awaiting_scope_decision" if common_ready and pending else "incomplete"
    return {"version":"timeguard-native-core-baseline-v1","status":status,
        "generated_at_beijing":datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "upstream":"RAGFlow v0.27.3 / 8090ffd8ba6098813d7f7ec2c4b8774f36c8423a",
        "model_transport":"codex-app-server-structured-output-v1","generation_effort":"low","judge_effort":"medium",
        "scope":scope,
        "core_runs":core,"interrupted_runs":interrupted,"capability_smokes":smokes,"selection":selection,"full150":full,"full150_by_split":by_split,
        "audit":audits,"core_comparability":comparable,"metadata_restore":restore,"supervisor":finish,"human_reviewed":0,"cost_usd":None,
        "limits":["模型评分未人工复核，不是人工准确率。","全量150题含开发题，不是新的独立泛化评测。",
          "普通原生引用编号仍有一基/零基差异，引用支持统计不作为主要选优依据。",
          "单题耗时与旧R0双检索总耗时口径不同，进程启动另计。","SSE只验证原生协议，当前模型桥不提供真实逐token流式。",
          "原生重排使用明确记录的分词截断传输适配；未适配路径的HTTP413另行保留。"]}


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("run",type=Path)
    p.add_argument("--audit",type=Path);p.add_argument("--output",type=Path,required=True)
    args=p.parse_args();audit=args.audit or args.run.resolve().parents[1]/".cache/native-complete-20261109/bridge-audit.jsonl"
    report=build(args.run,audit);write_json(args.output,report)
    print(json.dumps({"status":report["status"],"core_groups":len(report["core_runs"]),"full150":bool(report["full150"]),"output":str(args.output)},ensure_ascii=False))


if __name__=="__main__":main()
