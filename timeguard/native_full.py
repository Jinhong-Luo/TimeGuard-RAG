"""严格复用所选原生开发集39题，只新增111题及裁判，保持原始来源不变。"""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import time

from .common import digest,read_json,read_jsonl,write_json,write_jsonl
from .native_baseline import container_worker,private_runtime,select_queries


def reuse_reviews(lines,original_signature,target_signature,gold,generated):
    """未知判断仍原样复用；每条判断必须绑定同一个实际裁判输入及原始JSON。"""
    from .judge import build_request,parse_judgement,citation_context,apply_citation_policy,FIELDS
    rows=[];seen=set()
    for line in lines:
        row=json.loads(line);ident=row["id"]
        if ident in seen:raise ValueError("复用裁判存在重复题号")
        seen.add(ident)
        if (row.get("judge_status")!="ok" or row.get("judge_type")!="model"
                or row.get("judge_run_signature")!=original_signature
                or type(row.get("elapsed_seconds")) not in (int,float) or not math.isfinite(row["elapsed_seconds"]) or row["elapsed_seconds"]<=0
                or row.get("judge_input_sha256")!=digest(build_request(gold[ident],generated[ident]))):
            raise ValueError("裁判来源、状态或实际输入不匹配")
        parsed,_=parse_judgement({"model":row.get("judge_model"),"bridge_metadata":row.get("bridge_metadata"),
                                  "choices":[{"message":{"content":row.get("raw_judgement")}}]})
        context=citation_context(generated[ident]);expected=apply_citation_policy(parsed,context)
        if (any(row.get(k)!=expected[k] for k in (*FIELDS,"has_factual_claims"))
                or row.get("citation_mapping")!={k:v for k,v in context.items() if k!="citations"}):
            raise ValueError("裁判字段与原始判断或引用映射不一致")
        value=deepcopy(row);value["judge_run_signature"]=target_signature
        value["reuse_origin"]={"row_sha256":digest(line),"judge_run_signature":original_signature}
        rows.append(value)
    return rows


def audit(private,start,effort):
    path=private/"bridge-audit.jsonl";end=path.stat().st_size;raw=path.read_bytes()[start:end]
    calls=[json.loads(line) for line in raw.splitlines() if line.strip()]
    successful=[r for r in calls if r.get("success")]
    return {"audit_byte_start":start,"audit_byte_end":end,"audit_sha256":digest(raw),"actual_calls":len(calls),
        "failed_calls":sum(r.get("success") is not True for r in calls),
        "tokens":sum((r.get("usage") or {}).get("totalTokens",0) for r in calls),"cost_usd":None,
        "model_identity_verified":all(r.get("actual_model")=="gpt-6.1-sol" and r.get("reasoning_effort")==effort for r in successful) if successful else None}


def generate(root,source,out):
    from .matrix import frozen_index
    from . import native_baseline
    selection=read_json(root/"selection-core.json");chosen=selection["chosen"]
    if source.name!=chosen["run"] or not source.resolve().is_relative_to(root.resolve()):raise ValueError("来源不是本轮所选配置")
    for name,key in (("manifest.json","manifest_sha256"),("results.jsonl","results_sha256")):
        if digest((source/name).read_bytes())!=chosen[key]:raise ValueError("选择来源字节已改变")
    manifest=read_json(source/"manifest.json");identity=manifest["identity"]
    queries=select_queries(read_jsonl("data/financebench/queries.jsonl"),"all")
    dev=[q for q in queries if q["split"]=="dev"]
    lines=[line for line in (source/"results.jsonl").read_bytes().splitlines() if line.strip()]
    original=[json.loads(line) for line in lines];by_id={r["id"]:r for r in original}
    if (len(queries)!=150 or len(dev)!=39 or len(original)!=39 or len(by_id)!=39
            or identity["query_ids"]!=[q["id"] for q in dev] or identity["split"]!="dev"
            or identity["queries_sha256"]!=digest(Path("data/financebench/queries.jsonl").read_bytes())
            or identity["source_sha256"]!=digest(Path(native_baseline.__file__).read_bytes())
            or any(r["status"]!="ok" or r["question"]!=q["question"] for q in dev for r in [by_id[q["id"]]])):
        raise ValueError("复用要求相同题源、完整开发集及未改变的原生生成驱动")
    freeze=frozen_index(Path("data/ingestion.json"),Path("runs/r0-gpu-retrieval-all"),Path("data/financebench/queries.jsonl"))
    if freeze!=identity["freeze"]:raise ValueError("索引来源与开发集不同")
    spec=read_json(source/"spec.json")
    if digest(spec)!=identity["spec_sha256"]:raise ValueError("开发集实际生成参数与清单不一致")
    spec["queries"]=queries
    proof={"source_run":source.name,"manifest_sha256":chosen["manifest_sha256"],"results_sha256":chosen["results_sha256"],
           "reused_count":39,"new_count":111,"selection_sha256":digest((root/"selection-core.json").read_bytes()),
           # 生成阶段只读取选优元数据，不读取裁判文件或金标。
           "selection_source_hashes":{"model_reviews":chosen["reviews_sha256"],
             "model_reviews_manifest":chosen["reviews_manifest_sha256"]}}
    full_identity={**identity,"query_ids":[q["id"] for q in queries],"split":"all","spec_sha256":digest(spec),"reuse":proof}
    out.mkdir(exist_ok=False);fresh=out/"new111";fresh.mkdir()
    write_json(out/"manifest.json",{"identity":full_identity,"created_at_utc":datetime.now(timezone.utc).isoformat(),
                                  "merger_source_sha256":digest(Path(__file__).read_bytes())})
    write_json(out/"spec.json",spec)
    fresh_spec={**spec,"queries":[q for q in queries if q["id"] not in by_id]}
    write_json(fresh/"spec.json",fresh_spec)
    private,_=private_runtime();start=(private/"bridge-audit.jsonl").stat().st_size;started=time.monotonic()
    code=container_worker(fresh_spec,fresh)
    new=read_jsonl(fresh/"results.jsonl")
    for row,line in zip(original,lines):row["reuse_origin"]={"run":source.name,"row_sha256":digest(line),"results_sha256":chosen["results_sha256"]}
    combined={r["id"]:r for r in [*original,*new]}
    if len(combined)!=len(original)+len(new):raise ValueError("新增题与复用题重复")
    write_jsonl(out/"results.jsonl",[combined[q["id"]] for q in queries if q["id"] in combined])
    source_execution=read_json(source/"execution.json")
    write_json(out/"execution.json",{**audit(private,start,"low"),"returncode":code,"planned":150,"completed":len(combined),
        "wall_seconds":time.monotonic()-started,"new_count":len(new),"reused_count":39,
        "reused_source_wall_seconds":source_execution["wall_seconds"]+source_execution.get("reused_source_wall_seconds",0),
        "reused_source_calls":source_execution["actual_calls"]+source_execution.get("reused_source_calls",0),
        "reused_source_tokens":source_execution["tokens"]+source_execution.get("reused_source_tokens",0),
        "说明":"本次审计窗口只含新增111题；原39题成本在来源运行中，不重复计入全局调用合计。"})
    return code


def judge(root,source,out):
    from . import judge as module
    manifest=read_json(out/"manifest.json");ids=manifest["identity"]["query_ids"]
    generated=module.unique_rows(read_jsonl(out/"results.jsonl"),"完整生成")
    gold_path=Path("data/financebench/gold.jsonl");gold=module.unique_rows(read_jsonl(gold_path),"金标")
    origin=read_json(source/"model_reviews.manifest.json")
    if digest(origin["identity"])!=origin["signature"]:raise ValueError("原裁判签名不匹配")
    proof=manifest["identity"]["reuse"]
    if (digest((source/"model_reviews.jsonl").read_bytes())!=proof["selection_source_hashes"]["model_reviews"]
            or digest((source/"model_reviews.manifest.json").read_bytes())!=proof["selection_source_hashes"]["model_reviews_manifest"]):
        raise ValueError("原裁判文件已改变")
    identity={"version":module.JUDGE_VERSION,"judge_type":"model","model":module.MODEL,"effort":module.EFFORT,
        "transport":module.TRANSPORT,"expected_ids":ids,"concurrency":2,"request_timeout_seconds":360.0,
        "generation_manifest_sha256":digest((out/"manifest.json").read_bytes()),"generation_results_sha256":digest((out/"results.jsonl").read_bytes()),
        "gold_sha256":digest(gold_path.read_bytes()),"prompt_sha256":digest(module.INSTRUCTIONS),
        "code_sha256":digest(Path(module.__file__).read_bytes()),"schema":module.SCHEMA}
    ignored={"expected_ids","generation_manifest_sha256","generation_results_sha256"}
    if ({k:v for k,v in identity.items() if k not in ignored}!={k:v for k,v in origin["identity"].items() if k not in ignored}
            or origin["identity"]["generation_manifest_sha256"]!=proof["manifest_sha256"]
            or origin["identity"]["generation_results_sha256"]!=proof["results_sha256"]):
        raise ValueError("金标、裁判模型、提示、超时、并发或代码与来源不同")
    signature=digest(identity)
    lines=[line for line in (source/"model_reviews.jsonl").read_bytes().splitlines() if line.strip()]
    reused=reuse_reviews(lines,origin["signature"],signature,gold,generated)
    if len(reused)!=39 or {r["id"] for r in reused}!=set(origin["identity"]["expected_ids"]):raise ValueError("裁判复用未覆盖完整39题")
    output=out/"model_reviews.jsonl"
    if output.exists():raise ValueError("目标裁判已存在，不自动续发或覆盖")
    write_json(out/"model_reviews.manifest.json",{"identity":identity,"signature":signature,
        "executor_sha256":digest(Path(__file__).read_bytes()),"reuse_source_signature":origin["signature"]})
    private,rt=private_runtime();start=(private/"bridge-audit.jsonl").stat().st_size;started=time.monotonic()
    ok=module.judge_queries(ids,gold,generated,signature,output,rt["base_url"],rt["token"],360.0,concurrency=2,reuse_rows=reused)
    source_audit=read_json(source/"judge-audit.json")
    write_json(out/"judge-audit.json",{**audit(private,start,"medium"),"returncode":0 if ok else 1,"wall_seconds":time.monotonic()-started,
        "reused_count":39,"reused_source_calls":source_audit["actual_calls"],"reused_source_tokens":source_audit["tokens"],
        "reused_source_wall_seconds":source_audit["wall_seconds"]})
    return 0 if ok else 1


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--root",type=Path,required=True)
    p.add_argument("--source",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--stage",choices=("generation","judge"),required=True);args=p.parse_args()
    raise SystemExit(generate(args.root,args.source,args.output) if args.stage=="generation" else judge(args.root,args.source,args.output))


if __name__=="__main__":main()
