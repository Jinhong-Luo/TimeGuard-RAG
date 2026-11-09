"""原生能力矩阵驱动；按独立实验保留失败，不自动覆盖或换配置重试。"""
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .common import digest, read_json, read_jsonl, write_json
from .native_baseline import ARMS, private_runtime


def summarize_run(path):
    manifest=read_json(path/"manifest.json")
    planned=len(manifest["identity"]["query_ids"])
    rows=read_jsonl(path/"results.jsonl") if (path/"results.jsonl").exists() else []
    reviews=read_jsonl(path/"model_reviews.jsonl") if (path/"model_reviews.jsonl").exists() else []
    expected=set(manifest["identity"]["query_ids"])
    for records in (rows,reviews):
        if len({r["id"] for r in records})!=len(records) or {r["id"] for r in records}-expected:
            raise ValueError("结果或裁判题号重复、超出冻结清单")
    if reviews:
        bound=read_json(path/"model_reviews.manifest.json")["identity"]
        if bound["generation_manifest_sha256"]!=digest((path/"manifest.json").read_bytes()) or bound["generation_results_sha256"]!=digest((path/"results.jsonl").read_bytes()):
            raise ValueError("裁判与生成来源不匹配")
    counts=Counter("true" if r.get("answer_correct") is True else "false" if r.get("answer_correct") is False else "unknown" for r in reviews)
    known=counts["true"]+counts["false"]
    return {"run":path.name,"planned":planned,"completed":len(rows),"missing":planned-len(rows),
        "statuses":dict(Counter(r["status"] for r in rows)),"model_judgements":dict(counts),
        "judge_missing":planned-len(reviews),"judged_coverage":known/planned,
        "known_subset_accuracy":counts["true"]/known if known else None,
        "full_accuracy":counts["true"]/planned if known==planned else None,
        "correct_fraction_lower_bound":counts["true"]/planned,
        "mean_seconds":sum(r["elapsed_seconds"] for r in rows)/len(rows) if rows else None,
        "native_retrieval_calls":sum(sum(e["kind"]=="retrieval" for e in r.get("native_trace",{}).get("events",[])) for r in rows),
        "native_agent_events":dict(Counter(e["name"] for r in rows for e in r.get("native_trace",{}).get("events",[]) if e["kind"]=="agent_tool")),
        "native_rerank_calls":sum(sum(e["kind"]=="retrieval" and e.get("reranker_used",False) for e in r.get("native_trace",{}).get("events",[])) for r in rows),
        "metadata_filter_calls":sum(sum(e["kind"]=="metadata_filter" for e in r.get("native_trace",{}).get("events",[])) for r in rows),
        "results_sha256":digest((path/"results.jsonl").read_bytes()) if rows else None,
        "reviews_sha256":digest((path/"model_reviews.jsonl").read_bytes()) if reviews else None}


def judge(path):
    private,rt=private_runtime();audit=private/"bridge-audit.jsonl"
    env=os.environ.copy();env["CODEX_BRIDGE_TOKEN"]=rt["token"]
    start=audit.stat().st_size if audit.exists() else 0
    started=time.monotonic()
    with (path/"judge.log").open("w") as log:
        code=subprocess.call([sys.executable,"-m","timeguard.judge",str(path),"--base-url",rt["base_url"],
                              "--concurrency","2","--execute"],env=env,stdout=log,stderr=log)
    end=audit.stat().st_size if audit.exists() else 0
    raw=audit.read_bytes()[start:end] if end else b""
    rows=[json.loads(l) for l in raw.splitlines() if l.strip()]
    write_json(path/"judge-audit.json",{"returncode":code,"wall_seconds":time.monotonic()-started,
        "audit_byte_start":start,"audit_byte_end":end,"audit_sha256":digest(raw),"actual_calls":len(rows),
        "failed_calls":sum(r.get("success") is not True for r in rows),
        "tokens":sum((r.get("usage") or {}).get("totalTokens",0) for r in rows),
        "model_identity_verified":all(r.get("actual_model")=="gpt-6.1-sol" and r.get("reasoning_effort")=="medium" for r in rows if r.get("success")),
        "cost_usd":None})
    return code


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--stage",choices=("smoke","dev","all","report","judge"),required=True)
    p.add_argument("--run",type=Path)
    p.add_argument("--arms",nargs="+",choices=ARMS,default=list(ARMS))
    p.add_argument("--label",default="")
    args=p.parse_args();root=args.output
    if args.stage=="judge":
        if args.run is None:p.error("单独裁判必须提供--run")
        if not args.run.resolve().is_relative_to(root.resolve()):raise ValueError("裁判目录超出本次实验")
        raise SystemExit(judge(args.run))
    if args.stage=="report":
        paths=[path for path in sorted(root.iterdir()) if path.is_dir() and (path/"manifest.json").exists()
               and read_json(path/"manifest.json").get("identity",{}).get("query_ids")]
        summary=[summarize_run(path) for path in paths]
        write_json(root/"summary.json",{"runs":summary,"human_reviewed":0,"cost_usd":None,
            "说明":"普通原生链路保留已知引用编号错位；引用支持按原生返回规则评分，不将其当作提示块语义支持率。"})
        return
    split="all" if args.stage=="all" else "dev"
    plan={"stage":args.stage,"arms":args.arms,"split":split,"planned_per_arm":1 if args.stage=="smoke" else 150 if split=="all" else 39,
          "generation_model_effort":"low","native_agentic_modes":["low","medium","high","ultra"],
          "judge_effort":"medium","selection":"完整开发集按判对数量优先、未知数量次之、平均耗时再次之；只作模型裁判暂定选择。",
          "baseline_source_sha256":digest(Path(__file__).with_name("native_baseline.py").read_bytes()),
          "driver_source_sha256":digest(Path(__file__).read_bytes()),
          "scope":"固定84份英文财报的原生问答与检索能力；外网工具、多模态、知识编译派生索引不混入该任务。"}
    import re
    if args.label and not re.fullmatch(r"[a-z0-9_-]+",args.label):raise ValueError("运行标签格式错误")
    label="-"+args.label if args.label else ""
    planfile=root/(args.stage+label+"-plan.json")
    if planfile.exists():raise ValueError("计划已存在，不自动续发或覆盖")
    write_json(planfile,plan)
    for arm in args.arms:
        suffix="-smoke" if args.stage=="smoke" else "-"+split
        path=root/(arm+label+suffix)
        if path.exists():
            raise ValueError("输出已存在，禁止自动跳过或覆盖："+str(path))
        argv=[sys.executable,"-m","timeguard.native_baseline","--output",str(root),"--arm",arm]
        argv.extend(["--split",split])
        if args.label:argv.extend(["--label",args.label])
        if args.stage=="smoke":argv.extend(["--limit","1"])
        with (root/(arm+"-"+args.stage+".log")).open("w") as log:
            code=subprocess.call(argv,stdout=log,stderr=log)
        if args.stage in ("dev","all") and (path/"results.jsonl").exists():
            judge_code=judge(path)
        else:judge_code=None
        with (root/"matrix-execution.jsonl").open("a") as log:
            log.write(json.dumps({"arm":arm,"stage":args.stage,"generation_returncode":code,"judge_returncode":judge_code})+"\n")
        print(arm,args.stage,"生成退出码",code,"裁判退出码",judge_code,flush=True)


if __name__=="__main__":main()
