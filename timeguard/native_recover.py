"""网络恢复后在新目录完成中断的核心组，保留原失败及其完整计划分母。"""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import time

from .common import digest,read_json,read_jsonl,write_json,write_jsonl
from .native_baseline import container_worker,private_runtime
from .native_full import audit


def recovery_plan(queries,rows):
    ids={q["id"] for q in queries}
    if len(ids)!=len(queries) or len({r["id"] for r in rows})!=len(rows) or {r["id"] for r in rows}-ids:
        raise ValueError("恢复来源题号重复或超出原计划")
    good={r["id"]:deepcopy(r) for r in rows if r["status"]=="ok"}
    questions={q["id"]:q["question"] for q in queries}
    if any(r["question"]!=questions[r["id"]] for r in good.values()):raise ValueError("来源题目已变化")
    return good,[q for q in queries if q["id"] not in good]


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--source",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);args=p.parse_args()
    from . import native_baseline
    from .matrix import frozen_index
    source=args.source;out=args.output;manifest=read_json(source/"manifest.json");spec=read_json(source/"spec.json")
    if (digest(spec)!=manifest["identity"]["spec_sha256"] or len(spec["queries"])!=39
            or digest(Path(native_baseline.__file__).read_bytes())!=manifest["identity"]["source_sha256"]):
        raise ValueError("恢复只接受未改变原生算法和参数的完整39题组")
    freeze=frozen_index(Path("data/ingestion.json"),Path("runs/r0-gpu-retrieval-all"),Path("data/financebench/queries.jsonl"))
    if freeze!=manifest["identity"]["freeze"]:raise ValueError("索引来源改变，不能续接")
    raw=(source/"results.jsonl").read_bytes();lines=[l for l in raw.splitlines() if l.strip()];rows=[json.loads(l) for l in lines]
    good,pending=recovery_plan(spec["queries"],rows)
    if not pending:raise ValueError("原组已完成，不需要恢复")
    proof={"source_run":source.name,"manifest_sha256":digest((source/"manifest.json").read_bytes()),
        "results_sha256":digest(raw),"original_planned":39,"original_completed":len(rows),
        "original_status_counts":dict(Counter(r["status"] for r in rows)),"original_missing":39-len(rows),
        "reused_success_count":len(good),"new_count":len(pending),
        "reason":"专用SSH代理失效，已记录恢复证据；原失败和未发题仍留在原实验的39题分母。"}
    out.mkdir(exist_ok=False);fresh=out/"new-after-network";fresh.mkdir()
    write_json(out/"manifest.json",{**manifest,"identity":{**manifest["identity"],"recovery":proof},
                                  "recovery_executor_sha256":digest(Path(__file__).read_bytes())})
    write_json(out/"spec.json",spec);fresh_spec={**spec,"queries":pending};write_json(fresh/"spec.json",fresh_spec)
    private,_=private_runtime();start=(private/"bridge-audit.jsonl").stat().st_size;began=time.monotonic()
    code=container_worker(fresh_spec,fresh);new=read_jsonl(fresh/"results.jsonl")
    for line in lines:
        row=json.loads(line)
        if row["id"] in good:good[row["id"]]["recovery_origin"]={"results_sha256":digest(raw),"row_sha256":digest(line)}
    combined={**good,**{r["id"]:r for r in new}}
    write_jsonl(out/"results.jsonl",[combined[q["id"]] for q in spec["queries"] if q["id"] in combined])
    old=read_json(source/"execution.json")
    write_json(out/"execution.json",{**audit(private,start,"low"),"returncode":code,"planned":39,"completed":len(combined),
        "wall_seconds":time.monotonic()-began,"new_count":len(new),"reused_count":len(good),
        "reused_source_calls":old["actual_calls"],"reused_source_tokens":old["tokens"],"reused_source_wall_seconds":old["wall_seconds"],
        "source_failed_seconds":sum(r["elapsed_seconds"] for r in rows if r["status"]!="ok"),"recovery":proof})
    raise SystemExit(code)


if __name__=="__main__":main()
