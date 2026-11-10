"""在项目容器中重放两道历史零检索题，保存节点验证摘要。"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import shutil

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from timeguard.common import digest, read_json, read_jsonl, write_json
from timeguard.native_baseline import CONTAINER


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    spec=read_json(args.source/"spec.json")
    cases=[]
    for row in read_jsonl(args.source/"results.jsonl"):
        events=row["native_trace"]["events"]
        if any(e["kind"]=="retrieval" for e in events) or row["reference"].get("chunks"):
            continue
        allowed=next((e["selected_doc_ids"] for e in events if e["kind"]=="metadata_filter"),None)
        if allowed and allowed!=["-999"]:
            cases.append({"id":row["id"],"question":row["question"],"allowed":allowed})
    if not cases:
        raise ValueError("来源中没有满足触发条件的题目")
    args.output.mkdir(parents=True,exist_ok=False)
    package=Path(__file__).resolve().parents[1]/"timeguard"
    files=["__init__.py","agentic_candidate.py","agentic_fallback_check.py","common.py","scope.py","score.py"]
    dest="/tmp/timeguard-fallback-"+digest([digest((package/f).read_bytes()) for f in files])[:12]
    with tempfile.TemporaryDirectory() as tmp:
        worker=Path(tmp)/"timeguard"
        worker.mkdir()
        for name in files:
            shutil.copyfile(package/name,worker/name)
        subprocess.run(["docker","exec",CONTAINER,"mkdir","-p",dest],check=True)
        subprocess.run(["docker","cp",str(worker),CONTAINER+":"+dest+"/"],check=True)
    run=subprocess.run(["docker","exec","-i","-w","/ragflow","-e","PYTHONPATH="+dest+":/ragflow",CONTAINER,
                        "python","-m","timeguard.agentic_fallback_check"],
                       input=json.dumps({**spec,"cases":cases}),text=True,check=True,stdout=subprocess.PIPE)
    result=next(json.loads(line.removeprefix("TIMEGUARD_CHECK ")) for line in run.stdout.splitlines()
                if line.startswith("TIMEGUARD_CHECK "))
    write_json(args.output/"summary.json",{"method":"注入历史空证据，真实原生检索，替代最终模型生成。","rows":result})


if __name__=="__main__":
    main()
