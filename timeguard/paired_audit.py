"""用响应编号连接专用模型桥的落盘审计，只导出计数。"""
import argparse
from collections import Counter
from pathlib import Path

from .common import read_jsonl,write_json


def crosscheck(armpath, audit):
    rows=read_jsonl(armpath/"results.jsonl")
    counts=Counter()
    for row in rows:
        requests=[r for a in row["native_trace"]["evidence_layers"]["answers"] for r in a["requests"]]
        for req in requests:
            if req.get("status")!=200:
                counts["未成功回答"]+=1
                continue
            entry=audit.get(req.get("request_id"))
            if not entry or not entry["success"] or entry["actual_model"]!="gpt-6.1-sol" or entry["reasoning_effort"]!="low":
                raise ValueError("响应编号无法与指定模型的成功审计匹配")
            actual=entry["usage"] or {}
            usage=row.get("usage") or {}
            for key,other in (("prompt_tokens","inputTokens"),("completion_tokens","outputTokens"),("total_tokens","totalTokens")):
                if usage.get(key) is None or usage[key]!=actual.get(other):
                    raise ValueError("响应和落盘审计用量不一致")
            counts["模型与用量一致"]+=1
    return dict(counts)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("audit",type=Path)
    p.add_argument("output",type=Path)
    p.add_argument("arms",nargs="+",type=Path)
    args=p.parse_args()
    rows=read_jsonl(args.audit)
    audit={r["request_id"]:r for r in rows}
    if len(rows)!=len(audit):
        raise ValueError("桥审计编号重复")
    write_json(args.output,{str(d):crosscheck(d,audit) for d in args.arms})
