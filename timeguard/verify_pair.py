"""比对当前源码重建的请求与已成功发送的请求，不调用模型。"""
import argparse
from pathlib import Path
from .common import read_json,read_jsonl
from .paired_trial import render


def verify(source,root,arms):
    index={r["id"]:r for r in read_jsonl(source/"results.jsonl")}
    ids=read_json(root/arms[0]/"manifest.json")["identity"]["query_ids"]
    rows=[index[k] for k in ids]
    reviews=read_json(root/"review-private.json") if "review" in arms else None
    pages=read_json(root/"pages-private.json") if set(arms) & {"page","supplement"} else None
    guards=read_json(root/"guards-private.json") if "formula" in arms else None
    rebuilt=render(rows,arms,reviews,pages,guards)
    for arm in arms:
        actual=read_jsonl(root/arm/"results.jsonl")
        for src,v,r in zip(rows,rebuilt,actual,strict=True):
            a,=r["native_trace"]["evidence_layers"]["answers"]
            req,=a["requests"]
            if (src["id"]!=r["id"] or req.get("status")!=200 or
                v[arm]["body"]!=req["body"] or v[arm]["selected"]!=a["selected"] or
                v[arm]["visible"]!=req["complete_chunks"]):
                raise ValueError("当前实现与已发送的配对记录不一致")
        print(arm,len(actual),"题重建一致")


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("source",type=Path)
    p.add_argument("root",type=Path)
    p.add_argument("arms",nargs="+")
    args=p.parse_args()
    verify(args.source,args.root,args.arms)
