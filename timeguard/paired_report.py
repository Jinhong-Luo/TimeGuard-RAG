"""配对生成结束后按题对齐裁判、页覆盖和请求成本。"""
import argparse
from collections import Counter
from pathlib import Path

from .common import digest,read_json,read_jsonl,write_json
from .layer_report import page_set


def measure(run, labels):
    reviews = {r["id"]:r for r in read_jsonl(run/"model_reviews.jsonl")}
    out = {}
    for r in read_jsonl(run/"results.jsonl"):
        a, = r["native_trace"]["evidence_layers"]["answers"]
        req, = a["requests"]
        targets = {(e["doc_name"].casefold(),e["evidence_page_num"]+1) for e in labels[r["id"]]["evidence"]}
        rv = reviews[r["id"]]
        out[r["id"]] = {"判对":rv["answer_correct"],"引用支持":rv["citation_support"],"耗时":r["elapsed_seconds"],
                          "生成请求":1,"生成失败":int(r["status"]!="ok"),"用量":r.get("usage"),
                          "裁判耗时":rv["elapsed_seconds"],"裁判用量":rv.get("usage"),"标准页数":len(targets),
                          "选块顺序指纹":digest(a["selected"]),"选中块数":len(a["selected"]),"装入块数":len(a["packet"]),
                          "实收整块数":len(req.get("complete_chunks") or []) if req.get("status")==200 else None,
                          "输入指纹":a["snapshot"]["sha256"]}
        for k, chunks in (("选中",a["selected"]),("装入",a["packet"]),("实收",req.get("complete_chunks"))):
            pages,valid = page_set(chunks)
            out[r["id"]][k] = len(pages & targets) if valid else None
    return out


def compare(root, gold, arms):
    labels = {r["id"]:r for r in read_jsonl(gold)}
    per_arm = {arm:measure(root/arm,labels) for arm in arms}
    left,right = per_arm.values()
    if left.keys()!=right.keys() or any(left[k]["输入指纹"]!=right[k]["输入指纹"] for k in left):
        raise ValueError("配对题号或入口快照不一致")
    result = {"类型":"固定快照小样本，两个方案均重新生成和评分","题数":len(left),
              "错转对":[k for k in left if left[k]["判对"] is False and right[k]["判对"] is True],
              "对转错":[k for k in left if left[k]["判对"] is True and right[k]["判对"] is False]}
    for arm,rows in per_arm.items():
        summary = Counter()
        for r in rows.values():
            summary[{True:"判对",False:"判错",None:"未知"}[r["判对"]]] += 1
            summary["判对且引用支持"] += int(r["判对"] is True and r["引用支持"] is True)
            for field in ("选中块数","装入块数","实收整块数","选中","装入","实收","标准页数","生成请求","生成失败","耗时"):
                if r[field] is None:
                    summary[field+"未知题数"] += 1
                else:
                    summary[field] += r[field]
            for key in ("prompt_tokens","completion_tokens","total_tokens"):
                if r["用量"] is not None:
                    summary[key] += r["用量"][key]
                else:
                    summary[key+"未知题数"] += 1
        audit = read_json(root/arm/"judge-audit.json")
        result[arm] = {"汇总":dict(summary),"裁判":{k:audit[k] for k in ("actual_calls","failed_calls","tokens","wall_seconds")},
                       "逐题":rows}
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("root","gold","output"):
        p.add_argument(key,type=Path)
    p.add_argument("arms",nargs=2)
    args = p.parse_args()
    write_json(args.output,compare(args.root,args.gold,args.arms))
