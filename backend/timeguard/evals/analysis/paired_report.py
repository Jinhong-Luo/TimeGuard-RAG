"""配对生成结束后按题对齐裁判、页覆盖和请求成本。"""
import argparse
from collections import Counter
from pathlib import Path

from timeguard.common import digest,read_json,read_jsonl,write_json
from timeguard.evals.analysis.layer_report import page_set


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
    if set(arms) & {"page","supplement"}:
        pages = read_json(root/"pages-private.json")
        result["检索"] = {}
        for qid, r in pages.items():
            targets = {(e["doc_name"].casefold(),e["evidence_page_num"]+1) for e in labels[qid]["evidence"]}
            result["检索"][qid] = {k:r[k] for k in ("index_reads","page_searches","extra_searches","page_seconds","extra_seconds")}
            for variant in ("page","supplement"):
                hits,valid = page_set(r[variant])
                result["检索"][qid][variant+"命中题页数"] = len(hits & targets) if valid else None
    if "formula" in arms:
        guards=read_json(root/"guards-private.json")
        result["公式核验"]={qid:{key:r.get(key) for key in ("active","metric","reads","required","verified","calculated","seconds")}
                          for qid,r in guards.items()}
    if "history_read" in arms:
        saved = read_json(root/"history-private.json")
        final = {}
        for arm in arms:
            final[arm] = {}
            for row in read_jsonl(root/arm/"results.jsonl"):
                answer, = row["native_trace"]["evidence_layers"]["answers"]
                request, = answer["requests"]
                final[arm][row["id"]] = (digest(request["body"]), answer.get("history_fallback"),
                                         answer.get("history_displaced", 0))
        result["历史回读"] = {
            qid: {**r["audit"], "补入块数": len(r["chunks"]),
                  "最终请求相同": final["original"][qid][0] == final["history_read"][qid][0],
                  "原有实收块减少数": final["history_read"][qid][2],
                  "预算回退": final["history_read"][qid][1]}
            for qid, r in saved.items()}
    if "review" in arms:
        reviews = read_json(root/"review-private.json")
        result["审查"] = {
            qid: {"请求":r["request_count"],"耗时秒":r["seconds"],"回退":r["fallback"],
                  "错误类型":r.get("error"),"用量":r.get("usage"),
                  "选取块数":len(r.get("ids",[]))}
            for qid,r in reviews.items()}
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("root","gold","output"):
        p.add_argument(key,type=Path)
    p.add_argument("arms",nargs=2)
    args = p.parse_args()
    write_json(args.output,compare(args.root,args.gold,args.arms))
