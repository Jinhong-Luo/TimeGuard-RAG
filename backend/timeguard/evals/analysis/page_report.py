"""离线区分历史漏页、同批基线召回、页级选取和一次补查。"""
import argparse
from collections import Counter
from pathlib import Path

from timeguard.common import read_json,read_jsonl,write_json
from timeguard.evals.analysis.layer_report import layers,page_set


def report(old_path,source,root,gold_path):
    old = {r["id"]:r for r in read_json(old_path)}
    labels = {r["id"]:r for r in read_jsonl(gold_path)}
    runs = {arm:{r["id"]:r for r in read_jsonl(root/arm/"model_reviews.jsonl")}
            for arm in ("original","page","supplement")}
    search = read_json(root/"pages-private.json")
    groups={}
    records={}
    for row in read_jsonl(source/"results.jsonl"):
        qid=row["id"]
        prior=old.get(qid,{})
        group = ("历史错且标准证据对且漏页" if prior.get("normal_correct") is False and prior.get("oracle_correct") is True
                 and not prior["hit_pages"]["candidates"] else "历史判对" if prior.get("normal_correct") else "调优")
        stats=groups.setdefault(group,Counter())
        targets={(e["doc_name"].casefold(),e["evidence_page_num"]+1) for e in labels[qid]["evidence"]}
        actual,valid=layers(row)["candidates"]
        hit = {"原链路任一轮检索":actual & targets if valid else None}
        for arm in ("page","supplement"):
            p,valid=page_set(search[qid][arm])
            hit[arm]=p & targets if valid else None
        scores={arm:reviews[qid]["answer_correct"] for arm,reviews in runs.items()}
        stats["题数"]+=1
        for key,h in hit.items():
            if h is not None:
                stats[key+"命中题数"]+=bool(h)
                stats[key+"命中组合"]+=len(h)
        stats["题页组合数"]+=len(targets)
        if hit["page"] and scores["page"] is True and scores["original"] is False:
            stats["页级命中且错转对"]+=1
        for left,right in (("original","page"),("page","supplement")):
            stats[left+"到"+right+"错转对"]+=scores[left] is False and scores[right] is True
            stats[left+"到"+right+"对转错"]+=scores[left] is True and scores[right] is False
        records[qid]={"分组":group,"题页数":len(targets),"命中数":{k:len(h) if h is not None else None for k,h in hit.items()},
                      "模型判对":scores,"补查次数":search[qid]["extra_searches"]}
    return {"分组":groups,"逐题":records}


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("old","source","root","gold","output"):
        p.add_argument(name,type=Path)
    args=p.parse_args()
    write_json(args.output,report(args.old,args.source,args.root,args.gold))
