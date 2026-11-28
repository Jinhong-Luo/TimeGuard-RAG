"""核对历史原生编号与成功请求中的证据块，保留无法定位的引用。"""
import argparse
from collections import Counter
from pathlib import Path
from timeguard.common import read_jsonl,write_json
from timeguard.evals.judge import CITATION_MARKER,visible_answer


def audit(rows):
    totals=Counter()
    for row in rows:
        totals["题数"]+=1
        t=row["native_trace"]
        answers=t.get("evidence_layers",{}).get("answers",[])
        refs=row.get("reference",{}).get("chunks",[])
        markers={int(m[1]) for m in CITATION_MARKER.finditer(visible_answer(row.get("answer")))}
        totals["显式编号题数"]+=bool(markers)
        for i in markers:
            totals["题目与编号组合"]+=1
            if len(answers)!=1 or t.get("citation_origin")!="model":
                totals["来源未知"]+=1
                continue
            a=answers[0]
            req=[r for r in a["requests"] if r["status"]==200]
            packet=a.get("packet") or []
            if not 0<i<=len(packet) or len(req)!=1:
                totals["实收无法定位"]+=1
                continue
            expected=packet[i-1]
            pair=lambda c:(c.get("document_id"),c.get("id"))
            same=[c for c in req[0].get("complete_chunks") or [] if pair(c)==pair(expected)]
            if len(same)!=1:
                totals["实收无法定位"]+=1
            elif i<len(refs) and pair(same[0])==pair(refs[i]) and same[0].get("content")==refs[i].get("content"):
                totals["旧映射与实收相同"]+=1
            else:
                totals["旧映射与实收不一致"]+=1
    return dict(totals)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("history",type=Path)
    p.add_argument("output",type=Path)
    args=p.parse_args()
    write_json(args.output,audit(read_jsonl(args.history)))
