"""仅离线检查合成请求的完整实收块，逐项区分数值出现与期间、单位核验。"""
import argparse
import re
from pathlib import Path

from .common import read_json,read_jsonl,write_json
from .financial_agent import Operand,evidence_lines,verify_operand


def check(fact,chunks):
    pattern = r"(?<![\d.,])" + re.escape(fact["number"]) + r"(?![\d.,])"
    appeared = False
    for c in chunks:
        lines = [{"number":i,"text":line} for i,line in enumerate(evidence_lines(c["content"]),1)]
        for line in lines:
            match = re.search(pattern,line["text"])
            if not match:
                continue
            appeared = True
            label = next((x for x in fact["labels"] if x.casefold() in line["text"].casefold()),None)
            if not label:
                continue
            operand = Operand(chunk_id=c["id"],line_number=line["number"],number=fact["number"],
                              year=fact["year"],period=fact["period"],currency=fact["currency"],scale=fact["scale"],
                              row_label=label,kind="money")
            try:
                verify_operand(operand,c,lines)
            except ValueError:
                continue
            return {"数值出现":True,"行列单位核验":True}
    return {"数值出现":appeared,"行列单位核验":False}


def audit(run,rubric):
    results={}
    for row in read_jsonl(run/"results.jsonl"):
        if row["id"] not in rubric:
            continue
        answers=row["native_trace"]["evidence_layers"]["answers"]
        reqs=[r for a in answers for r in a["requests"] if r.get("status")==200]
        if len(reqs)!=1 or reqs[0]["complete_chunks"] is None:
            results[row["id"]]={"可核验":False}
            continue
        facts=[check(f,reqs[0]["complete_chunks"]) for f in rubric[row["id"]]]
        results[row["id"]]={"可核验":True,"应有操作数":len(facts),"数值出现数":sum(f["数值出现"] for f in facts),
                            "严格核验数":sum(f["行列单位核验"] for f in facts)}
    return results


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("run","rubric","output"):
        p.add_argument(key,type=Path)
    args=p.parse_args()
    write_json(args.output,audit(args.run,read_json(args.rubric)))
