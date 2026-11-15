"""从可见行和索引回读原文交叉验证财务操作数，再按明确的财务公式计算。"""
from decimal import Decimal
import re

from .financial_agent import Operand, amount, calculate_financial_metric, evidence_lines, verify_operand

ROWS = {
    "收入":r"^(?:total )?(?:net )?(?:revenues?|sales)$",
    "固定资产净额":r"^(?:(?:net )?property(?:,? plant)? and equipment|property, plant and equipment)(?:,? net)?$",
    "现金及等价物":r"^cash and cash equivalents$",
    "短期投资":r"^(?:short.term investments|marketable securities)$",
    "应收款净额":r"^(?:accounts? receivable|receivables)(?:,? net)?$",
    "流动资产":r"^(?:total )?current assets$",
    "存货":r"^(?:total )?inventor(?:y|ies)(?:,? net)?$",
    "流动负债":r"^(?:total )?current liabilities$",
    "毛利":r"^(?:total )?gross profit$",
    "毛利率":r"^gross (?:profit )?margin(?: %)?$",
    "资本开支":r"^(?:capital expenditures?|purchases of (?:property(?:,? plant)? and equipment|fixed assets))$",
}
NUM = re.compile(r"(?<![\w.])(?:\(\d[\d,]*(?:\.\d+)?\)|-?\d[\d,]*(?:\.\d+)?)(?!\w)")


def requirements(q):
    q=q.casefold()
    years=sorted({int(y) for y in re.findall(r"(?:19|20)\d{2}",q)})
    if not years or (re.search(r"\bquarter|\bq[1-4]\b|\bh[12]\b",q) and "quick ratio" not in q):
        return None
    y=years[-1]
    compare = re.search(r"improv|declin|trend|compar|between|change",q)
    periods = years if len(years)>1 or not compare else [y-1,y]
    if "fixed asset turnover" in q:
        return "固定资产周转率",[("收入",y),("固定资产净额",y-1),("固定资产净额",y)]
    if "quick ratio" in q and "receivable" in q and "cash" in q:
        rows = ["现金及等价物", "应收款净额"]
        if re.search(r"short.term investment|marketable securit",q):
            rows.insert(1,"短期投资")
        return "速动比率（列举的速动资产）",[(m,period) for period in periods for m in [*rows,"流动负债"]]
    if "quick ratio" in q and not re.search("cash (?:and|plus)|marketable securities|prepaid",q):
        return "速动比率",[(m,period) for period in periods for m in ("流动资产","存货","流动负债")]
    if "gross margin" in q and len(years)==2 and re.search("change|difference|percentage point",q):
        return "百分点差",[("毛利率",y),("毛利率",years[0])]
    if re.search("sales|revenue",q) and "growth" in q and len(years)==2:
        return "收入增长率",[("收入",y),("收入",years[0])]
    if "gross margin" in q and compare:
        return "多期毛利率",[(m,period) for period in periods for m in ("毛利","收入")]
    if "gross margin" in q and len(years)==1:
        return "毛利率",[("毛利",y),("收入",y)]
    if re.search(r"capital expenditure|\bcapex\b",q) and len(years)==1 and not re.search(r"ratio|percent|growth|change",q):
        return "资本开支",[("资本开支",y)]
    return None


def formula(metric,operands):
    if metric.startswith("速动比率") and len({o.year for o in operands})>1:
        return "；".join(f"{year} 年 "+formula(metric,[o for o in operands if o.year==year])
                         for year in sorted({o.year for o in operands}))
    if metric=="多期毛利率":
        return "；".join(f"{year} 年 "+formula("毛利率",[o for o in operands if o.year==year])
                         for year in sorted({o.year for o in operands}))
    if metric in ("毛利率","收入增长率","百分点差"):
        return calculate_financial_metric({"毛利率":"margin","收入增长率":"growth","百分点差":"percentage_point_change"}[metric],*operands)
    if len({(v.currency,v.scale,v.period) for v in operands})!=1:
        raise ValueError("币种、单位或期间不一致")
    if operands[0].currency=="XXX" and len({v.chunk_id for v in operands})!=1:
        raise ValueError("跨表计算缺少明确币种")
    vals=[amount(v) for v in operands]
    if metric=="资本开支":
        return f"原始列示金额 {vals[0]}，支出规模 {abs(vals[0])} {operands[0].currency} {operands[0].scale}"
    if metric=="速动比率":
        a,inv,liab=vals
        if not liab:raise ValueError("分母为零")
        return f"（流动资产－存货）÷流动负债 = ({a} - {inv}) / {liab} = {(a-inv)/liab:.4f}"
    if metric=="速动比率（列举的速动资产）":
        if not vals[-1]:raise ValueError("分母为零")
        return "所列速动资产合计÷流动负债 = ("+" + ".join(map(str,vals[:-1]))+f") / {vals[-1]} = {sum(vals[:-1])/vals[-1]:.4f}"
    if metric!="固定资产周转率":
        raise ValueError("未识别公式")
    rev,begin,end=vals
    avg=(begin+end)/Decimal(2)
    if not avg:raise ValueError("平均固定资产为零")
    return f"收入÷期初期末固定资产净额均值 = {rev} / (({begin}+{end})/2) = {rev/avg:.4f}"


def candidates(chunk,raw,metric,year,period):
    if raw["id"]!=chunk["id"] or raw["document_id"]!=chunk["document_id"]:
        raise ValueError("回读来源不匹配")
    shown=evidence_lines(chunk["content"])
    src=evidence_lines(raw["content"])
    visible=[{"number":i,"text":t} for i,t in enumerate(shown,1)]
    original=[{"number":i,"text":t} for i,t in enumerate(src,1)]
    for i,line in enumerate(shown,1):
        label=line.strip(" |").split("|")[0].strip().casefold()
        if not re.fullmatch(ROWS[metric].replace(" ",r"\s*"),label) or src.count(line)!=1:
            continue
        for match in NUM.finditer(line[len(label):]):
            for currency in (("XXX",) if metric=="毛利率" else ("USD","EUR","GBP","XXX")):
                for scale in (("ones",) if metric=="毛利率" else ("millions","thousands","billions","ones")):
                    op=Operand(chunk_id=chunk["id"],line_number=i,number=match[0],year=year,period=period,
                               currency=currency,scale=scale,kind="percentage" if metric=="毛利率" else "money",row_label=label)
                    try:
                        verify_operand(op,chunk,visible)
                        verify_operand(op.model_copy(update={"line_number":src.index(line)+1}),raw,original)
                    except ValueError:
                        continue
                    yield op


def check(question,visible,allowed,read):
    plan=requirements(question)
    if plan is None:
        return {"active":False,"reads":0,"instruction":""}
    metric,needs=plan
    from .scope import quarters
    quarter = quarters(question)
    period = f"Q{quarter[0]}" if len(quarter)==1 else "FY"
    options={key:{} for key in needs}
    reads=0
    for slot,c in enumerate(visible,1):
        if c["document_id"] not in allowed:
            raise ValueError("取数块越界")
        if not any(re.search(ROWS[m][1:-1].replace(" ",r"\s*"),c["content"],re.I) for m,_ in needs):
            continue
        raw=read(c)
        reads+=1
        # 部门或地区表需要问题明确指定相同口径；无法确定则不给全公司比率背书。
        if re.search(r"segment|division|geographic",raw["content"],re.I) and not re.search(r"consolidated",raw["content"],re.I):
            continue
        if re.search(r"\b(?:segment|division|department|domestic|international|geographic|business unit)\b",question,re.I):
            continue
        for key in needs:
            for op in candidates(c,raw,*key,period):
                options[key].setdefault((amount(op),op.currency,op.scale),(op,c.get("citation_number",slot)))
    missing=[f"{y} 年{m}" for m,y in needs if len(options[m,y])!=1]
    note={"active":True,"metric":metric,"reads":reads,"required":len(needs),"verified":len(needs)-len(missing),"missing":missing}
    if missing:
        note["instruction"]="行列与金额单位尚未全部核验："+"、".join(missing)+"。只能引用实际证据中可复核的期间与口径；若操作数不足，列出缺项。"
    else:
        facts=[next(iter(options[key].values())) for key in needs]
        try:
            result=formula(metric,[op for op,_ in facts])
            note["calculated"]=True
        except ValueError as exc:
            result=str(exc)
            note["calculated"]=False
        note["instruction"]="原文校验结果："+result+"。只在列出的原块实际可见时使用。"
        note["facts"]=[{"chunk_id":op.chunk_id,"year":op.year,"number":op.number,"marker":slot} for op,slot in facts]
        note["instruction"]+=" "+"；".join(f"{op.year} 年 {op.row_label} {op.number} {op.scale} [ID:{slot}]" for op,slot in facts)
    if metric.startswith("速动比率"):
        note["instruction"]+=" 明示所采用的速动比率公式；若问题指定不同口径，遵循题目定义。"
    return note


def guard_row(row,client):
    import time
    from urllib.parse import quote
    started=time.monotonic()
    a,=row["native_trace"]["evidence_layers"]["answers"]
    req,=[r for r in a["requests"] if r.get("status")==200]
    dataset=a["snapshot"]["input"]["identity"]["dataset_id"]
    hashes=a["block_hashes"]
    visible=[{**c,"citation_number":hashes.index(c["block_sha256"])+1} for c in req["complete_chunks"]
             if hashes.count(c["block_sha256"])==1]
    def read(c):
        result=client.request("GET",f"/datasets/{dataset}/documents/{quote(c['document_id'],safe='')}/chunks?id={quote(c['id'],safe='')}")
        raw,=result["chunks"]
        return raw
    result=check(row["question"],visible,a["snapshot"]["input"]["tools"]["doc_scope"],read)
    result["seconds"]=time.monotonic()-started
    return result
