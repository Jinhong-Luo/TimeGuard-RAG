"""用问题与合法财报分流查数，按缺项决定是否升级一次审查。"""
import re
from decimal import Decimal

from .common import digest
from .page_select import features


def simple_metric(question):
    q=question.casefold()
    years=set(re.findall(r"(?:19|20)\d{2}",q))
    if len(years)!=1 or not re.search(r"usd (?:millions|billions|thousands)",q):
        return None
    if re.search(r"growth|compare|margin|ratio|change|why|explain|segment|quarter|\bq[1-4]\b",q):
        return None
    labels={"资本开支":r"capital expenditure|\bcapex\b","固定资产净额":r"net (?:ppne|pp&e|property,? plant)",
            "收入":r"(?:net|total) (?:sales|revenues?)","应收款净额":r"net (?:ar|receivables?|accounts receivable)"}
    return next(((k,int(next(iter(years)))) for k,pattern in labels.items() if re.search(pattern,q)),None)


def needs_review(question,chunks):
    _,years,groups=features(question)
    if len(years)>1 or re.search(r"average|historically|trend|between.+and|year.over.year|multi.year",question,re.I):
        return "多期"
    if not chunks:
        return "无候选"
    texts=[features(c.get("content") or c.get("content_with_weight") or "") for c in chunks]
    if any(not any(y in cy and (not groups or bool(cg & groups)) for _,cy,cg in texts) for y in years):
        return "缺少同期间指标"
    if groups and not any(cg & groups for _,_,cg in texts):
        return "缺少目标指标"
    return None


def simple_lookup(question, chunks, allowed, read):
    from .formula_guard import candidates
    from .financial_agent import amount
    metric=simple_metric(question)
    if not metric:
        return None
    name,year=metric
    values={}
    for c in chunks:
        if c["document_id"] not in allowed:
            raise ValueError("轻量查数超出本题范围")
        raw=read(c)
        if re.search(r"segment|division",c["content"],re.I) and not re.search(r"consolidated",c["content"],re.I):
            continue
        for op in candidates(c,raw,name,year,"FY"):
            if op.currency!="USD":
                continue
            scale={"ones":1,"thousands":1000,"millions":10**6,"billions":10**9}[op.scale]
            values.setdefault(amount(op)*scale,c)
    if len(values)!=1:
        return None
    value,source=next(iter(values.items()))
    unit=re.search(r"USD (millions|billions|thousands)",question,re.I)[1].lower()
    divisor={"thousands":1000,"millions":10**6,"billions":10**9}[unit]
    result=value/Decimal(divisor)
    if name=="资本开支":
        result=abs(result)
    text=f"{year} 财年{name}为 {result:f} USD {unit} [ID:0]。"
    return {"answer":text,"reference":{"chunks":[source]},
            "citation_trace":{"version":1,"origin":"rule","raw_answer":text,
                              "visible":{"0":{"chunk_id":source["id"],"document_id":source["document_id"],
                                              "content_sha256":digest(source["content"])}}}}
