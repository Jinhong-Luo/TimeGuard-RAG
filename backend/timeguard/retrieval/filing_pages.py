"""在合法财报的原生索引内按 PDF 页检索，并对可定位的缺项最多补查一次。"""
from collections import Counter, defaultdict
import math
import re
import time

from timeguard.retrieval.page_select import features
from timeguard.retrieval.scope import scope

# 将财报常见缩写与页标题、会计科目对齐；适用于全部公司。
TERMS = {
    "receivable":r"\b(?:net ar|accounts? receivable|receivables?)\b",
    "litigation":r"\b(?:legal (?:battles?|proceedings?)|litigation|lawsuits?|contingencies)\b",
    "margin":r"\b(?:margins?|profitability)\b",
    "investments":r"\b(?:investments?|securities)\b",
    "liquidity":r"\b(?:liquidity|liquid assets?|cash equivalents?)\b",
}


def terms(text):
    words,years,groups = features(text)
    return words | years | groups | {k for k,p in TERMS.items() if re.search(p,text,re.I)}


def pages_for(question, chunks, count=4, exclude=()):
    grouped = defaultdict(list)
    query = terms(question)
    for c in chunks:
        for p in {pos[0] for pos in c["positions"] if pos and type(pos[0]) is int and pos[0]>0}:
            grouped[c["document_id"],p].append(c)
    tokens,df = {},Counter()
    for page,rows in grouped.items():
        merged=set()
        for c in rows:
            key = (c["document_id"],c["id"])
            if key not in tokens:
                tokens[key] = terms(c["content"])
            merged |= tokens[key]
        df.update(merged & query)
    weights={t:math.log1p((len(grouped)+1)/(df[t]+1)) for t in query}
    def score(c):
        found=tokens[c["document_id"],c["id"]] & query
        return sum(weights[t] for t in sorted(found))
    ordered=[]
    for page,rows in grouped.items():
        if page in exclude:
            continue
        ordered_rows=sorted(rows,key=lambda c:(-score(c),c["id"]))
        union=set().union(*(tokens[c["document_id"],c["id"]] & query for c in rows))
        value=score(ordered_rows[0])+0.5*sum(weights[t] for t in sorted(union))
        if value>0:
            ordered.append((-value,page,ordered_rows[:3]))
    ordered.sort(key=lambda item:(item[0],item[1]))
    seen=set()
    chunks=[]
    for _,page,rows in ordered[:count]:
        for c in rows:
            key=(c["document_id"],c["id"])
            if key not in seen:
                seen.add(key)
                chunks.append(c)
    return chunks,[p for _,p,_ in ordered[:count]]


def missing_query(question,chunks):
    _,years,groups=features(question)
    if not chunks:
        return question
    facts = groups | {k for k,p in TERMS.items() if re.search(p,question,re.I)}
    passages = [line for c in chunks for line in re.split(r"\n\s*\n|(?<=[.;])\s+",c["content"]) if line.strip()]
    body = [terms(line) for line in passages]
    if re.search(r"historically|over time|trend|each year",question,re.I):
        periods = set().union(*(features(line)[1] for line in passages if not facts or terms(line) & facts))
        if len(periods)<2:
            return question+" comparative annual periods"
    for year in sorted(years):
        if not any(year in passage and (not facts or bool(facts & passage)) for passage in body):
            # 聚焦到缺失期间，保留原问题的指标、报表及部门限定。
            return re.sub(r"(?:FY\s*)?(?:19|20)\d{2}", "", question, flags=re.I) + " fiscal year " + year
    for key,pattern in TERMS.items():
        if re.search(pattern,question,re.I) and not any(re.search(pattern,c["content"],re.I) for c in chunks):
            return question+" "+key
    return None


class FilingPages:
    def __init__(self,client,dataset,documents):
        self.client,self.dataset,self.documents=client,dataset,documents
        self.by_id={d["id"]:d for d in documents}
        self.cache={}

    def load(self,docid):
        page=1
        result=[]
        while True:
            data=self.client.request("GET",f"/datasets/{self.dataset}/documents/{docid}/chunks?page={page}&page_size=100")
            rows=data["chunks"]
            if data.get("doc",{}).get("id")!=docid or any(c.get("document_id")!=docid for c in rows):
                raise ValueError("原生索引返回其他报告")
            result.extend(rows)
            if len(result)>=data["total"]:
                break
            if not rows:
                raise ValueError("文档分页中断")
            page+=1
        if len(result)!=data["total"] or len({c["id"] for c in result})!=len(result):
            raise ValueError("文档分页重复或缺失")
        self.cache[docid]=[{**c,"document_name":self.by_id[docid]["doc_name"]} for c in result
                          if c.get("available") and c.get("content") and c.get("positions")]
        return page

    def retrieve(self,question,snapshot,as_of=None,allow_extra=True):
        started=time.monotonic()
        plan=scope(question,self.documents,as_of)
        allowed=set(plan["document_ids"] or []) if plan["companies"] else set()
        allowed.intersection_update(snapshot["input"]["tools"]["doc_scope"] or [])
        requests=0
        for d in sorted(allowed):
            if d not in self.cache:
                requests+=self.load(d)
        chunks=[c for d in sorted(allowed) for c in self.cache[d]]
        first,pages=pages_for(question,chunks)
        first_time=time.monotonic()-started
        gap=missing_query(question,first) if allowed and allow_extra else None
        extra=[]
        extra_time=0
        extra_error=None
        if gap:
            started=time.monotonic()
            try:
                extra,_=pages_for(gap,chunks,count=2,exclude=pages)
            except (TimeoutError,ConnectionError) as exc:
                extra_error=type(exc).__name__
                extra=[]
            extra_time=time.monotonic()-started
        keys={(c["document_id"],c["id"]) for c in first}
        extra=[c for c in extra if (c["document_id"],c["id"]) not in keys]
        return {"page":first,"supplement":first if not extra else (first[:max(0,12-len(extra))]+extra),
                "allowed":sorted(allowed),"index_reads":requests,"page_searches":int(bool(allowed)),
                "extra_searches":int(bool(gap)),"gap":gap,"extra_error":extra_error,"page_seconds":first_time,"extra_seconds":extra_time}
