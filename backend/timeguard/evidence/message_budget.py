"""从整条消息扣除固定开销，按冻结顺序装入完整块。"""
import re

from timeguard.retrieval.page_select import features

MARKER = "〈财报证据装入位置〉"


def shorten_table(text, question):
    # 只缩减可按原始行完整保留的管线表格；复杂跨列 HTML 保留整块。
    lines = text.splitlines(keepends=True)
    data = [i for i,line in enumerate(lines) if line.count("|") >= 2 and re.search(r"\d",line)]
    if len(data)<3 or "<table" in text.casefold():
        return None
    years = [i for i in data if re.search(r"\b(?:19|20)\d{2}\b",lines[i])]
    if not years:
        return None
    header = years[0]
    if not re.search(r"\b(?:thousands?|millions?|billions?|percent|dollars?|usd)\b|[$%]",text,re.I):
        return None
    if not re.search(r"balance sheet|statement|cash flows|operations|income|table|sales|revenue", "".join(lines[:header+1]),re.I):
        return None
    qw, _, qg = features(question)
    reduce = set()
    kept = 0
    for i in data:
        if i<=header or i in years:
            continue
        label = lines[i].lstrip(" |").split("|")[0]
        words,_,groups = features(label)
        if (words & qw or groups & qg or re.search(r"note|unit|footnote|\*|\[\d+\]", lines[i],re.I)):
            kept += 1
        else:
            reduce.add(i)
    # 金额单位所在行和所有附注必须随表保留。
    for i in list(reduce):
        if re.search(r"\b(?:thousands?|millions?|billions?|percent|dollars?|usd|footnotes?|notes?)\b",lines[i],re.I):
            reduce.discard(i)
    if not kept or not reduce:
        return None
    return "".join(line for i,line in enumerate(lines) if i not in reduce)


def pack(messages, budget, chunks, question, render, count):
    if messages[-1]["content"].count(MARKER)!=1:
        raise ValueError("消息装入位置不唯一")
    prefix, suffix = messages[-1]["content"].split(MARKER)
    fixed = sum(count(m["content"]) for m in messages[:-1])
    if fixed + count(prefix + suffix) >= budget:
        raise ValueError("固定指令与研究摘要已用尽本题预算")
    mapped = []
    evidence = ""
    for source in chunks:
        chunk = source
        block = render(chunk,len(mapped)+1)
        if not block:
            continue
        def fits(candidate):
            return fixed + count(prefix + evidence + candidate + suffix) < budget
        if not fits(block):
            key = "content" if source.get("content") else "content_with_weight"
            compact = shorten_table(source.get(key) or "",question)
            if not compact:
                continue
            chunk = {**source,key:compact}
            block = render(chunk,len(mapped)+1)
            if not block or not fits(block):
                continue
        mapped.append((chunk,block))
        evidence += block+"\n"
    updated = [*messages[:-1],{**messages[-1],"content":prefix + evidence.removesuffix("\n") + suffix}]
    return mapped,updated
