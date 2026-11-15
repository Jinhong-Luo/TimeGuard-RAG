"""按问题与财报原文挑选互补证据，缺失标签不记作冲突。"""
import re

STOP = set("what which when how who from with that this does using based answer question give response details shown information primarily relying assume analyst calculate company fiscal million millions billion billions amount year years fy as for the and are was its has have per usd of to in on is at a an by or it".split())
GROUPS = {
    "资本开支": r"cap(?:ital\s+expenditure|ex)|purchases? of (?:property|plant)|additions to (?:property|plant)",
    "收入": r"\brevenues?\b|\bnet sales\b",
    "固定资产": r"pp\s*&\s*e|ppne|property,?\s+plant(?:,?\s+and)?\s+equipment|fixed assets?",
    "利润率": r"operating margins?|operating income|operating profit",
    "现金流量表": r"(?:statement[s]? of )?cash flows?",
    "资产负债表": r"balance sheets?|statements? of financial position",
    "利润表": r"(?:income|earnings) statements?|statements? of (?:income|earnings|operations)",
    "部门": r"\bsegments?\b|\bdivisions?\b",
    "合并": r"\bconsolidated\b|company[- ]wide",
}


def features(text):
    text = text.casefold()
    words = set(re.findall(r"[a-z][a-z0-9]+", text)) - STOP
    years = set(re.findall(r"(?:19|20)\d{2}", text))
    groups = {label for label, pattern in GROUPS.items() if re.search(pattern, text)}
    return words, years, groups


def select(question, chunks, allowed, cap=12):
    if allowed is None:
        raise ValueError("缺少允许文档范围")
    allowed = set(allowed)
    qw, qy, qg = features(question)
    remaining = []
    seen = set()
    for i, c in enumerate(chunks):
        doc = c.get("doc_id") or c.get("document_id")
        body = c.get("content") or c.get("content_with_weight") or ""
        if doc not in allowed:
            raise ValueError("入口候选超出允许文档")
        normalized = " ".join(body.casefold().split())
        if not normalized or (doc, normalized) in seen:
            continue
        seen.add((doc, normalized))
        words, years, groups = features(body)
        title_words, _, _ = features(c.get("docnm_kwd") or c.get("document_name") or "")
        facts = (qg & groups) | (qy & years)
        base = 3 * len(qg & groups) + len(qy & years) + len(qw & words) / max(1,len(qw)) * 3
        base += min(1, len(qw & title_words)) + float(c.get("similarity") or c.get("score") or 0)
        if "合并" in qg and "部门" not in qg and "部门" in groups and "合并" not in groups:
            if re.search(r"(?:for (?:the|our) [\w -]+ segment only|segment.only)", body, re.I):
                base -= 2
        positions = c.get("positions") or c.get("position_int") or []
        pages = {(doc,p[0]) for p in positions if isinstance(p,(list,tuple)) and p}
        remaining.append((base, facts, pages, i, c))
    result, covered, page_counts = [], set(), {}
    while remaining and len(result) < cap:
        def priority(item):
            base, facts, pages, i, _ = item
            repeats = max((page_counts.get(p,0) for p in pages),default=0)
            return base + len(facts-covered) - min(3,repeats), -i
        best = max(range(len(remaining)), key=lambda n: priority(remaining[n]))
        _, facts, pages, _, chunk = remaining.pop(best)
        result.append(chunk)
        covered.update(facts)
        for p in pages:
            page_counts[p] = page_counts.get(p,0) + 1
    return result
