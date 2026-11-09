"""从问题和语料目录推导检索范围。运行时不接收标准答案或题目关联文档。"""
import re
from datetime import date, datetime

from .score import doc_name

ALIASES = {
    "AMD": ("Advanced Micro Devices",), "AES Corporation": ("AES",),
    "American Express": ("Amex",), "American Water Works": ("American Water",),
    "CVS Health": ("CVS",), "JPMorgan": ("JP Morgan", "JPMorgan Chase", "J.P. Morgan", "JPM"),
    "Coca-Cola": ("Coca Cola",), "Johnson & Johnson": ("Johnson and Johnson", "J&J", "JnJ"),
    "MGM Resorts": ("MGM",), "Ulta Beauty": ("Ulta",), "Kraft Heinz": ("Kraft-Heinz",),
}
YEAR = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
AS_OF = re.compile(r"(?:\bas of\b|information available by|publicly known by|截至)\s*[:：]?\s*(\d{4}-\d{2}-\d{2})", re.I)
EN_DATE = re.compile(r"\b(?:as of|known by|available by)\s+([A-Za-z]{3,9} \d{1,2},? \d{4})\b", re.I)
CN_DATE = re.compile(r"截至\s*(\d{4})年\s*(\d{1,2})月\s*(\d{1,2})日")
QUARTER = (r"(?:\bQ1\b|\bfirst[ -]quarter\b|第一季度)", r"(?:\bQ2\b|\bsecond[ -]quarter\b|第二季度)",
           r"(?:\bQ3\b|\bthird[ -]quarter\b|第三季度)", r"(?:\bQ4\b|\bfourth[ -]quarter\b|第四季度)")


def quarters(text):
    return [i for i, pattern in enumerate(QUARTER, 1) if re.search(pattern, text, re.I)]


def scope(question, documents, as_of=None):
    cutoff_match = AS_OF.search(question)
    cutoffs = [date.fromisoformat(v) for v in (as_of, cutoff_match[1] if cutoff_match else None) if v]
    written_date = EN_DATE.search(question)
    if written_date:
        value = written_date[1].replace(",", "")
        try:
            cutoffs.append(datetime.strptime(value, "%B %d %Y").date())
        except ValueError:
            cutoffs.append(datetime.strptime(value, "%b %d %Y").date())
    cn_date = CN_DATE.search(question)
    if cn_date:
        cutoffs.append(date(*map(int, cn_date.groups())))
    if not cutoffs and re.search(r"known (?:as of|by)|information available (?:as of|by)|截至|as of\s+\d{1,2}/", question, re.I):
        raise ValueError("请用 YYYY-MM-DD 补充历史信息的截止日期")
    cutoff = min(cutoffs, default=None)
    period_text = CN_DATE.sub("", EN_DATE.sub("", AS_OF.sub("", question)))
    years = set(map(int, YEAR.findall(period_text)))
    years.update(2000 + int(value) for value in re.findall(r"\bFY\s*['’]?(\d{2})(?!\d)", period_text, re.I))
    for start, end in re.findall(r"(20\d{2})\s*(?:[-–—]|through|至)\s*(?:FY\s*)?(20\d{2})", period_text, re.I):
        if 0 <= int(end) - int(start) <= 10:
            years.update(range(int(start), int(end) + 1))
    companies = set()
    for doc in documents:
        company = doc["company"]
        for alias in (company, *ALIASES.get(company, ())):
            if re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", question, re.I):
                companies.add(company)
                break
    allowed = []
    for doc in documents:
        if companies and doc["company"] not in companies:
            continue
        if cutoff:
            published = doc.get("published_at")
            if not published or date.fromisoformat(published) > cutoff:
                continue
        allowed.append(doc["id"])
    return {"companies": sorted(companies), "years": sorted(years), "quarters": quarters(period_text),
            "as_of": cutoff.isoformat() if cutoff else None,
            "document_ids": allowed if companies or cutoff else None}


def period_score(chunk, years, documents, target_quarters=()):
    """文本所含年份优先，报告期仅作同分排序，允许晚期财报承载早期数据。"""
    content = chunk.get("content") or chunk.get("content_with_weight") or ""
    covered = set(map(int, YEAR.findall(content))) & set(years)
    report_year = documents.get(doc_name(chunk), {}).get("doc_period")
    quarter_hits = set(quarters(content)) & set(target_quarters)
    return len(covered), len(quarter_hits), int(report_year in years)


def rank(chunks, years, documents, target_quarters=()):
    if not years and not target_quarters:
        return list(chunks)
    return sorted(chunks, key=lambda c: period_score(c, years, documents, target_quarters), reverse=True)


def needs_expansion(chunks, years, target_quarters=()):
    contents = " ".join(c.get("content") or c.get("content_with_weight") or "" for c in chunks)
    return (not chunks or not set(years).issubset(map(int, YEAR.findall(contents)))
            or not set(target_quarters).issubset(quarters(contents)))


def search(client, config, question, plan, arm, documents, k, weight, reranker=""):
    params = {**config["retrieval"], "dataset_ids": config["dataset_ids"], "question": question,
              "page_size": k, "vector_similarity_weight": weight, "rerank_id": reranker}
    if arm != "A" and plan["document_ids"] is not None:
        if not plan["document_ids"]:
            return [], "无可见文档"
        params["document_ids"] = plan["document_ids"]
    def fetch(size):
        result = client.request("POST", "/retrieval", {**params, "page_size": size})
        chunks = result["chunks"]
        if "document_ids" in params and any(c.get("document_id") not in params["document_ids"] for c in chunks):
            raise ValueError("原生检索返回范围以外的文档")
        return chunks
    chunks = fetch(k)
    path = "原生排序"
    if arm in ("C", "D"):
        chunks = rank(chunks, plan["years"], documents, plan["quarters"])
        path = "期间排序"
        if arm == "D" and needs_expansion(chunks, plan["years"], plan["quarters"]):
            chunks = rank(fetch(max(24, k * 2)), plan["years"], documents, plan["quarters"])
            path = "扩大同范围候选"
    return chunks[:k], path

