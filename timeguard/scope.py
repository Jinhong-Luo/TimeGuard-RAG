"""从问题和语料目录推导检索范围。运行时不接收标准答案或题目关联文档。"""
import re
from datetime import date

from .score import doc_name

ALIASES = {
    "AMD": ("Advanced Micro Devices",), "AES Corporation": ("AES",),
    "American Express": ("Amex",), "American Water Works": ("American Water",),
    "CVS Health": ("CVS",), "JPMorgan": ("JP Morgan", "JPMorgan Chase", "J.P. Morgan"),
    "Coca-Cola": ("Coca Cola",), "Johnson & Johnson": ("Johnson and Johnson", "J&J"),
    "MGM Resorts": ("MGM",), "Ulta Beauty": ("Ulta",), "Kraft Heinz": ("Kraft-Heinz",),
}
YEAR = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
AS_OF = re.compile(r"(?:\bas of\b|information available by|publicly known by|截至)\s*[:：]?\s*(\d{4}-\d{2}-\d{2})", re.I)


def scope(question, documents, as_of=None):
    cutoff_match = AS_OF.search(question)
    cutoff = date.fromisoformat(as_of or cutoff_match[1]) if as_of or cutoff_match else None
    period_text = AS_OF.sub("", question)
    years = set(map(int, YEAR.findall(period_text)))
    years.update(2000 + int(value) for value in re.findall(r"\bFY\s*['’]?(\d{2})(?!\d)", period_text, re.I))
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
    return {"companies": sorted(companies), "years": sorted(years), "as_of": cutoff.isoformat() if cutoff else None,
            "document_ids": allowed if companies or cutoff else None}


def period_score(chunk, years, documents):
    """文本所含年份优先，报告期仅作同分排序，允许晚期财报承载早期数据。"""
    content = chunk.get("content") or chunk.get("content_with_weight") or ""
    covered = set(map(int, YEAR.findall(content))) & set(years)
    report_year = documents.get(doc_name(chunk), {}).get("doc_period")
    return len(covered), int(report_year in years)


def rank(chunks, years, documents):
    return sorted(chunks, key=lambda c: period_score(c, years, documents), reverse=True) if years else list(chunks)


def needs_expansion(chunks, years):
    contents = " ".join(c.get("content") or c.get("content_with_weight") or "" for c in chunks)
    return not chunks or not set(years).issubset(map(int, YEAR.findall(contents)))


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
        chunks = rank(chunks, plan["years"], documents)
        path = "期间排序"
        if arm == "D" and needs_expansion(chunks, plan["years"]):
            chunks = rank(fetch(max(24, k * 2)), plan["years"], documents)
            path = "扩大同范围候选"
    return chunks[:k], path

