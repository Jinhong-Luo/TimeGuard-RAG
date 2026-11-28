"""从许可财报的实际章节标题建页码目录，按缺项进入章节并回读原生索引块。"""
from hashlib import sha256
from pathlib import Path
import re
import subprocess
from threading import Lock

from timeguard.retrieval.filing_pages import FilingPages, pages_for, terms

ITEM = re.compile(r"^(?:item\s+\d{1,2}[a-z]?[\s.:–—-]+|note\s+\d{1,2}[\s.:–—-]+)\S", re.I)
TABLE = re.compile(r"^(?:consolidated |condensed |combined |comparative )*(?:balance sheets?|statements? of (?:cash flows?|operations|earnings|income|financial position|equity|comprehensive income))$", re.I)
ROUTES = (
    (r"receivable|liabilit|assets?|debt|inventor|cash equivalents", "balance sheets financial position"),
    (r"capital expenditure|capex|cash flow|cash paid", "statements of cash flows"),
    (r"profit|revenue|net sales|cost of sales|net income", "statements of operations income earnings"),
    (r"segment|customer|acqui[rs]|merger|litigation", "business notes to financial statements"),
    (r"reason|driver|increase|decline|margin", "management discussion and analysis"),
)


def headings(text):
    pages = text.split("\f")
    if pages and not pages[-1].strip():
        pages.pop()
    labels = {}
    for page, content in enumerate(pages, 1):
        found = []
        for line in content.splitlines():
            line = " ".join(line.split()).strip(" .")
            # 目录页罗列的印刷页码不等于 PDF 物理页码，只取正文中的标题。
            if not 5 <= len(line) <= 120 or re.search(r"\d$|\.{2,}", line):
                continue
            if ITEM.match(line) or TABLE.fullmatch(line):
                found.append(line)
        if found:
            labels[page] = " / ".join(dict.fromkeys(found))
    starts = list(labels)
    return [(labels[p], p, following-1) for p,following in zip(starts,starts[1:]+[len(pages)+1])]


def rank_sections(query, outlines):
    words = terms(query)
    hints = set()
    for pattern, names in ROUTES:
        if re.search(pattern, query, re.I):
            hints.update(terms(names))
    scored = []
    for docid, sections in outlines.items():
        for title, first, last in sections:
            tags = terms(title)
            score = 3 * len(words & tags) + len(hints & tags)
            if score:
                scored.append((-score, docid, first, last, title))
    return sorted(scored)[:4]


class OutlineSearch:
    def __init__(self, client, dataset, documents, root=Path("data/financebench")):
        self.index = FilingPages(client, dataset, documents)
        self.root = root.resolve()
        self.outlines = {}
        self.lock = Lock()

    def read_outline(self, docid):
        doc = self.index.by_id[docid]
        path = (self.root / doc["local_path"]).resolve()
        if not path.is_relative_to(self.root) or sha256(path.read_bytes()).hexdigest() != doc["pdf_sha256"]:
            raise ValueError("原 PDF 路径或指纹不匹配")
        output = subprocess.run(["pdftotext","-layout","-enc","UTF-8",str(path),"-"],
                                capture_output=True, timeout=90, check=True)
        return headings(output.stdout.decode("utf-8"))

    def search(self, query, allowed, fallback, audit):
        allowed = sorted(set(allowed))
        if not allowed or set(allowed) - self.index.by_id.keys():
            raise ValueError("目录补搜缺少合法文档")
        audit.update(目录解析次数=0, 索引分页读取次数=0, 原生补搜次数=0)
        try:
            with self.lock:
                for docid in allowed:
                    if docid not in self.outlines:
                        audit["目录解析次数"] += 1
                        self.outlines[docid] = self.read_outline(docid)
                ranked = rank_sections(query, {d:self.outlines[d] for d in allowed})
                audit["章节范围"] = [list(row[1:]) for row in ranked]
                selected = {}
                for _, docid, first, last, _ in ranked:
                    if docid not in self.index.cache:
                        audit["索引分页读取次数"] += self.index.load(docid)
                    for c in self.index.cache[docid]:
                        if any(p and first <= p[0] <= last for p in c["positions"]):
                            selected[docid,c["id"]] = c
            rows, _ = pages_for(query, list(selected.values()), count=7)
            if rows:
                audit["路径"] = "目录内找页"
                return rows[:20]
            audit["回退原因"] = "无匹配章节或章节内无匹配原块"
        except (OSError, subprocess.SubprocessError) as exc:
            audit["回退原因"] = type(exc).__name__
        audit.update(路径="原生补搜", 原生补搜次数=1)
        return fallback(query)
