"""把成功请求中可见的引用文本定位到同版索引，保留摘录边界。"""
import re

from timeguard.common import digest


HEADER = re.compile(r"^ID: ([1-9]\d*)\n├── Title: ([^\n]+)\n└── Content:\n", re.M)


def visible_sections(messages):
    """按终答证据区的实际编号取文本，重复或乱序编号拒绝解析。"""
    if not messages or messages[-1].get("role") != "user":
        return {}
    _, sep, evidence = messages[-1]["content"].partition("\nEvidence:\n")
    if not sep:
        return {}
    headers = list(HEADER.finditer(evidence))
    numbers = [int(h[1]) for h in headers]
    if numbers != sorted(set(numbers)):
        return {}
    sections = {}
    for i, header in enumerate(headers):
        end = headers[i + 1].start() if i + 1 < len(headers) else len(evidence)
        sections[numbers[i]] = (header[2], evidence[header.end():end].rstrip())
    return sections


def normalized(text):
    # 保留算式中的乘号，去掉检索词高亮和多余空白。
    return re.sub(r"\s+", " ", re.sub(r"(?<!\d)\*|\*(?!\d)", "", text)).strip()


def excerpt_spans(text, original):
    """区间坐标位于规范化原文；每个片段独立匹配，间隔不补写进证据。"""
    haystack = normalized(original)
    pieces = re.split(r"\n\s*\n|\.{3,}", text)
    spans = []
    for piece in pieces:
        needle = normalized(piece)
        if not needle:
            continue
        start = haystack.find(needle)
        if start < 0:
            return []
        spans.append([start, start + len(needle)])
    # 严格匹配所有碎片，短尾巴可以存在，但至少要有一个可辨识的长片段。
    return spans if any(end - start >= 12 for start, end in spans) else []


def table_prefix(text):
    """只取截断前的完整表行，让裁判看到实际存在的表头和单位。"""
    if text.startswith("<table") and "</tr>" in text:
        return text[:text.rindex("</tr>") + len("</tr>")]
    lines = text.splitlines()
    if not lines or not lines[0].lstrip(".").startswith("**Table:"):
        return ""
    while lines and (lines[-1].strip().count("|") < 2 or not lines[-1].rstrip().endswith("|")):
        lines.pop()
    return "\n".join(lines) if len(lines) >= 4 else ""


def recover(number, layer, section, complete, source, historical_hashes, rendered=None):
    packet = layer.get("packet") or []
    hashes = layer.get("block_hashes") or []
    if not section or number < 1 or number > min(len(packet), len(hashes)):
        return None
    anchor = packet[number - 1]
    block_hash = hashes[number - 1]
    title, visible = section
    if (not source or title != anchor["document_name"] or hashes.count(block_hash) != 1
            or source["doc_id"] != anchor["document_id"]):
        return None
    raw = source["content_with_weight"]
    fingerprint = digest(raw)
    if complete:
        if (complete["id"] != anchor["id"] or complete["block_sha256"] != block_hash
                or complete["document_id"] != anchor["document_id"]
                or complete["content"].strip() != visible.strip()
                or complete.get("source_sha256") != fingerprint):
            return None
        kind = "segmented"
        text = visible
    else:
        if fingerprint not in historical_hashes:
            return None
        kind = "partial_table"
        text = table_prefix(visible)
    if not text:
        return None
    for name, view in (("原文", raw), ("原生表格视图", rendered)):
        if view:
            spans = excerpt_spans(text, view)
            if spans:
                return {**anchor, "block_sha256": block_hash, "source_sha256": fingerprint,
                        "content": text, "kind": kind, "source_view": name, "spans": spans}
    return None
