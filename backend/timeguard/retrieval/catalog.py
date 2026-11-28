"""按来源校验披露日期，统一原生检索返回的财报名。"""
from pathlib import Path

from timeguard.common import read_json


def doc_name(chunk):
    # v0.27.3 原生检索使用 document_keyword；空值不能遮蔽后续有效字段。
    name = next((chunk[key].strip() for key in ("document_name", "document_keyword", "docnm_kwd")
                 if isinstance(chunk.get(key), str) and chunk[key].strip()), "")
    name = name.replace("\\", "/").rsplit("/", 1)[-1]
    # 本语料均为 PDF，仅去掉实际后缀，保留公司或文档标识中的其他句点。
    return (name[:-4] if name.casefold().endswith(".pdf") else name).casefold()


def catalog(ledger, state):
    dates = read_json(Path(__file__).resolve().parents[3] / "configs/filing_dates.json")
    docs = []
    for doc in ledger:
        filing = dates.get(doc["doc_name"], {})
        if filing:
            source = state["documents"][doc["doc_name"]]
            matched = (filing["pdf_sha256"] == source.get("sha256") if filing.get("pdf_sha256") else
                       bool(filing.get("accession")) and filing["accession"] in doc["doc_link"])
            if not matched:
                raise ValueError("披露日期与原始 PDF 指纹或申报登记号不匹配")
        docs.append({**doc, **filing, "id": state["documents"][doc["doc_name"]]["id"]})
    return docs
