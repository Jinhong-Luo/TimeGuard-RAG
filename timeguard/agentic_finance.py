"""在原生工具会话内复用财报原文回读、表头校验与 Decimal 计算。"""
import json
import re

from .common import digest
from .financial_agent import (Selection, build_graph, Operand, present_evidence,
                              verify_operand)
from .scope import scope


async def financial_route(tools, question, documents, generator, readback, simple):
    plan = scope(question, documents)
    if len(plan["companies"]) != 1 or not plan["years"] or not tools.doc_scope:
        return None
    allowed = [d for d in plan["document_ids"] if d in tools.doc_scope]
    if not allowed:
        return None
    found = await tools.retrieve(question, doc_scope=allowed, using_embedding=True)
    normalized = generator.chunks_format(found)
    if any(c["document_id"] not in allowed for c in normalized):
        raise ValueError("计算工具收到范围以外的文档")
    raw = {c.get("chunk_id") or c["id"]: c for c in found["chunks"]}

    async def read(c):
        stored = readback(c["id"])
        if not stored or stored["doc_id"] != c["document_id"]:
            raise ValueError("回读的操作数来源不一致")
        return {**c, "content": stored["content_with_weight"]}

    if simple:
        return await direct_lookup(question, plan, normalized, raw, read)

    async def search(q, p):
        return normalized

    async def select(q, chunks):
        evidence = [{"id": c["id"], "document": c.get("document_name"), "lines": c["visible_lines"]} for c in chunks]
        prompt = ("仅返回符合以下结构的 JSON。为提问选择操作数，逐一绑定可见行、原文行名、列年份和单位。"
                  "只有公式和报表口径明确才选取操作数。资料不足时将缺项写入 missing，其余置空。百分数用 XXX 和 ones。"
                  + json.dumps(Selection.model_json_schema(), ensure_ascii=False))
        answer = await tools.chat_mdl.async_chat(prompt, [{"role":"user","content":json.dumps({"question":q,"evidence":evidence},ensure_ascii=False)}], {})
        if isinstance(answer, tuple):
            answer = answer[0]
        selection = Selection.model_validate_json(answer.split("</think>")[-1].strip())
        return selection, None, {"name":"calculate_financial_metric"}

    result = await build_graph(search, read, select, documents).ainvoke({"question":question})
    if result["status"] != "answered":
        return result["answer"], [], []
    ids = list(dict.fromkeys(f["chunk_id"] for f in result["facts"]))
    refs = [raw[i] for i in ids]
    markers = " ".join(f"[ID:{i}]" for i in range(len(ids)))
    return result["answer"] + " " + markers, refs, result["facts"]


async def direct_lookup(question, plan, normalized, raw, read):
    # 窄入口只处理明确要求 USD millions 的单财年合并报表查数。
    metrics = {"total assets", "total liabilities", "net sales", "cash and cash equivalents"}
    if len(plan["years"]) != 1 or plan["quarters"] or not re.search(r"USD millions", question, re.I):
        return None
    metric = next((m for m in metrics if re.fullmatch(
        r"What (?:is|was) [\w .’'&-]+(?:FY)?\d{4} " + m + r" \(in USD millions\)\?", question, re.I)), None)
    if metric is None:
        return None
    year = plan["years"][0]
    facts = []
    for c in present_evidence(normalized):
        if "consolidated" not in c["content"].casefold() or "USD" not in c["content"]:
            continue
        source = await read(c)
        if source["content"] != c["content"]:
            continue
        for line in c["visible_lines"]:
            cells = line["text"].rstrip(" |").split("|")
            if cells[0].strip().casefold() != metric:
                continue
            for cell in cells[1:]:
                val = cell.strip().strip("$").strip()
                if not re.fullmatch(r"-?[\d,]+(?:\.\d+)?|\([\d,]+(?:\.\d+)?\)", val):
                    continue
                op = Operand(chunk_id=c["id"],line_number=line["number"],year=year,period="FY",
                             number=val,currency="USD",scale="millions",kind="money",row_label=cells[0].strip())
                try:
                    quote = verify_operand(op, source, c["visible_lines"])
                except ValueError:
                    continue
                facts.append({**op.model_dump(), "quote":quote, "document_id":c["document_id"],
                              "company":plan["companies"][0],"content_sha256":digest(c["content"])})
    if len({f["number"] for f in facts}) != 1:
        return None
    f = facts[0]
    return f"{year} {metric}: USD {f['number']} million [ID:0]", [raw[f["chunk_id"]]], [f]

