"""在固定 RAGFlow 容器中用原版函数渲染配对方案，不访问裁判或金标。"""
import asyncio
from copy import deepcopy
import json
import sys
from pathlib import Path

from .common import digest
from .evidence_layers import location, match_blocks, visible_blocks
from .final_snapshot import BRIEF_RULE, replay
from .page_select import select
from .message_budget import MARKER, pack


async def render(item):
    from rag.advanced_rag import agentic_rag_graph as graph
    from rag.prompts import generator
    snapshot = item["snapshot"]
    value = snapshot["input"]
    for path, sha in value["identity"]["upstream_files"].items():
        if digest(Path("/ragflow", path).read_bytes()) != sha:
            raise ValueError("容器与快照上游版本不同")
    from rag.advanced_rag import agentic_rag
    original_budget = agentic_rag._EVIDENCE_BUDGET_TOKENS
    result = {}
    from .final_snapshot import fingerprint
    if fingerprint(value)!=snapshot["sha256"]:
        raise ValueError("基础快照被修改")
    original = graph.kb_prompt
    fit = graph.message_fit_in
    for arm in item["arms"]:
        record = {"requests": [], "snapshot": {"sha256": snapshot["sha256"]}}
        class Model:
            max_length = value["max_length"]
            async def async_chat_streamly_delta(self, system, history, conf):
                record["body"] = {**item["body"], "messages": [{"role": "system", "content": system}, *history]}
                yield ""
        def prompt(info, budget):
            chunks = info["chunks"]
            if "fours" in item:
                chunks = item["fours"][arm]
                valid_size = 1 <= len(chunks) <= 12 if item.get("adaptive") else len(chunks)==4
                if not valid_size or any(c["document_id"] not in value["tools"]["doc_scope"] for c in chunks):
                    raise ValueError("四块对照超出范围")
            elif arm in ("page_select", "budget"):
                chunks = select(item["question"],value["state"]["kbinfos"]["chunks"],value["tools"]["doc_scope"])
            elif arm in ("page","supplement"):
                chunks = item["page_result"][arm]
                allowed = set(item["page_result"]["allowed"])
                if set(value["tools"]["doc_scope"] or []) - allowed:
                    raise ValueError("原生研究摘要范围大于本题允许范围")
                if any(c["document_id"] not in allowed for c in chunks):
                    raise ValueError("最终引用越界")
            elif arm == "history_read":
                extra = item["history"]
                if any(c["document_id"] not in value["tools"]["doc_scope"] for c in extra):
                    raise ValueError("历史块越界")
                if extra:
                    chunks = [*extra, *chunks][:12]
            elif arm == "review":
                if not item["review"]["fallback"]:
                    from .evidence_review import catalog
                    entries = catalog(snapshot)
                    chunks = [entries[key][0] for key in item["review"]["ids"]]
            elif arm not in ("original","formula","brief","wide"):
                raise ValueError("未知方案")
            info = {**info,"chunks": chunks}
            if arm == "budget":
                record["to_pack"] = chunks
                record["selected"] = [location(c) for c in generator.chunks_format(info)]
                return [MARKER]
            blocks = original(info,budget)
            mapped = match_blocks(chunks,blocks,generator._kb_block)
            if mapped is None:
                raise ValueError("证据渲染编号无法对齐")
            record["selected"] = [location(c) for c in generator.chunks_format(info)]
            record["mapped"] = [(generator.chunks_format({"chunks":[c]})[0], b) for c,b in mapped]
            record["block_hashes"] = [digest(b) for b in blocks]
            return blocks
        def fit_message(messages,limit):
            if arm == "budget":
                mapped,messages = pack(messages,limit,record.pop("to_pack"),item["question"],generator._kb_block,
                                       generator.num_tokens_from_string)
                record["mapped"] = [(generator.chunks_format({"chunks":[c]})[0],b) for c,b in mapped]
                record["block_hashes"] = [digest(b) for c,b in mapped]
            return fit(messages,limit)
        if arm=="wide":
            agentic_rag._EVIDENCE_BUDGET_TOKENS = 12000
        graph.message_fit_in = fit_message
        graph.kb_prompt = prompt
        try:
            active = snapshot
            if "fours" in item:
                chosen = item["fours"][arm]
                state = {"question":item["question"], "kbinfos":{"chunks":chosen}}
                changed = {**value,"state":state}
                active = {"input":changed,"sha256":fingerprint(changed)}
            if arm in ("formula","brief"):
                extension = (item["guard"]["instruction"] if arm=="formula" else
                             BRIEF_RULE)
                if extension:
                    cfg = {**value["tools"],"system_prompt":(value["tools"].get("system_prompt") or "")+"\n"+extension}
                    changed = {**value,"tools":cfg}
                    active = {"input":changed,"sha256":fingerprint(changed)}
            await replay(active, graph._compose_answer_from_evidence, Model())
        finally:
            graph.kb_prompt = original
            graph.message_fit_in = fit
            agentic_rag._EVIDENCE_BUDGET_TOKENS = original_budget
        mapped = record.pop("mapped")
        record["packet"] = [location(c) for c,b in mapped]
        record["visible"] = visible_blocks(mapped,record["body"]["messages"])
        if arm == "original" and (record["body"] != item["body"] or record["selected"] != item["selected"] or
                                  record["block_hashes"] != item["block_hashes"]):
            raise ValueError("配对驱动与真实原版不一致")
        if arm=="formula":
            ids = {c["id"] for c in record["visible"]}
            if any(f["chunk_id"] not in ids for f in item["guard"].get("facts",[])):
                raise ValueError("校验操作数所在块未完整送达")
        if arm == "history_read":
            def identities(chunks):
                return {(c["document_id"], c["id"], digest(c["content"])) for c in chunks}
            old = identities(result["original"]["visible"])
            received = identities(record["visible"])
            lost = old - received
            if not identities(item["history"]).issubset(received) or (lost and not item["history_swap"]):
                record = deepcopy(result["original"])
                record["history_fallback"] = "新增或原有完整块无法在原预算内送达"
            else:
                record["history_displaced"] = len(lost)
        result[arm] = record
    return result


async def main():
    for item in json.load(sys.stdin):
        print("TIMEGUARD_RENDER " + json.dumps(await render(item),ensure_ascii=False),flush=True)


if __name__ == "__main__":
    asyncio.run(main())
