"""在固定 RAGFlow 容器中用原版函数渲染配对方案，不访问裁判或金标。"""
import asyncio
import json
import sys
from pathlib import Path

from .common import digest
from .evidence_layers import location, match_blocks, visible_blocks
from .final_snapshot import replay
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
    result = {}
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
            if arm in ("page_select", "budget"):
                chunks = select(item["question"],value["state"]["kbinfos"]["chunks"],value["tools"]["doc_scope"])
            elif arm != "original":
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
        graph.message_fit_in = fit_message
        graph.kb_prompt = prompt
        try:
            await replay(snapshot, graph._compose_answer_from_evidence, Model())
        finally:
            graph.kb_prompt = original
            graph.message_fit_in = fit
        mapped = record.pop("mapped")
        record["packet"] = [location(c) for c,b in mapped]
        record["visible"] = visible_blocks(mapped,record["body"]["messages"])
        if arm == "original" and (record["body"] != item["body"] or record["selected"] != item["selected"] or
                                  record["block_hashes"] != item["block_hashes"]):
            raise ValueError("配对驱动与真实原版不一致")
        result[arm] = record
    return result


async def main():
    for item in json.load(sys.stdin):
        print("TIMEGUARD_RENDER " + json.dumps(await render(item),ensure_ascii=False),flush=True)


if __name__ == "__main__":
    asyncio.run(main())
