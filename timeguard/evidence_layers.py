"""在原生组合链路旁记录逐次检索、最终证据包和已成功送达的完整原文。"""
from contextvars import ContextVar
from functools import wraps
import json

from .common import digest


def location(chunk):
    return {k: chunk.get(k) for k in ("id", "document_id", "document_name", "positions")}


def match_blocks(chunks, blocks, render):
    """按上游实际渲染顺序映射编号，遇到不一致就保留未知。"""
    result = []
    for chunk in chunks:
        if len(result) == len(blocks):
            break
        index = len(result)
        block = render(chunk, index + 1)
        if block is None:
            continue
        if block != blocks[index]:
            return None
        result.append((chunk, block))
    return result if len(result) == len(blocks) else None


def visible_blocks(mapped, messages):
    # 逐条核对用户侧的独立证据区，排除研究摘要、历史消息和被裁掉的半块。
    if not messages or messages[-1].get("role") != "user":
        return None
    content = messages[-1].get("content")
    if not isinstance(content, str):
        return None
    _, sep, evidence = content.partition("\nEvidence:\n")
    if mapped is None:
        return None
    return [{**location(chunk), "content": chunk.get("content") or chunk.get("content_with_weight"),
             "block_sha256": digest(block)} for chunk, block in mapped if sep and block in evidence]


def install(generator, settings, current, tools_class):
    import httpx
    from rag.advanced_rag import agentic_rag_graph as graph
    from rag.advanced_rag.harness.tools import search
    from rag.nlp.search import index_name

    composing = ContextVar("本次最终证据包", default=None)
    node_context = ContextVar("检索所在图节点", default=(None, None))
    class ObservedGraph(graph.StateGraph):
        def add_node(self, name, action, **kwargs):
            @wraps(action)
            async def node(state):
                token = node_context.set((name, state.get("search_rounds")))
                try:
                    return await action(state)
                finally:
                    node_context.reset(token)
            return super().add_node(name, node, **kwargs)
    graph.StateGraph = ObservedGraph
    def observe(owner, channel):
        original = getattr(owner, channel)
        @wraps(original)
        async def retrieve(*args, **kwargs):
            state = current.get()
            if state is None:
                return await original(*args, **kwargs)
            rounds = state["evidence_layers"]["retrievals"]
            name, number = node_context.get()
            entry = {"call": len(rounds) + 1, "channel": channel, "node": name, "search_round": number, "chunks": None}
            rounds.append(entry)
            value = await original(*args, **kwargs)
            entry["chunks"] = [{**location(c), "content_sha256": digest(c["content"])}
                               for c in generator.chunks_format(value)]
            return value
        setattr(owner, channel, retrieve)
    observe(settings.retriever, "retrieval")
    observe(tools_class, "fetch_full_document")
    for channel in ("bm25_search", "hybrid_search", "grep_search", "vector_search", "metadata_search", "list_chunks"):
        observe(search, channel)

    original_compose = graph._compose_answer_from_evidence
    @wraps(original_compose)
    async def compose(state, tools, *args, **kwargs):
        record = {"selected": None, "packet": None, "requests": []}
        current.get()["evidence_layers"]["answers"].append(record)
        slot = {"record": record, "mapped": None}
        token = composing.set(slot)
        try:
            result = await original_compose(state, tools, *args, **kwargs)
            if record["selected"] is None and tools.empty_response and result.get("final_answer") == tools.empty_response:
                record.update(selected=[], packet=[], empty_response=True)
            # 在回答完成后按索引原块核验。回读只进入诊断记录。
            for req in record["requests"]:
                for c in req["complete_chunks"] or []:
                    c["source_check"] = "unknown"
                    try:
                        raw = settings.docStoreConn.get(c["id"], index_name(tools.tenant_ids[0]), tools.kb_ids)
                        if raw and raw.get("doc_id") == c["document_id"] and raw.get("content_with_weight"):
                            source = raw["content_with_weight"]
                            c["source_sha256"] = digest(source)
                            c["source_check"] = "intact" if source == c["content"] else "edited"
                    except Exception as exc:
                        c["readback_error"] = type(exc).__name__
            return result
        finally:
            composing.reset(token)
    graph._compose_answer_from_evidence = compose

    original_prompt = graph.kb_prompt
    @wraps(original_prompt)
    def prompt(info, *args, **kwargs):
        blocks = original_prompt(info, *args, **kwargs)
        slot = composing.get()
        if slot is not None:
            chunks = generator.chunks_format(info)
            slot["record"]["selected"] = [location(c) for c in chunks]
            mapped = match_blocks(info["chunks"], blocks, generator._kb_block)
            if mapped is not None:
                mapped = [(generator.chunks_format({"chunks": [c]})[0], b) for c, b in mapped]
                slot["record"]["packet"] = [location(c) for c, _ in mapped]
            slot["mapped"] = mapped
        return blocks
    graph.kb_prompt = prompt

    original_send = httpx.AsyncClient.send
    @wraps(original_send)
    async def send(client, request, *args, **kwargs):
        slot = composing.get()
        if slot is None or request.url.port != 18866 or not request.url.path.endswith("/chat/completions"):
            return await original_send(client, request, *args, **kwargs)
        entry = {"status": None, "complete_chunks": None}
        slot["record"]["requests"].append(entry)
        response = await original_send(client, request, *args, **kwargs)
        entry["status"] = response.status_code
        if response.status_code == 200:
            # 专用模型桥在生成完成后才发送成功响应。只保存业务原文，不记录请求头。
            entry["complete_chunks"] = visible_blocks(slot["mapped"], json.loads(request.content)["messages"])
            entry["model"] = json.loads(request.content).get("model")
        return response
    httpx.AsyncClient.send = send
