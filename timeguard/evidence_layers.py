"""在原生组合链路旁记录逐次检索、最终证据包和已成功送达的完整原文。"""
from contextvars import ContextVar
from functools import wraps
import json
import time

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


def install(generator, settings, current, tools_class, snapshot_spec=None):
    import httpx
    from rag.advanced_rag import agentic_rag_graph as graph
    from rag.advanced_rag.harness.tools import search
    from rag.nlp.search import index_name

    composing = ContextVar("本次最终证据包", default=None)
    wide=bool(snapshot_spec and snapshot_spec["candidate"]=="wide")
    if wide:
        original_fit=graph.message_fit_in
        def fit(messages,limit):
            slot=composing.get()
            return original_fit(messages,min(12000,slot["tools"].chat_mdl.max_length) if slot else limit)
        graph.message_fit_in=fit
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
            if state.get("allowed") is not None and any(c["document_id"] not in state["allowed"] for c in entry["chunks"]):
                raise ValueError("检索材料超出允许的报告范围")
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
        if snapshot_spec and snapshot_spec["candidate"] in ("page", "supplement"):
            variant = snapshot_spec["candidate"]
            found = current.get()["page_search"]
            allowed = set(tools.doc_scope or ()) & set(found["allowed"])
            if set(tools.doc_scope or ()) - set(found["allowed"]):
                raise ValueError("研究范围超出公司或披露截止日")
            chosen = [c for c in found[variant] if c["document_id"] in allowed]
            native = [{**c, "chunk_id":c["id"], "doc_id":c["document_id"], "docnm_kwd":c["document_name"],
                       "content_with_weight":c["content"]} for c in chosen]
            state = {**state,"kbinfos":{**(state.get("kbinfos") or {}),"chunks":native}}
            if chosen:
                state.update(empty_result=False)
            record["retrieval_policy"] = variant
            events = current.get()["evidence_layers"]["retrievals"]
            events.append({"call":len(events)+1,"channel":variant,"node":"最终页级找页","search_round":None,
                           "chunks":[{**location(c),"content_sha256":digest(c["content"])} for c in chosen]})
        slot = {"record": record, "mapped": None}
        if snapshot_spec is not None:
            from .final_snapshot import freeze
            config = kwargs["answer_conf"] if "answer_conf" in kwargs else args[1]
            identity = {k: snapshot_spec[k] for k in ("dataset_id", "model", "arm_config", "upstream_files")}
            slot["tools"] = tools
            record["snapshot"] = freeze(state, tools, config, identity)
            if snapshot_spec["candidate"]=="route":
                import asyncio
                import os
                from .finance_route import needs_review
                from .evidence_review import catalog,inspect_once
                trace=current.get()
                chunks=record["snapshot"]["input"]["state"]["kbinfos"]["chunks"]
                reason=needs_review(trace["question"],chunks)
                if reason and "review_attempt" not in trace:
                    trace["review_attempt"]={"reason":reason}
                    result=await asyncio.to_thread(inspect_once,{"base_url":os.environ["TIMEGUARD_REVIEW_URL"],
                        "token":os.environ["TIMEGUARD_REVIEW_TOKEN"]},trace["question"],record["snapshot"])
                    trace["review_attempt"].update(result)
                    trace["route_decision"]="审查回退" if result["fallback"] else "一次审查"
                    if not result["fallback"]:
                        entries=catalog(record["snapshot"])
                        slot["review_chunks"]=[entries[key][0] for key in result["ids"]]
                else:
                    trace.setdefault("route_decision","原生回答")
            record["pool_locations"] = [location(c) for c in generator.chunks_format(state.get("kbinfos") or {"chunks": []})]
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
            if snapshot_spec is not None:
                from .final_snapshot import replay, RequestCaptured
                check = {"selected": None, "packet": None, "requests": []}
                slot.update(record=check, mapped=None, replay=True)
                try:
                    await replay(record["snapshot"], original_compose, tools.chat_mdl)
                except RequestCaptured:
                    pass
                if record.get("empty_response"):
                    check.update(selected=[], packet=[])
                record["replay"] = {
                    "selected_equal": record["selected"] == check["selected"],
                    "packet_equal": record.get("block_hashes") == check.get("block_hashes"),
                    "request_equal": bool(record.get("empty_response") and not check["requests"]) or
                        any(r.get("status") == 200 and r.get("body") == check.get("body") for r in record["requests"]),
                }
            return result
        finally:
            composing.reset(token)
    if snapshot_spec and snapshot_spec["candidate"]=="brief":
        from .final_snapshot import BRIEF_RULE
        @wraps(original_compose)
        async def brief(state,tools,*args,**kwargs):
            previous = tools.system_prompt
            tools.system_prompt = (previous or "") + "\n" + BRIEF_RULE
            try:
                return await compose(state,tools,*args,**kwargs)
            finally:
                tools.system_prompt = previous
        graph._compose_answer_from_evidence = brief
    else:
        graph._compose_answer_from_evidence = compose

    original_prompt = graph.kb_prompt
    @wraps(original_prompt)
    def prompt(info, *args, **kwargs):
        slot = composing.get()
        if slot is not None and snapshot_spec and snapshot_spec["candidate"] == "page_select":
            from .page_select import select
            source = current.get()["evidence_layers"]["answers"][-1]["snapshot"]["input"]
            selected = select(current.get()["question"], source["state"]["kbinfos"]["chunks"], source["tools"]["doc_scope"])
            info = {**info, "chunks": selected}
            slot["tools"]._rag_cite_chunk_ids = [str(c.get("chunk_id") or c.get("id") or "") for c in selected]
        if slot is not None and snapshot_spec and snapshot_spec["candidate"] in ("page", "supplement"):
            chosen = current.get()["evidence_layers"]["answers"][-1]["snapshot"]["input"]["state"]["kbinfos"]["chunks"]
            info = {**info,"chunks":chosen}
            slot["tools"]._rag_cite_chunk_ids = [str(c.get("chunk_id") or c.get("id") or "") for c in chosen]
        if slot is not None and slot.get("review_chunks") is not None:
            chunks=slot["review_chunks"]
            info={**info,"chunks":chunks}
            slot["tools"]._rag_cite_chunk_ids=[str(c.get("chunk_id") or c.get("id") or "") for c in chunks]
        if wide and slot is not None:
            args=(min(12000,slot["tools"].chat_mdl.max_length),*args[1:])
        blocks = original_prompt(info, *args, **kwargs)
        if slot is not None:
            if snapshot_spec is not None:
                slot["record"]["block_hashes"] = [digest(b) for b in blocks]
            chunks = generator.chunks_format(info)
            slot["record"]["selected"] = [location(c) for c in chunks]
            mapped = match_blocks(info["chunks"], blocks, generator._kb_block)
            if mapped is not None:
                mapped = [(generator.chunks_format({"chunks": [c]})[0], b) for c, b in mapped]
                slot["record"]["packet"] = [location(c) for c, _ in mapped]
            slot["mapped"] = mapped
        return blocks
    graph.kb_prompt = prompt

    if snapshot_spec is not None:
        from rag.advanced_rag.harness.stats import _CURRENT_PHASE
        from .agentic_candidate import bridge_event
        transport = httpx.AsyncClient.send
        @wraps(transport)
        async def audited(client, request, *args, **kwargs):
            state = current.get()
            if state is None or request.url.port != 18866 or not request.url.path.endswith("/chat/completions"):
                return await transport(client, request, *args, **kwargs)
            started = time.monotonic()
            node,number = node_context.get()
            event = {"kind":"bridge_request","node":node,"search_round":number,
                     "phase":_CURRENT_PHASE.get(),"usage":None}
            state["events"].append(event)
            try:
                response = await transport(client,request,*args,**kwargs)
                await response.aread()
                event["http_status"] = response.status_code
                try:
                    event.update(bridge_event(response))
                except (ValueError,KeyError,TypeError):
                    event["usage_error"] = "模型桥响应缺少可解析的用量"
                return response
            except httpx.HTTPError as exc:
                event["error"] = type(exc).__name__
                raise
            finally:
                event["seconds"] = time.monotonic()-started
        httpx.AsyncClient.send = audited

    original_send = httpx.AsyncClient.send
    @wraps(original_send)
    async def send(client, request, *args, **kwargs):
        slot = composing.get()
        if slot is None or request.url.port != 18866 or not request.url.path.endswith("/chat/completions"):
            return await original_send(client, request, *args, **kwargs)
        entry = {"status": None, "complete_chunks": None}
        slot["record"]["requests"].append(entry)
        if snapshot_spec is not None:
            payload = json.loads(request.content)
            if slot.get("replay"):
                from .final_snapshot import RequestCaptured
                slot["record"]["body"] = payload
                raise RequestCaptured()
            entry["body"] = payload
        response = await original_send(client, request, *args, **kwargs)
        entry["status"] = response.status_code
        if response.status_code == 200:
            # 专用模型桥在生成完成后才发送成功响应。只保存业务原文，不记录请求头。
            entry["complete_chunks"] = visible_blocks(slot["mapped"], json.loads(request.content)["messages"])
            entry["model"] = json.loads(request.content).get("model")
        return response
    httpx.AsyncClient.send = send
