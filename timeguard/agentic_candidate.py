"""固定版 Agentic 财报候选。钩子只安装在一次实验的独立工作进程。"""
import asyncio
from dataclasses import replace
from functools import wraps
import time

from .common import digest
from .scope import rank, scope

VARIANTS = ("evidence", "concise", "fallback", "temporal", "sca1", "turn2")
RULES = "使用问题的语言。只给目标结论、必要数值和公式。逐项引用所用报表，区分期间、单位及合并口径。缺少操作数时指出缺项。"


def evidence_blocks(chunks, render, count, budget):
    blocks, kept, seen = [], [], set()
    for chunk in chunks:
        key = (chunk.get("doc_id"), chunk.get("chunk_id") or chunk.get("id"))
        if not all(key) or key in seen:
            continue
        block = render(chunk, len(kept))
        if not block:
            continue
        size = count(block)
        if size > budget:
            continue
        budget -= size
        kept.append(chunk)
        blocks.append(block)
        seen.add(key)
    return kept, blocks


def install(service, generator, settings, spec, current, event):
    from rag.advanced_rag import agentic_rag_graph as graph
    from rag.advanced_rag.harness.config import THINKING_MODES
    from rag.advanced_rag.harness.stats import LLMUsageStats
    from rag.nlp.search import index_name

    variant = spec["candidate"]
    documents = spec.get("documents", [])
    by_name = {d["doc_name"].casefold(): d for d in documents}
    if variant in ("sca1", "turn2"):
        overrides = {"sca_max_rounds": 1} if variant == "sca1" else {"action_max_turns": 2}
        THINKING_MODES["medium"] = replace(THINKING_MODES["medium"], **overrides)

    original_stats = LLMUsageStats.log
    def log_stats(self, *args, **kwargs):
        event("phase_usage", phases=self.snapshot())
        return original_stats(self, *args, **kwargs)
    LLMUsageStats.log = log_stats

    class TracedGraph(graph.StateGraph):
        def add_node(self, name, action, **kwargs):
            @wraps(action)
            async def node(state):
                started = time.monotonic()
                event("node_start", node=name, round=state.get("search_rounds"),
                      chunks=len((state.get("kbinfos") or {}).get("chunks", [])))
                try:
                    result = await action(state)
                except BaseException as exc:
                    event("node_error", node=name, reason=type(exc).__name__)
                    raise
                event("node_exit", node=name, elapsed=time.monotonic()-started,
                      outcome={k: result[k] for k in ("no_progress", "empty_result", "abstain", "search_rounds") if k in result},
                      verdict=(result.get("verdict") or {}).get("status"))
                return result
            return super().add_node(name, node, **kwargs)
    graph.StateGraph = TracedGraph

    if variant == "temporal":
        original_filter = service.apply_meta_data_filter
        @wraps(original_filter)
        async def filtered(*args, **kwargs):
            selected = await original_filter(*args, **kwargs)
            allowed = current.get()["plan"]["document_ids"]
            value = [d for d in allowed if selected is None or d in selected] if allowed is not None else selected
            current.get()["allowed"] = value
            return value or ["-999"]
        service.apply_meta_data_filter = filtered
        original_retrieve = settings.retriever.retrieval
        @wraps(original_retrieve)
        async def retrieve(*args, **kwargs):
            s = current.get()
            allowed = s.get("allowed")
            if allowed is not None:
                restricted = kwargs.get("doc_ids")
                ids = [d for d in allowed if restricted is None or d in restricted]
                if not ids:
                    return {"chunks": [], "doc_aggs": [], "total": 0}
                kwargs["doc_ids"] = ids
            value = await original_retrieve(*args, **kwargs)
            if allowed is not None and any(c["doc_id"] not in allowed for c in value["chunks"]):
                raise ValueError("原生检索越界")
            value["chunks"] = rank(value["chunks"], s["plan"]["years"], by_name, s["plan"]["quarters"])
            return value
        settings.retriever.retrieval = retrieve

    original_compose = graph._compose_answer_from_evidence

    async def compose(state, tools, queue, conf):
        s = current.get()
        info = state.get("kbinfos") or {"chunks": [], "doc_aggs": []}
        if (variant == "fallback" and not info["chunks"] and tools.doc_scope and tools.doc_scope != ["-999"]
                and not s.get("fallback") and not any(e["kind"] == "retrieval" for e in s["events"])):
            s["fallback"] = True
            ids = list(tools.doc_scope)
            info = await tools.retrieve(s["question"], doc_scope=ids, using_embedding=True)
            if any(c["doc_id"] not in ids for c in info["chunks"]):
                raise ValueError("补查返回范围以外的文档")
            event("fallback", chunks=len(info["chunks"]), doc_ids=ids)
            state = {**state, "kbinfos": info, "abstain": not info["chunks"], "empty_result": not info["chunks"]}
        if variant == "concise":
            original = tools.system_prompt
            tools.system_prompt = (original or "") + "\n" + RULES
            try:
                return await original_compose(state, tools, queue, conf)
            finally:
                tools.system_prompt = original
        if variant == "temporal":
            allowed = s.get("allowed")
            if allowed is not None and any(c.get("doc_id") not in allowed for c in info["chunks"]):
                raise ValueError("最终证据越界")
        if variant != "evidence":
            result = await original_compose(state, tools, queue, conf)
            return {**result, "kbinfos": info}

        # 保留原生槽位、表格与相关度筛选，只替换该节点内部的编号与正文核验。
        s["compose_tools"] = tools
        s["visible"] = {}
        try:
            result = await original_compose(state, tools, queue, conf)
            return {**result, "kbinfos": tools.kbinfos}
        finally:
            s.pop("compose_tools", None)
    graph._compose_answer_from_evidence = compose
    if variant != "evidence":
        return
    original_prompt = graph.kb_prompt
    def prompt(info, budget):
        s = current.get()
        tools = s.get("compose_tools")
        if tools is None:
            return original_prompt(info, budget)
        confirmed = []
        for c in info["chunks"]:
            ident = c.get("chunk_id") or c.get("id")
            if not ident or not c.get("doc_id") or (tools.doc_scope is not None and c["doc_id"] not in tools.doc_scope):
                continue
            raw = settings.docStoreConn.get(ident, index_name(tools.tenant_ids[0]), tools.kb_ids)
            if (raw and raw.get("doc_id") == c["doc_id"] and raw.get("content_with_weight")
                    and raw["content_with_weight"] == (c.get("content") or c.get("content_with_weight"))):
                confirmed.append(c)
        kept, blocks = evidence_blocks(confirmed, generator._kb_block, generator.num_tokens_from_string, int(.97*budget))
        tools.kbinfos = {**info, "chunks": kept}
        tools._rag_cite_chunk_ids = [c.get("chunk_id") or c["id"] for c in kept]
        s["evidence_blocks"] = list(zip(blocks, generator.chunks_format(tools.kbinfos)))
        return blocks
    graph.kb_prompt = prompt

    fit = graph.message_fit_in
    def capture(messages, budget):
        count, fitted = fit(messages, budget)
        s = current.get()
        if s and s.get("compose_tools") is not None:
            user = fitted[-1]["content"]
            for i, (block, c) in enumerate(s.pop("evidence_blocks", [])):
                if block in user:
                    s["visible"][str(i)] = {"chunk_id": c["id"], "document_id": c["document_id"], "content_sha256": digest(c["content"])}
        return count, fitted
    graph.message_fit_in = capture

