"""在单独进程中调用固定版 RAGFlow 原生问答，记录真正送入模型的知识块。"""
import asyncio
import contextvars
import json
import os
from pathlib import Path
import sys
import time

from .common import digest
from .scope import needs_expansion, rank, scope

STATE = contextvars.ContextVar("当前独立题目")
FINGERPRINTS = {
    "api/db/services/dialog_service.py": "38a875d4167248f9858f12d7f00feef333072e449de8b9da73cfb804db00cdbb",
    "rag/prompts/generator.py": "da517c60f994f67200736b21f5f434b6e0aac5164289f26ad517cb07847e76f6",
}
FINANCE_PROMPT = """请用问题所用的语言简洁回答，仅使用以下财报证据。核对公司、期间、单位、币种及指标口径。
计算时列出原始数值、各自引用、公式和结果。证据不足时写明本次资料缺少什么。每个关键事实引用对应块。
{knowledge}"""


def install_hooks(service, generator, retriever):
    """所有钩子仅存在于本次批处理进程。检索、模板渲染、模型调用和补引用仍由原生链路执行。"""
    retrieve = retriever.retrieval
    async def scoped_retrieval(*args, **kwargs):
        s = STATE.get()
        result = await retrieve(*args, **kwargs)
        if s["arm"] in ("C", "D"):
            years = s["plan"]["years"]
            quarters = s["plan"]["quarters"]
            if s["arm"] == "D" and needs_expansion(result["chunks"], years, quarters):
                expanded = list(args)
                expanded[5] = max(24, args[5] * 2)
                result = await retrieve(*expanded, **kwargs)
                s["fallback"] = True
            result["chunks"] = rank(result["chunks"], years, s["documents"], quarters)[:args[5]]
        allowed = kwargs.get("doc_ids")
        if allowed and any(c["doc_id"] not in allowed for c in result["chunks"]):
            raise ValueError("原生问答检索跨越限定范围")
        s["retrieved"] = generator.chunks_format(result)
        return result

    def kb_prompt(info, max_tokens):
        s = STATE.get()
        blocks, used = [], 0
        for i, chunk in enumerate(info["chunks"]):
            block = generator._kb_block(chunk, i)
            if block is None:
                continue
            used += generator.num_tokens_from_string(block)
            if used > max_tokens * .97:
                break
            blocks.append(block)
            s["blocks"].append((i, block, generator.chunks_format({"chunks": [chunk]})[0]))
        return blocks

    native_chat = service.LLMBundle.async_chat
    async def traced_chat(self, system, history, *args, **kwargs):
        s = STATE.get()
        s["prompt_tokens_estimate"] = generator.num_tokens_from_string(system)
        for i, block, c in s["blocks"]:
            if block in system:
                s["visible"][str(i)] = {"chunk_id": c["id"], "document_id": c["document_id"],
                                        "content_sha256": digest(c["content"])}
        answer = await native_chat(self, system, history, *args, **kwargs)
        s["raw_answer"] = answer
        return answer

    insert = retriever.insert_citations
    def traced_insert(*args, **kwargs):
        STATE.get()["origin"] = "native_auto"
        return insert(*args, **kwargs)

    retriever.retrieval = scoped_retrieval
    retriever.insert_citations = traced_insert
    service.kb_prompt = kb_prompt
    service.LLMBundle.async_chat = traced_chat


async def run_batch(spec):
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    for path, expected in FINGERPRINTS.items():
        if digest(Path("/ragflow", path).read_bytes()) != expected:
            raise ValueError("固定版上游源码不匹配：" + path)
    from api.apps import app
    from common import settings
    from api.db.services import dialog_service as service
    from rag.prompts import generator
    settings.init_settings()
    install_hooks(service, generator, settings.retriever)
    docs = {d["doc_name"].casefold(): d for d in spec["documents"]}
    async def one(q):
        started = time.monotonic()
        plan = scope(q["question"], spec["documents"], q.get("as_of"))
        s = {"blocks": [], "visible": {}, "origin": "model", "arm": spec["arm"], "plan": plan,
             "documents": docs, "fallback": False, "retrieved": []}
        token = STATE.set(s)
        row = {**q, "scope": plan, "status": "ok", "usage": None, "cost_usd": None}
        try:
            found, dialog = service.DialogService.get_by_id(spec["chat_id"])
            if not found or dialog.llm_id != spec["model"] or dialog.kb_ids != [spec["dataset_id"]]:
                raise ValueError("原生对话与模型或索引清单不匹配")
            dialog.top_n = spec["k"]
            dialog.vector_similarity_weight = spec["weight"]
            if spec["finance_prompt"]:
                dialog.prompt_config = {**dialog.prompt_config, "system": FINANCE_PROMPT}
            allowed = plan["document_ids"] if spec["arm"] != "A" else None
            if allowed == []:
                row.update(answer="指定历史日期内没有可核验的已发布报告。", reference={"chunks": []})
            else:
                msg = {"role": "user", "content": q["question"]}
                if allowed is not None:
                    msg["doc_ids"] = allowed
                async with app.app_context():
                    async for result in service.async_chat(dialog, [msg], stream=False):
                        ref = result.get("reference") or {}
                        row.update(answer=result["answer"], usage=result.get("usage"),
                                   reference={"chunks": generator.chunks_format(ref)})
                if not row.get("answer") or "**ERROR**" in row["answer"]:
                    raise ValueError("原生问答返回错误或空答案")
        except Exception as exc:
            row.update(status="request_error", error=f"{type(exc).__name__}: {exc}"[:500])
        finally:
            STATE.reset(token)
        row.update(citation_trace={"version": 1, "visible": s["visible"], "origin": s["origin"],
                                   "raw_answer": s.get("raw_answer")},
                   retrieved_chunks=s["retrieved"], fallback=s["fallback"],
                   prompt_tokens_estimate=s.get("prompt_tokens_estimate"), elapsed_seconds=time.monotonic()-started)
        print("TIMEGUARD_RESULT " + json.dumps(row, ensure_ascii=False), flush=True)
        return row["status"] == "ok"
    for start in range(0, len(spec["queries"]), 2):
        if not all(await asyncio.gather(*(one(q) for q in spec["queries"][start:start + 2]))):
            return False
    return True


if __name__ == "__main__":
    raise SystemExit(0 if asyncio.run(run_batch(json.load(sys.stdin))) else 1)

