"""在固定版工作容器向最终节点注入空证据，保留真实索引、范围与原生检索。"""
import asyncio
from contextvars import ContextVar
import json
import sys
from .agentic_candidate import install


async def run(spec):
    from api.apps import app
    from api.db.services import dialog_service as service
    from common import settings
    from rag.prompts import generator
    from rag.advanced_rag import agentic_rag_graph as graph
    settings.init_settings()
    current = ContextVar("回归题")
    def event(kind, **kwargs):
        current.get()["events"].append({"kind":kind, **kwargs})

    async def end(state, tools, queue, conf):
        return {"final_answer":"已记录检索结果"}
    graph._compose_answer_from_evidence = end
    install(service, generator, settings, {"candidate":"fallback"}, current, event)
    rows=[]
    async with app.app_context():
        ok, dialog = service.DialogService.get_by_id(spec["chat_id"])
        if not ok or dialog.kb_ids != [spec["dataset_id"]]:
            raise ValueError("原生对话与索引不匹配")
        kbs, embed, _, model, _ = service.get_models(dialog)
        for case in spec["cases"]:
            ids=case["allowed"]
            if not ids or ids==["-999"]:
                raise ValueError("目标回归缺少有效范围")
            tools = service.RAGTools(list({kb.tenant_id for kb in kbs}),model,embed_mdl=embed,kb_ids=dialog.kb_ids,
                doc_scope=ids,thinking_mode="medium",similarity_threshold=dialog.similarity_threshold,
                vector_similarity_weight=dialog.vector_similarity_weight,top_n=dialog.top_n)
            state={"question":case["question"],"events":[]}
            current.set(state)
            empty={"question":case["question"],"kbinfos":{"chunks":[],"doc_aggs":[]}, "abstain":True}
            result=await graph._compose_answer_from_evidence(empty,tools,asyncio.Queue(),{})
            await graph._compose_answer_from_evidence(empty,tools,asyncio.Queue(),{})
            chunks=result["kbinfos"]["chunks"]
            calls=state.get("retrieval_attempts",0)
            if calls!=1 or any(c["doc_id"] not in ids for c in chunks):
                raise ValueError("重复补查或返回范围外的文档")
            rows.append({"id":case["id"],"allowed":len(ids),"retrieval_calls":calls,"chunks":len(chunks)})
    return rows


if __name__=="__main__":
    print("TIMEGUARD_CHECK "+json.dumps(asyncio.run(run(json.load(sys.stdin)))))
