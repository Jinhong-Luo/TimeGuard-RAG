"""固定 v0.27.3 原生能力评测；观察执行过程，不替换解析、检索或回答算法。"""
import argparse
import asyncio
from collections import Counter
from contextlib import ExitStack
import contextvars
from datetime import datetime, timezone
import functools
import hashlib
import inspect
from itertools import groupby
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

from timeguard.worker_bundle import PACKAGE_ROOT, package_files, source_hashes, stage_package
from timeguard.common import Client, digest, read_json, read_jsonl, write_json

CONTAINER = "timeguard_ljh-ragflow-cpu-1"
INSTANCE = "native_baseline_20261109"
PROVIDER = "OpenAI-API-Compatible"
MODEL = f"gpt-6.1-sol@low@{INSTANCE}@{PROVIDER}"
DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L6-v2@timeguard_r1_ce_tokenizer@HuggingFace"
ARMS = ("plain", "keyword", "rerank", "metadata_auto", "metadata_all", "agent_low", "agent_medium", "agent_high", "agent_ultra", "combined")
CURRENT = contextvars.ContextVar("原生基线题目", default=None)
PAIRED = ("pair_a0", "pair_a1")
SNAPSHOTS = ("snapshot", "page_select", "page", "supplement", "brief", "wide", "route", "final", *PAIRED)
SCOPED = ("temporal", "adaptive", "finance", "page", "supplement", "route", "final", *PAIRED)
BOUND = ("final", "route", "page", "supplement", *PAIRED)


def arm_config(name, reranker):
    if name not in ARMS:
        raise ValueError("未知原生配置")
    config = {"name": name, "reasoning": {"agent_low": 1, "agent_medium": 2, "agent_high": 3,
            "agent_ultra": 4, "combined": 2}.get(name, 0),
            "keyword": name == "keyword", "rerank_id": reranker if name in ("rerank", "combined") else "",
            "meta_data_filter": {"method": "semi_auto", "semi_auto": ["tg_native_company"]}
            if name in ("metadata_auto", "combined") else {}}
    if name=="metadata_all":config["meta_data_filter"]={"method":"auto"}
    return config


def select_queries(rows, split, limit=None):
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("题目包含重复编号")
    selected = [{k: r[k] for k in ("id", "question", "split", "as_of") if k in r} for r in rows if split == "all" or r["split"] == split]
    if limit is not None:
        if limit < 1:
            raise ValueError("样本数必须为正数")
        selected = selected[:limit]
    if not selected:
        raise ValueError("没有计划题目")
    return selected


def trace_event(kind, **values):
    state = CURRENT.get()
    if state is not None:
        state["events"].append({"kind": kind, **values})


def install_observers(service, generator, retriever):
    """只记录调用参数与结果摘要，所有返回值原样交回固定版上游。"""
    retrieve = retriever.retrieval
    @functools.wraps(retrieve)
    async def observed_retrieve(*args, **kwargs):
        started = time.monotonic()
        value = await retrieve(*args, **kwargs)
        trace_event("retrieval", question=str(args[0])[:1000] if args else kwargs.get("question"),
                    doc_ids=kwargs.get("doc_ids"), reranker_used=kwargs.get("rerank_mdl") is not None,
                    chunks=len(value.get("chunks", [])), seconds=time.monotonic()-started)
        state = CURRENT.get()
        if state is not None:
            state["retrieved"] = generator.chunks_format(value)
        return value
    retriever.retrieval = observed_retrieve
    original_prompt = service.kb_prompt
    @functools.wraps(original_prompt)
    def observed_prompt(info, *args, **kwargs):
        blocks = original_prompt(info, *args, **kwargs)
        state = CURRENT.get()
        if state is not None:
            state["prompt_blocks"] = [{"text": block, "id": int(re.search(r"\bID:\s*(\d+)", block)[1])}
                                      for block in blocks if re.search(r"\bID:\s*(\d+)", block)]
        return blocks
    service.kb_prompt = observed_prompt
    original_filter = service.apply_meta_data_filter
    @functools.wraps(original_filter)
    async def observed_filter(*args, **kwargs):
        value = await original_filter(*args, **kwargs)
        trace_event("metadata_filter", config=args[0] if args else kwargs.get("meta_data_filter"),
                    selected_doc_ids=value)
        return value
    service.apply_meta_data_filter = observed_filter
    original_insert = retriever.insert_citations
    @functools.wraps(original_insert)
    def observed_insert(*args, **kwargs):
        state = CURRENT.get()
        if state is not None:
            state["citation_origin"] = "native_auto"
        return original_insert(*args, **kwargs)
    retriever.insert_citations = observed_insert
    original_chat = service.LLMBundle.async_chat
    @functools.wraps(original_chat)
    async def observed_chat(self, system, history, *args, **kwargs):
        trace_event("model_chat", tools_bound=bool(getattr(self.mdl, "tools", None)),
                    tool_support=self.is_tools, system_sha256=digest(system))
        state = CURRENT.get()
        if state is not None:
            state["visible_ids"] = [b["id"] for b in state["prompt_blocks"] if b["text"] in system]
        value = await original_chat(self, system, history, *args, **kwargs)
        if state is not None:
            state["raw_model_answer"] = value
        return value
    service.LLMBundle.async_chat = observed_chat
    for method in ("rag", "retrieve", "judge_sufficiency", "gen_followups", "formalize", "pick_documents"):
        original = getattr(service.RAGTools, method, None)
        if original is None or not inspect.iscoroutinefunction(original):
            continue
        def wrap(fn, label):
            @functools.wraps(fn)
            async def observed(self, *args, **kwargs):
                trace_event("agent_tool", name=label, thinking_mode=getattr(self, "thinking_mode", None))
                return await fn(self, *args, **kwargs)
            return observed
        setattr(service.RAGTools, method, wrap(original, method))


async def native_worker(spec):
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    for name, expected in spec.get("upstream_files", {}).items():
        if digest(Path("/ragflow", name).read_bytes()) != expected:
            raise ValueError("运行容器与固定上游源码不同："+name)
    from api.apps import app
    from common import settings
    settings.init_settings()
    from api.db.services import dialog_service as service
    from rag.prompts import generator
    install_observers(service, generator, settings.retriever)
    candidate = spec.get("candidate")
    paired = candidate in PAIRED
    official = spec.get("model_provider") == "openai"
    if official:
        from timeguard.llm.openai_native import install
        install(spec["model_endpoint"], CURRENT)
    if candidate in ("layers", *SNAPSHOTS):
        from timeguard.evidence.evidence_layers import install
        install(generator, settings, CURRENT, service.RAGTools, spec if candidate in SNAPSHOTS else None)
    elif candidate:
        from timeguard.research.agentic_candidate import install
        install(service, generator, settings, spec, CURRENT, trace_event)
    if candidate in BOUND:
        from timeguard.retrieval.scope import intersect_scope
        native_filter=service.apply_meta_data_filter
        @functools.wraps(native_filter)
        async def bounded_filter(*args,**kwargs):
            selected=await native_filter(*args,**kwargs)
            trace=CURRENT.get()
            allowed=intersect_scope(trace["plan"]["document_ids"],selected)
            trace["allowed"]=allowed
            return allowed or ["-999"]
        service.apply_meta_data_filter=bounded_filter
    async def one(q):
        started = time.monotonic()
        if spec.get("ordered_pairs"):
            print("TIMEGUARD_START " + json.dumps({"position": q["position"]}), flush=True)
        state = {"events": [], "retrieved": [], "prompt_blocks": [], "visible_ids": [], "citation_origin": "model"}
        state["question"] = q["question"]
        if paired:
            state["pair_arm"] = q.get("pair_arm", "A1" if candidate == "pair_a1" else "A0")
        if spec.get("candidate") in ("page", "supplement"):
            state["page_search"] = spec["page_searches"][q["id"]]
        if candidate in ("layers", *SNAPSHOTS):
            state["evidence_layers"] = {"version": 2, "retrievals": [], "answers": []}
        token = CURRENT.set(state)
        row = {**q, "status": "ok", "answer": None, "reference": {}, "usage": None, "cost_usd": None}
        try:
            found, dialog = service.DialogService.get_by_id(spec["chat_id"])
            if not found or dialog.llm_id != spec.get("model", MODEL) or dialog.kb_ids != [spec["dataset_id"]]:
                raise ValueError("独立原生对话身份不匹配")
            cfg = spec["arm_config"]
            dialog.rerank_id = cfg["rerank_id"]
            dialog.meta_data_filter = cfg["meta_data_filter"]
            dialog.prompt_config = {**dialog.prompt_config, "reasoning": cfg["reasoning"], "keyword": cfg["keyword"]}
            if cfg["reasoning"]:
                models = service.get_models(dialog)
                support = bool(models[3].is_tools)
                trace_event("agent_prerequisite",tool_support=support,
                            provider_bound_before_bind=bool(getattr(models[3].mdl,"is_tools",False)))
                if not support:
                    raise ValueError("模型未声明工具能力，禁止把普通问答回退当作Agentic成功")
            messages = [{"role": "user", "content": q["question"]}]
            if candidate in SCOPED:
                from timeguard.retrieval.scope import scope
                state["plan"] = scope(q["question"], spec["documents"], q.get("as_of"))
                if not state["plan"]["companies"] or (paired and len(state["plan"]["companies"]) != 1):
                    row.update(answer="请明确要查询的公司。", reference={"chunks": []})
                    if paired:
                        row["answer_status"] = "clarification"
                elif state["plan"]["document_ids"] == []:
                    row.update(answer="截止日内没有可核验的已披露文档。", reference={"chunks": []})
                    if paired:
                        row["answer_status"] = "no_evidence"
                else:
                    messages[0]["doc_ids"] = state["plan"]["document_ids"]
            if spec.get("candidate")=="route":
                light=spec["light_checks"].get(q["id"])
                state["light_check"]={k:v for k,v in (light or {}).items() if k!="result"}
                if light and light.get("result") and row["answer"] is None:
                    row.update(light["result"])
                    state["route_decision"]="轻量原文查数"
            async with asyncio.timeout(spec.get("timeout_seconds",1200)):
                async with app.app_context():
                    if row["answer"] is None:
                        async for result in service.rag_agent(dialog, messages, stream=False, reasoning=cfg["reasoning"]):
                            row.update(answer=result.get("answer"), reference={"chunks": generator.chunks_format(result.get("reference"))},
                                       usage=result.get("usage"))
                        if official and not any(e["kind"] == "openai_request" and e["success"] for e in state["events"]):
                            raise ValueError("本题没有可验证的官方 API 调用")
                        if paired and official and any(e["kind"] == "openai_request" and not e["attempted"]
                                                      for e in state["events"]):
                            raise ValueError("原生模型请求有未通过转换校验的位置")
                        if paired:
                            layers = state["evidence_layers"]["answers"]
                            if len(layers) != 1:
                                raise ValueError("本题缺少唯一的最终作答节点")
                            if layers[0].get("empty_response"):
                                row["answer_status"] = "no_evidence"
                            elif not any(r["status"] == 200 for r in layers[0]["requests"]):
                                raise ValueError("本题缺少成功的最终模型请求")
                            else:
                                row["answer_status"] = "completed"
            if spec.get("candidate") == "evidence":
                row["citation_trace"] = {"version": 1, "visible": state.get("visible", {}),
                                         "origin": "model", "raw_answer": state.get("evidence_raw_answer")}
            if not row["answer"] or "**ERROR**" in row["answer"]:
                raise ValueError("原生回答为空或带错误标记")
        except Exception as exc:
            row.update(status="request_error", error=(type(exc).__name__+": "+str(exc))[:1500])
        finally:
            CURRENT.reset(token)
            state.pop("_openai_tool_turns", None)
        state.pop("prompt_blocks", None)
        row.update(native_trace=state, retrieved_chunks=state["retrieved"], elapsed_seconds=time.monotonic()-started)
        print("TIMEGUARD_RESULT "+json.dumps(row,ensure_ascii=False),flush=True)
        return row["status"] == "ok"
    if spec.get("ordered_pairs"):
        groups = [list(group) for _, group in groupby(spec["queries"], key=lambda query: query["id"])]

        async def pair(group):
            results = [await one(query) for query in group]
            return all(results)

        successful = True
        for i in range(0, len(groups), spec["concurrency"]):
            results = await asyncio.gather(*(pair(group) for group in groups[i:i + spec["concurrency"]]))
            successful = all(results) and successful
        return successful
    for i in range(0,len(spec["queries"]),spec["concurrency"]):
        ok = await asyncio.gather(*(one(q) for q in spec["queries"][i:i+spec["concurrency"]]))
        if not all(ok):
            return False
    return True


def worker_sources(candidate, official=False):
    files = ["evals/baselines/native_baseline.py"]
    if official:
        files.extend(["llm/openai_transport.py", "llm/openai_native.py"])
    if candidate:
        files.extend(["research/agentic_candidate.py", "retrieval/scope.py", "retrieval/catalog.py"])
    if candidate in ("layers", *SNAPSHOTS):
        files.append("evidence/evidence_layers.py")
    if candidate in SNAPSHOTS:
        files.append("evidence/final_snapshot.py")
    if candidate in ("route", "page_select"):
        files.append("retrieval/page_select.py")
    if candidate == "route":
        files.extend(["research/finance_route.py", "evidence/evidence_review.py", "llm/bridge.py"])
    if candidate in ("adaptive", "finance"):
        files.append("research/agentic_finance.py")
    if candidate in ("adaptive", "finance", "route", "pair_a1"):
        files.append("research/financial_agent.py")
    if candidate in ("route", "pair_a1"):
        files.append("evidence/formula_guard.py")
    if candidate == "pair_a1":
        files.extend(["research/live_finance.py", "evidence/fact_selection.py",
                      "evidence/candidates.py", "llm/bridge.py"])
    return package_files(*files)


def container_worker(spec, out):
    official = spec.get("model_provider") == "openai"
    with tempfile.TemporaryDirectory() as temp:
        package = Path(temp, "timeguard")
        files = worker_sources(spec.get("candidate"), official)
        hashes = stage_package(package, files)
        if spec.get("candidate_files") and spec["candidate_files"] != hashes:
            raise ValueError("候选源码与冻结清单不匹配")
        target = "/tmp/timeguard-native-baseline-"+digest(hashes)[:16]
        subprocess.run(["docker","exec",CONTAINER,"mkdir","-p",target],check=True)
        subprocess.run(["docker","cp",str(package),CONTAINER+":"+target+"/"],check=True,stdout=subprocess.DEVNULL)
    argv=["docker","exec","-i","-w","/ragflow","-e","PYTHONPATH="+target+":/ragflow",CONTAINER,"python","-m","timeguard.evals.baselines.native_baseline","--worker"]
    env=os.environ.copy()
    if official:
        from timeguard.llm.openai_transport import api_key
        env["OPENAI_API_KEY"] = api_key()
        index = argv.index(CONTAINER)
        argv[index:index] = ["-e", "OPENAI_API_KEY"]
    if spec.get("candidate")=="route":
        _,rt=private_runtime()
        env.update(TIMEGUARD_REVIEW_TOKEN=rt["token"],TIMEGUARD_REVIEW_URL=rt["base_url"])
        index=argv.index(CONTAINER)
        argv[index:index]=["-e","TIMEGUARD_REVIEW_TOKEN","-e","TIMEGUARD_REVIEW_URL"]
    with ExitStack() as handles:
        log = handles.enter_context((out/"worker.log").open("w", encoding="utf-8"))
        rows = handles.enter_context((out/"results.jsonl").open("x", encoding="utf-8"))
        started = handles.enter_context((out/"started.jsonl").open("x", encoding="utf-8")) if spec.get("ordered_pairs") else None
        with subprocess.Popen(argv,env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,
                              text=True,encoding="utf-8",errors="replace") as proc:
            proc.stdin.write(json.dumps(spec));proc.stdin.close()
            for line in proc.stdout:
                if line.startswith("TIMEGUARD_RESULT "):
                    row=json.loads(line.partition(" ")[2]);rows.write(json.dumps(row,ensure_ascii=False)+"\n");rows.flush()
                    print(out.name,row.get("position",row["id"]),row["status"],round(row["elapsed_seconds"],2),flush=True)
                elif started is not None and line.startswith("TIMEGUARD_START "):
                    started.write(line.partition(" ")[2])
                    started.flush()
                else:
                    log.write(line);log.flush()
            return proc.wait()


def private_runtime():
    base=Path("data").resolve().parent
    private=base/".cache/native-complete-20261109"
    return private,read_json(private/"bridge.runtime.json")


def metadata_action(root, action):
    private,_=private_runtime()
    state=read_json("data/ingestion.json")
    docs=[{**d,"id":state["documents"][d["doc_name"]]["id"]}
          for d in read_jsonl("data/financebench/documents.jsonl")]
    spec={"action":action,"documents":docs,"dataset_id":state["dataset_id"]}
    backup=private/"metadata-backup.json"
    if action!="snapshot":spec["backup"]=read_json(backup)
    if action=="snapshot" and backup.exists():raise ValueError("不能覆盖元数据原始快照")
    with tempfile.TemporaryDirectory() as temp:
        package = Path(temp, "timeguard")
        files = package_files("evals/baselines/native_metadata.py")
        hashes = stage_package(package, files)
        target = "/tmp/timeguard-native-metadata-" + digest(hashes)[:12]
        subprocess.run(["docker","exec",CONTAINER,"mkdir","-p",target],check=True)
        subprocess.run(["docker","cp",str(package),CONTAINER+":"+target+"/"],check=True,stdout=subprocess.DEVNULL)
    proc=subprocess.run(["docker","exec","-i","-w","/ragflow","-e","PYTHONPATH="+target+":/ragflow",CONTAINER,
                         "python","-m","timeguard.evals.baselines.native_metadata"],input=json.dumps(spec),text=True,capture_output=True)
    (private/("metadata-"+action+".log")).write_text(proc.stderr+proc.stdout,encoding="utf-8")
    if proc.returncode:raise RuntimeError("原生元数据操作失败，详情保存在私有日志")
    values=[json.loads(l.partition(" ")[2]) for l in proc.stdout.splitlines() if l.startswith("TIMEGUARD_METADATA ")]
    if len(values)!=1:raise ValueError("缺少原生元数据结果")
    if action=="snapshot":
        write_json(backup,values[0]);backup.chmod(0o600)
        write_json(root/"metadata-snapshot.json",{"documents":len(values[0]),"original_nonempty":sum(bool(v) for v in values[0].values()),"backup_sha256":digest(backup.read_bytes())})
    else:write_json(root/("metadata-"+action+".json"),values[0])
    print("原生元数据操作完成："+action,flush=True)


def prepare(out):
    private,rt=private_runtime()
    client=Client(read_json("configs/r0-low.json")["base_url"],timeout=1200)
    instances=client.request("GET",f"/providers/{PROVIDER}/instances")
    if any(i["instance_name"]==INSTANCE for i in instances):
        raise ValueError("专用模型实例已存在，拒绝覆盖")
    client.request("POST",f"/providers/{PROVIDER}/instances",{
        "instance_name":INSTANCE,"api_key":rt["token"],"base_url":rt["base_url"],"region":"","verify":False,
        "model_info":[{"model_name":"gpt-6.1-sol@low","model_type":["chat"],"max_tokens":32768,"extra":{"is_tools":True}}]})
    from timeguard.evals.baselines.run import chat_body
    state=read_json("data/ingestion.json");config=read_json("configs/r0-low.json")
    config.update(model=MODEL,dataset_ids=[state["dataset_id"]])
    payload=chat_body(config,"default","TimeGuard-原生完整基线-20261109")
    chat=client.request("POST","/chats",payload)
    write_json(out/"setup.json",{"chat_id":chat["id"],"dataset_id":state["dataset_id"],"model":MODEL,
        "config":config,"bridge_base_url":rt["base_url"],"instance":INSTANCE,"chat_config":chat})
    print("已创建独立模型实例和原生对话；工具能力仍须从实际模型对象和调用核验",flush=True)


def run_arm(root, arm, split, limit, reranker, label="", candidate=None, query_ids=None,
            model_provider="codex", setup_path=None):
    if label and not re.fullmatch(r"[a-z0-9_-]+",label):raise ValueError("运行标签格式错误")
    if candidate and arm != "combined":
        raise ValueError("优化候选只适用于公司过滤与 Agentic medium 组合")
    official = model_provider == "openai"
    if official:
        from timeguard.llm import openai_transport as api
        api.api_key()
    setup=read_json(setup_path or root/"setup.json")
    out=root/(arm+("-"+candidate if candidate else "")+("-openai" if official else "")+
              ("-"+label if label else "")+("-smoke" if limit else "-"+split))
    out.mkdir(parents=True,exist_ok=False)
    queries=select_queries(read_jsonl("data/financebench/queries.jsonl"),split,limit)
    if query_ids:
        if limit or len(query_ids) != len(set(query_ids)):
            raise ValueError("固定题号不能重复或与数量截断同时使用")
        options = {q["id"]: q for q in queries}
        queries = [options[key] for key in query_ids]
    from timeguard.evals.baselines.matrix import frozen_index
    freeze=frozen_index(Path("data/ingestion.json"),Path("runs/r0-gpu-retrieval-all"),Path("data/financebench/queries.jsonl"))
    upstream_files={name:digest(Path("vendor/ragflow",name).read_bytes()) for name in
                    ("api/db/services/dialog_service.py","rag/prompts/generator.py","rag/nlp/search.py",
                     "common/metadata_utils.py","rag/advanced_rag/harness/config.py")}
    spec={**{k:setup[k] for k in ("chat_id","dataset_id","model")},"queries":queries,
          "arm_config":arm_config(arm,reranker),"concurrency":2,"upstream_files":upstream_files,"timeout_seconds":1200}
    if official:
        spec.update(model_provider=model_provider, model_endpoint=setup["bridge_base_url"])
    if candidate:
        spec["candidate"] = candidate
        if candidate in SNAPSHOTS:
            for name in ("rag/advanced_rag/harness/tools/search.py", "rag/advanced_rag/harness/prompts/report_prompt.py"):
                upstream_files[name] = digest(Path("vendor/ragflow", name).read_bytes())
        spec["candidate_source_sha256"] = digest((PACKAGE_ROOT / "research/agentic_candidate.py").read_bytes())
        for name in ("rag/advanced_rag/agentic_rag_graph.py", "rag/advanced_rag/agentic_rag.py"):
            upstream_files[name] = digest(Path("vendor/ragflow", name).read_bytes())
        if candidate in SCOPED:
            from timeguard.retrieval.catalog import catalog
            spec["documents"] = catalog(read_jsonl("data/financebench/documents.jsonl"), read_json("data/ingestion.json"))
    if candidate or official:
        spec["candidate_files"] = source_hashes(worker_sources(candidate, official))
    if candidate in ("page", "supplement"):
        from timeguard.retrieval.filing_pages import FilingPages
        from timeguard.retrieval.scope import scope
        worker = FilingPages(Client(setup["config"]["base_url"]),setup["dataset_id"],spec["documents"])
        spec["page_searches"] = {}
        for q in queries:
            plan = scope(q["question"], spec["documents"], q.get("as_of"))
            limits = {"input":{"tools":{"doc_scope":plan["document_ids"]}}}
            spec["page_searches"][q["id"]] = worker.retrieve(q["question"],limits,q.get("as_of"),candidate=="supplement")
        spec["page_search_code"] = source_hashes(("retrieval/filing_pages.py", "retrieval/page_select.py",
                                                "retrieval/catalog.py", "retrieval/scope.py"))
    if candidate=="route":
        from timeguard.research.finance_route import simple_metric,simple_lookup
        from timeguard.retrieval.filing_pages import FilingPages
        from timeguard.retrieval.scope import scope
        client=Client(setup["config"]["base_url"])
        worker=FilingPages(client,setup["dataset_id"],spec["documents"])
        spec["light_checks"]={}
        for q in queries:
            if not simple_metric(q["question"]):
                continue
            started=time.monotonic()
            plan=scope(q["question"],spec["documents"],q.get("as_of"))
            pool=worker.retrieve(q["question"],{"input":{"tools":{"doc_scope":plan["document_ids"]}}},q.get("as_of"),False)
            reads=[]
            def read(c):
                data=client.request("GET",f"/datasets/{setup['dataset_id']}/documents/{c['document_id']}/chunks?id={c['id']}")
                raw,=data["chunks"]
                reads.append(raw["id"])
                return raw
            result=simple_lookup(q["question"],pool["page"],pool["allowed"],read)
            spec["light_checks"][q["id"]]={"result":result,"index_reads":pool["index_reads"],"readbacks":len(reads),
                                            "seconds":time.monotonic()-started}
    manifest={"identity":{"query_ids":[q["id"] for q in queries],"split":split,"config":setup["config"],
        "arm":spec["arm_config"],"scope":"指定题号回归" if query_ids else ("连通验证" if limit else "完整分组评测"),
        "source_sha256":digest(Path(__file__).read_bytes()),"queries_sha256":digest(Path("data/financebench/queries.jsonl").read_bytes()),
        "upstream":read_json("upstream.lock.json"),"upstream_files":upstream_files,"freeze":freeze,"spec_sha256":digest(spec)},
        "created_at_utc":datetime.now(timezone.utc).isoformat()}
    if official:
        manifest["identity"]["transport"] = api.TRANSPORT
    write_json(out/"manifest.json",manifest);write_json(out/"spec.json",spec)
    audit = None
    if not official:
        private,_=private_runtime()
        audit=private/"bridge-audit.jsonl"
    start=audit.stat().st_size if audit and audit.exists() else 0
    began=time.monotonic()
    code=container_worker(spec,out)
    rows=read_jsonl(out/"results.jsonl")
    if official:
        records = [event for row in rows for event in row["native_trace"]["events"]
                   if event["kind"] == "openai_request" and event["attempted"]]
        raw = "\n".join(json.dumps(r, ensure_ascii=False) for r in records).encode()
        end = len(raw)
        ids = [r["openai_request_id"] for r in records if r.get("openai_request_id")]
        successes = [r for r in records if r["success"]]
        verified = bool(successes) and all(api.model_matches(r.get("actual_model")) and
                                           r.get("reasoning_effort") == "low" for r in successes)
        tokens = sum((r.get("raw_usage") or {}).get("total_tokens", 0) for r in records)
    else:
        end=audit.stat().st_size if audit.exists() else 0
        raw=audit.read_bytes()[start:end] if end else b""
        records=[json.loads(l) for l in raw.splitlines() if l.strip()]
        ids = [r.get("request_id") for r in records]
        verified = all(r.get("actual_model")=="gpt-6.1-sol" and r.get("reasoning_effort")=="low"
                       for r in records if r.get("success"))
        tokens = sum((r.get("usage") or {}).get("totalTokens",0) for r in records)
    write_json(out/"execution.json",{"returncode":code,"planned":len(queries),"completed":len(rows),
        "wall_seconds":time.monotonic()-began,"statuses":dict(Counter(r["status"] for r in rows)),
        "audit_byte_start":start,"audit_byte_end":end,"audit_sha256":digest(raw),"actual_calls":len(records),
        "failed_calls":sum(r.get("success") is not True for r in records),
        "model_identity_verified":verified, "unique_audit_request_ids":len(set(ids))==len(records),
        "tokens":tokens,
        "results_sha256":digest((out/"results.jsonl").read_bytes()),"cost_usd":None,
        "raw_upstream":not bool(candidate),"candidate":candidate,
        "notes":"独立候选实验" if candidate else "原生基线"})
    if code or len(rows)!=len(queries):
        raise SystemExit(1)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--worker",action="store_true")
    p.add_argument("--prepare",action="store_true")
    p.add_argument("--metadata",choices=("snapshot","populate","checks","restore"))
    p.add_argument("--output",type=Path)
    p.add_argument("--arm",choices=ARMS,default="plain")
    p.add_argument("--split",choices=("tune","dev","test","all"),default="dev")
    p.add_argument("--limit",type=int)
    p.add_argument("--query-ids", nargs="+")
    p.add_argument("--label",default="")
    p.add_argument("--model-provider", choices=("codex", "openai"), default="codex")
    p.add_argument("--setup", type=Path, help="可复用现有专用原生对话配置")
    if "--worker" not in sys.argv:
        from timeguard.research.agentic_candidate import VARIANTS
        p.add_argument("--candidate", choices=VARIANTS)
    p.add_argument("--reranker", default=DEFAULT_RERANKER)
    args=p.parse_args()
    if args.worker:
        raise SystemExit(0 if asyncio.run(native_worker(json.load(sys.stdin))) else 1)
    if args.output is None:
        p.error("必须提供独立输出目录")
    if args.metadata:
        metadata_action(args.output,args.metadata)
    elif args.prepare:
        prepare(args.output)
    else:
        run_arm(args.output,args.arm,args.split,args.limit,args.reranker,args.label,args.candidate,args.query_ids,
                args.model_provider,args.setup)


if __name__=="__main__":
    main()
