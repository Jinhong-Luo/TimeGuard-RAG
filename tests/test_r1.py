"""修正引用、数据可见性、表格保持及正常接口路径的回归用例。"""
import tempfile
import asyncio
import io
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from uuid import uuid4

from timeguard.common import digest
from timeguard.judge import citation_context
from timeguard.scope import needs_expansion, rank, scope, search
from timeguard.table_trial import windows
from timeguard.native_trial import STATE, install_hooks
from timeguard.tei_adapter import Handler


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.docs = [{"id": "a", "company": "AMD", "published_at": "2023-02-01"},
                     {"id": "b", "company": "AMD"}, {"id": "c", "company": "Amcor"}]

    def test_historical_cutoff(self):
        self.assertEqual(scope("AMD FY22 as of 2023-01-31", self.docs)["document_ids"], [])
        self.assertEqual(scope("AMD FY22 as of February 1, 2023", self.docs)["years"], [2022])
        self.assertEqual(scope("AMD FY22 as of February 1, 2023", self.docs)["document_ids"], ["a"])
        self.assertEqual(scope("AMD FY22 as of 2023-01-31", self.docs, "2024-01-01")["document_ids"], [])
        self.assertIsNone(scope("报表有哪些？", self.docs)["document_ids"])

    def test_multi_year_ranking(self):
        chunks = [{"id": "a", "content": "2022"}, {"id": "b", "content": "2021 and 2022"}]
        self.assertEqual(rank(chunks, [2021, 2022], {})[0]["id"], "b")
        self.assertTrue(needs_expansion(chunks[:1], [2021]))
        self.assertFalse(needs_expansion(chunks, [2021]))
        period = scope("AMD Q2 FY2021–FY2023", self.docs)
        self.assertEqual(period["years"], [2021, 2022, 2023])
        self.assertEqual(period["quarters"], [2])
        other = {"content": "FY2022 first quarter"}
        target = {"content": "FY2022 second quarter"}
        self.assertEqual(rank([other, target], [2022], {}, [2])[0], target)
        self.assertTrue(needs_expansion([other], [2022], [2]))

    def test_empty_scope_never_queries_all(self):
        class Forbidden:
            def request(self, *args):
                raise AssertionError("空范围不应访问检索")
        result, _ = search(Forbidden(), {"retrieval": {}, "dataset_ids": ["x"]}, "问题",
                           {"document_ids": [], "years": []}, "D", {}, 6, .3)
        self.assertEqual(result, [])

    def test_trace_integrity_and_auto(self):
        ref = [{"id": "one", "document_id": "d", "content": ""},
               {"id": "two", "document_id": "d", "content": "证据二"}]
        trace = {"version": 1, "origin": "native_auto", "visible": {"1": {
            "chunk_id": "two", "document_id": "d", "content_sha256": digest("证据二")}}}
        r = {"reference": {"chunks": ref}, "answer": "有证据[ID:1]", "citation_trace": trace}
        self.assertEqual(citation_context(r)["citations"][0]["chunk_id"], "two")
        self.assertEqual(citation_context(r)["citation_origin"], "native_auto")
        trace["visible"].clear()
        self.assertEqual(citation_context(r)["mapping_status"], "incomplete")

    def test_tei_native_length_policy(self):
        data = {"query": "净销售额", "texts": ["原始完整财报"], "truncate": False, "raw_scores": False}
        raw = json.dumps(data).encode()
        handler = Handler.__new__(Handler)
        handler.path = "/rerank"
        handler.rfile = io.BytesIO(raw)
        handler.headers = {"Content-Length": str(len(raw))}
        handler.respond = Mock()
        opener = Mock()
        opener.open.return_value = io.BytesIO(b'[{"index": 0, "score": 0.5}]')
        with patch("timeguard.tei_adapter.urllib.request.build_opener", return_value=opener):
            handler.do_POST()
        payload = json.loads(opener.open.call_args.args[0].data)
        self.assertEqual(payload, {**data, "truncate": True})
        handler.respond.assert_called_once_with(200, [{"index": 0, "score": .5}])

    def test_table_headers(self):
        table = "<table><caption>USD millions</caption><tr><th>2022</th></tr>" + "".join(
            f"<tr><td>项目 {i}</td></tr>" for i in range(10)) + "</table>"
        parts = windows(table, 3)
        self.assertGreater(len(parts), 1)
        self.assertTrue(all("USD millions" in s and "2022" in s for s in parts))
        self.assertEqual(windows(table.replace("<th>", "<th rowspan=2>")), [table.replace("<th>", "<th rowspan=2>")])

    def test_native_numbering_skips_empty_and_final_crop(self):
        class Bundle:
            async def async_chat(self, system, history):
                return "两份资料[ID:1][ID:2]"
        gen = SimpleNamespace(_kb_block=lambda c, i: f"ID: {i}\n{c['content']}" if c["content"] else None,
                              num_tokens_from_string=lambda s: 1, chunks_format=lambda r: r["chunks"])
        native = SimpleNamespace(LLMBundle=Bundle)
        ret = SimpleNamespace(retrieval=None, insert_citations=lambda: "自动补引用")
        install_hooks(native, gen, ret)
        state = {"blocks": [], "visible": {}, "origin": "model"}
        token = STATE.set(state)
        try:
            chunks = [{"id": str(i), "document_id": "d", "content": c}
                      for i, c in enumerate(("", "整段", "其余正文", "超出分块预算"))]
            blocks = native.kb_prompt({"chunks": chunks}, 2.1)
            self.assertEqual(blocks, ["ID: 1\n整段", "ID: 2\n其余正文"])
            asyncio.run(Bundle().async_chat(blocks[0]+"\nID: 2\n其余", []))
            self.assertEqual(set(state["visible"]), {"1"})
            self.assertEqual(state["raw_answer"], "两份资料[ID:1][ID:2]")
            ret.insert_citations()
            self.assertEqual(state["origin"], "native_auto")
        finally:
            STATE.reset(token)


try:
    from fastapi.testclient import TestClient
    from timeguard.app import create_app
    from timeguard.financial_agent import Operand, build_graph, calculate_financial_metric, verify_operand, tool_selection
except ModuleNotFoundError as dependency:
    if dependency.name not in {"fastapi", "httpx", "langgraph", "pydantic"}:
        raise
    TestClient = None


@unittest.skipUnless(TestClient, "需安装独立应用依赖")
class AppTests(unittest.TestCase):
    def operand(self, value, year=2022):
        return Operand(chunk_id="c", quote=f"{year} USD millions {value}", number=value, year=year,
                       period="FY", currency="USD", scale="millions", kind="money")

    def test_calculation_guards(self):
        a, b = self.operand("120"), self.operand("800")
        self.assertEqual(calculate_financial_metric("margin", a, b), "15%")
        self.assertEqual(calculate_financial_metric("growth", a, self.operand("100", 2021)), "20%")
        self.assertEqual(calculate_financial_metric("margin", self.operand("(120)"), b), "-15%")
        for bad in (dict(number="0"), dict(currency="EUR"), dict(year=2021), dict(period="Q1"),
                    dict(scale="ones"), dict(kind="percentage")):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                calculate_financial_metric("margin", a, b.model_copy(update=bad))
        with self.assertRaises(ValueError):
            verify_operand(a, {"content": "2022 USD millions (120)"})
        with self.assertRaises(ValueError):
            calculate_financial_metric("margin", a.model_copy(update={"currency": "XXX"}),
                                       b.model_copy(update={"currency": "XXX", "chunk_id": "其他表"}))

    def test_real_graph_http_replay_restart(self):
        a, b = self.operand("120"), self.operand("800")
        data = {"id": "c", "document_id": "d", "content": a.quote+"\n"+b.quote}
        async def find(q, p):
            self.assertEqual(p["document_ids"], ["d"])
            return [data]
        async def get(c):
            return data
        async def select(q, chunks):
            fn = {"name": "calculate_financial_metric", "arguments": json.dumps(
                {"metric": "margin", "left": a.model_dump(), "right": b.model_dump()})}
            selection, call = tool_selection({"tool_calls": [{"id": "call-test", "type": "function", "function": fn}]})
            return selection, {"total_tokens": 18}, call
        graph = build_graph(find, get, select, [{"company": "AMD", "id": "d"}])
        with tempfile.TemporaryDirectory() as root:
            db = Path(root, "sessions.db")
            req = {"request_id": str(uuid4()), "session_id": str(uuid4()), "question": "AMD 净利率是多少？"}
            with TestClient(create_app(db, graph)) as client:
                self.assertEqual(client.post("/turn", json=req).json()["status"], "clarify")
            with TestClient(create_app(db, graph)) as client:
                req.update(revision=1, question="FY2022", request_id=str(uuid4()))
                result = client.post("/turn", json=req).json()
                self.assertEqual(result["answer"], "120 ÷ 800 = 15%")
                self.assertEqual(len(result["facts"]), 2)
                self.assertEqual(result["tool_call"]["name"], "calculate_financial_metric")
                self.assertEqual(client.post("/turn", json=req).json(), result)
                self.assertEqual(client.post("/turn", json={**req,"question":"其他"}).status_code, 409)
                self.assertEqual(client.get("/metrics").json()["total_tokens"], 18)

    def test_capacity_timeout_and_recovery(self):
        entered = threading.Event()
        class SlowGraph:
            async def ainvoke(self, state, config):
                if state["question"] == "慢请求":
                    entered.set()
                    await asyncio.sleep(2)
                return {"status": "clarify", "answer": "请补充财年"}
        def turn(question):
            return {"question": question, "session_id": str(uuid4()), "request_id": str(uuid4())}
        with tempfile.TemporaryDirectory() as root, TestClient(
                create_app(Path(root, "test.db"), SlowGraph(), capacity=1, timeout=.5)) as client:
            with ThreadPoolExecutor(max_workers=1) as pool:
                slow = pool.submit(client.post, "/turn", json=turn("慢请求"))
                self.assertTrue(entered.wait(2))
                self.assertEqual(client.post("/turn", json=turn("第二个请求")).status_code, 429)
                self.assertEqual(slow.result().status_code, 504)
            self.assertEqual(client.post("/turn", json=turn("恢复后")).status_code, 200)
            self.assertEqual(client.get("/metrics").json()["errors"], 1)

    def test_circuit_probe(self):
        class Graph:
            async def ainvoke(self, state, config):
                if state["question"] == "故障":
                    raise ValueError("上游输入损坏")
                return {"status": "clarify", "answer": "请补充财年"}
        def request(q):
            return {"request_id": str(uuid4()), "session_id": str(uuid4()), "question": q}
        with tempfile.TemporaryDirectory() as root, TestClient(
                create_app(Path(root, "test.db"), Graph(), cooldown=.2)) as client:
            for _ in range(3):
                self.assertEqual(client.post("/turn", json=request("故障")).status_code, 502)
            self.assertEqual(client.post("/turn", json=request("恢复")).status_code, 503)
            time.sleep(.25)
            self.assertEqual(client.post("/turn", json=request("恢复")).status_code, 200)
            self.assertEqual(client.get("/metrics").json()["consecutive_errors"], 0)

