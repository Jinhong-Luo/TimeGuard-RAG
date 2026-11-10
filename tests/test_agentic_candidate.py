"""针对组合组实际错题的回归检查。"""
import asyncio
from pathlib import Path
import tempfile
import unittest
from uuid import uuid4

from timeguard.agentic_candidate import evidence_blocks
from timeguard.scope import scope


class CandidateTests(unittest.TestCase):
    def test_ids_after_dedup_and_budget(self):
        chunks = [{"doc_id":"a","chunk_id":k,"text":t} for k,t in
                  [("a",""),("b","原文"),("b","原文"),("c","过长"*8),("d","短")]]
        kept, blocks = evidence_blocks(chunks, lambda c,i: f"{i}-{c['text']}" if c["text"] else None, len, 8)
        self.assertEqual([c["chunk_id"] for c in kept], ["b","d"])
        self.assertEqual(blocks, ["0-原文","1-短"])

    def test_restated_period_and_disclosure(self):
        docs = [{"id":"old","company":"AMD","published_at":"2021-02-01"},
                {"id":"new","company":"AMD","published_at":"2023-02-01"},
                {"id":"unknown","company":"AMD"},{"id":"other","company":"3M","published_at":"2020-01-01"}]
        self.assertEqual(scope("AMD FY2020",docs)["document_ids"],["old","new","unknown"])
        for day, expected in [("2021-01-31",[]),("2021-02-01",["old"]),("2023-02-01",["old","new"])]:
            self.assertEqual(scope("AMD FY2020 as of "+day, docs)["document_ids"],expected)

try:
    from timeguard.agentic_finance import direct_lookup
    from timeguard.financial_agent import Operand, calculate_financial_metric, present_evidence, verify_operand
    from fastapi.testclient import TestClient
    from timeguard.app import create_app
except ModuleNotFoundError as exc:
    if exc.name not in {"langgraph","pydantic","fastapi","httpx"}:
        raise
    TestClient = None


@unittest.skipUnless(TestClient, "需要应用依赖")
class FinancialTests(unittest.TestCase):
    def setUp(self):
        html = "<table><caption>Consolidated USD millions</caption><tr><th></th><th>2022</th><th>2021</th></tr><tr><td>Total assets</td><td>20</td><td>18</td></tr><tr><td>Tax rate</td><td>0.62%</td><td>-14.76%</td></tr></table>"
        self.chunk = {"id":"c","document_id":"d","content":html}
        self.shown = present_evidence([self.chunk])[0]["visible_lines"]
        self.a = Operand(chunk_id="c",year=2022,period="FY",number="20",currency="USD",scale="millions",
                         kind="money",line_number=3,row_label="Total assets")

    def test_wrong_year_and_metric(self):
        verify_operand(self.a,self.chunk,self.shown)
        for change in [{"year":2021},{"row_label":"Net sales"},{"scale":"ones"}]:
            with self.subTest(change=change),self.assertRaises(ValueError):
                verify_operand(self.a.model_copy(update=change),self.chunk,self.shown)

    def test_rate_sign_and_points(self):
        left = self.a.model_copy(update={"kind":"percentage","line_number":4,"row_label":"Tax rate","number":"0.62"})
        right = left.model_copy(update={"year":2021,"number":"-14.76"})
        for op in (left,right):
            verify_operand(op,self.chunk,self.shown)
        with self.assertRaises(ValueError):
            verify_operand(right.model_copy(update={"number":"14.76"}),self.chunk,self.shown)
        self.assertEqual(calculate_financial_metric("percentage_point_change",left,right),"15.38 个百分点")

    def test_multiperiod_sum(self):
        a=self.a.model_copy(update={"row_label":"Net sales"})
        b=a.model_copy(update={"year":2021,"number":"18"})
        c=a.model_copy(update={"year":2020,"number":"12"})
        self.assertEqual(calculate_financial_metric("period_sum",a,b,[c]),"50 USD millions")
        with self.assertRaises(ValueError):
            calculate_financial_metric("period_sum",a,b,[b])

    def test_light_route_checks_source(self):
        async def read(c):
            return c
        plan={"companies":["AMD"],"years":[2022],"quarters":[]}
        q="What was AMD's FY2022 total assets (in USD millions)?"
        result=asyncio.run(direct_lookup(q,plan,[self.chunk],{"c":self.chunk},read))
        self.assertEqual(result[2][0]["number"],"20")
        self.assertIsNone(asyncio.run(direct_lookup("比较各季度增长率",plan,[self.chunk],{},read)))

    def test_bad_json_and_retry(self):
        import json
        class Graph:
            fails=True
            async def ainvoke(self, state, config):
                if self.fails:
                    json.loads("{")
                return {"status":"clarify","answer":"请补充财年"}
        graph=Graph()
        req={"request_id":str(uuid4()),"session_id":str(uuid4()),"question":"计算利润率"}
        with tempfile.TemporaryDirectory() as tmp, TestClient(create_app(Path(tmp)/"db",graph)) as client:
            self.assertEqual(client.post("/turn",json=req).status_code,502)
            graph.fails=False
            result=client.post("/turn",json=req).json()
            self.assertEqual(result["revision"],1)
            self.assertEqual(result["trace"][-1]["stage"],"计算结果")
            self.assertEqual(client.post("/turn",json=req).json(),result)

    def test_native_http_failures(self):
        import httpx
        from unittest.mock import patch
        from timeguard.app import native_graph

        async def run():
            docs = [{"company":"AMD","id":"d","doc_name":"AMD_2022_10K"}]
            config = {"base_url":"http://test","retrieval":{}}
            for response in (httpx.Response(502), httpx.Response(200, content=b"{"),
                             httpx.Response(200, json={"code":0,"data":{"chunks":[{"document_id":"other"}]}})):
                async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _:response)) as http:
                    with patch("timeguard.app.read_json",return_value=config), patch("timeguard.app.read_jsonl",return_value=[]), patch("timeguard.app.catalog",return_value=docs), patch.dict("os.environ",RAGFLOW_API_KEY="test",CODEX_BRIDGE_TOKEN="test"):
                        graph = native_graph(http,{"dataset_id":"kb"})
                        with self.assertRaises((ValueError,httpx.HTTPStatusError)):
                            await graph.ainvoke({"question":"AMD FY2022 margin"})
        asyncio.run(run())

