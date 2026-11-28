import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

from timeguard.research.live_finance import FEEDBACK, enhance


def chunk(i, document="d", content="原文说明第 2021 财年。"):
    return {"id": str(i), "document_id": document, "document_name": "3M 年报", "positions": [[i, 0, 1, 0, 1]],
            "content": content}


def selection(ids, resolved):
    return {"selected": ids, "facts": [
        {"block": i, "fact": "关于原文的具体事实", "reason": "原文提供必要说明", "lines": [1],
         "missing_index": 1 if resolved and i == ids[-1] else None,
         "dimensions": dict.fromkeys(("metric", "period", "scope", "unit"))} for i in ids],
        "unresolved": [] if resolved else [1]}


class LiveFinanceTest(unittest.IsolatedAsyncioTestCase):
    async def execute(self, replies, extra=(), read=lambda c: c, cut=False):
        originals = [chunk(i) for i in range(1, 5)]
        model = SimpleNamespace(async_chat=AsyncMock(side_effect=[json.dumps(r) for r in replies]))
        retrieve = AsyncMock(side_effect=extra if isinstance(extra, Exception) else None,
                             return_value={"chunks": extra})
        tools = SimpleNamespace(doc_scope=["d"], chat_mdl=model, retrieve=retrieve)

        async def preview(chosen, instruction):
            return originals if chosen is None else chosen[:1] if cut else chosen

        trace = {}
        result = await enhance("What changed for 3M in FY2021?", {
            "document_ids": ["d"], "years": [2021], "quarters": []}, {},
            {"kbinfos": {"chunks": originals}}, tools, lambda info: info["chunks"], read, preview, trace)
        return result, trace, model.async_chat, retrieve

    async def test_f07_sufficient_originals_stop_before_selection_and_search(self):
        result, trace, model, retrieve = await self.execute([{"missing": []}])
        self.assertEqual(result, (None, ""))
        self.assertEqual(trace["selected_count"], 4)
        self.assertEqual(model.await_args.args[0], FEEDBACK)
        retrieve.assert_not_awaited()

    async def test_f08_existing_sources_are_read_and_used(self):
        read_ids = []

        def read(row):
            read_ids.append(row["id"])
            return row

        result, trace, _, retrieve = await self.execute([{"missing": ["2019 年财报原因"]},
                                                       selection([1, 2, 3, 4], True)], read=read)
        self.assertEqual(read_ids, ["1", "2", "3", "4"] * 2)
        self.assertEqual(trace["path"], "事实筛选")
        self.assertEqual(len(result[0]), 4)
        retrieve.assert_not_awaited()

    async def test_selector_gets_index_text_when_research_pass_has_a_summary(self):
        def read(row):
            return {**row, "content": "索引原文包含 2021 财年及所需口径。"}

        result, trace, model, _ = await self.execute([{"missing": ["2021 年报告口径"]},
                                                    selection([1, 2, 3, 4], True)], read=read)
        self.assertEqual(trace["rehydrated_candidates"], 4)
        request = json.loads(model.await_args_list[-1].args[1][0]["content"])
        self.assertEqual(request["候选"][0]["原文行"][0]["text"], result[0][0]["content"])

    async def test_f09_supplement_is_once_and_same_scope(self):
        result, trace, _, retrieve = await self.execute([
            {"missing": ["cash flows"]}, selection([1, 2, 3, 4], False), selection([1, 2, 3, 5], True)],
            [chunk(5)])
        retrieve.assert_awaited_once_with("cash flows", doc_scope=["d"], using_embedding=True)
        self.assertEqual(trace["supplement_calls"], 1)
        self.assertIn("5", [c["id"] for c in result[0]])

    async def test_f10_supplement_failure_falls_back(self):
        for extra in ([], [chunk(5, "foreign")], TimeoutError()):
            with self.subTest(extra=extra):
                result, trace, _, retrieve = await self.execute([
                    {"missing": ["cash flows"]}, selection([1, 2, 3, 4], False)], extra)
                self.assertEqual(result, (None, ""))
                self.assertEqual(trace["path"], "原生选块")
                retrieve.assert_awaited_once()

    async def test_f11_trimmed_evidence_falls_back_before_final_answer(self):
        result, trace, _, retrieve = await self.execute([
            {"missing": ["cash flows"]}, selection([1, 2, 3, 4], True)], cut=True)
        self.assertEqual(result, (None, ""))
        self.assertIn("无法完整送达", trace["reason"])
        retrieve.assert_not_awaited()
