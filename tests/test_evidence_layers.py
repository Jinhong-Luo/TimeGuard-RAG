"""集中覆盖多轮合并、跨报告同页、截断及诊断证据隔离。"""
import unittest

from timeguard.common import digest
from timeguard.evidence_layers import match_blocks, visible_blocks
from timeguard.layer_report import assess, layers


def chunk(doc, page, text):
    return {"id": f"{doc}-{page}", "document_id": doc, "document_name": doc + ".pdf",
            "positions": [[page]], "content": text, "source_check": "intact", "content_sha256": digest(text)}


def example(calls, packet, received):
    return {"status": "ok", "native_trace": {"evidence_layers": {"version": 2,
            "retrievals": [{"chunks": c} for c in calls], "answers": [{"selected": packet, "packet": packet,
            "requests": [{"status": 200, "complete_chunks": received}]}]}}}


class LayerTests(unittest.TestCase):
    def test_rounds_scope_and_loss(self):
        a, b = chunk("甲", 3, "原文甲"), chunk("乙", 3, "原文乙")
        row = example([[a], [b]], [b], [b])
        result = assess(row, {"answer_correct": False}, {"answer_correct": True},
                        {"id": "题一", "evidence": [{"doc_name": "甲", "evidence_page_num": 2}]})
        self.assertEqual(result["hit_pages"]["candidates"], [("甲", 3)])
        self.assertEqual(result["hit_pages"]["received"], [])
        self.assertEqual(result["next_check"], "先查证据选择或预算")

    def test_full_block_only(self):
        a, b = chunk("甲", 3, "第一行"), chunk("甲", 4, "第二行")
        render = lambda c, i: f"{i} {c['content']}" if c.get("content") else None
        blocks = ["1 第一行", "2 第二行"]
        mapped = match_blocks([{}, a, b], blocks, render)
        sent = [{"role": "user", "content": "摘要 2 第二行\nEvidence:\n1 第一行\n2 第二"}]
        self.assertEqual([c["id"] for c in visible_blocks(mapped, sent)], [a["id"]])
        self.assertIsNone(match_blocks([a, b], ["9 第一行"], render))

    def test_changed_body_is_unverified(self):
        a = chunk("甲", 2, "完整原文")
        row = example([[a]], [a], [{**a, "content": "完整", "source_check": "edited"}])
        self.assertEqual(layers(row)["received"], ({("甲", 2)}, True))
        self.assertEqual(layers(row)["intact"], (set(), True))

    def test_historical_absence_and_empty_are_distinct(self):
        self.assertEqual(layers({"status": "ok"})["candidates"], (set(), False))
        row = example([], [], [])
        self.assertEqual(layers(row)["candidates"], (set(), True))
        row["native_trace"]["evidence_layers"]["answers"][0]["requests"] = [{"status": 502}]
        self.assertEqual(layers(row)["received"], (set(), False))

    def test_multiple_compositions_not_merged(self):
        a = chunk("甲", 2, "正文")
        row = example([[a]], [a], [a])
        row["native_trace"]["evidence_layers"]["answers"] *= 2
        self.assertEqual(layers(row)["received"], (set(), False))

    def test_partial_pages_do_not_become_full_coverage(self):
        a = chunk("甲", 1, "正文")
        row = example([[a]], [a], [a])
        result = assess(row, {"answer_correct": False}, {"answer_correct": False},
                        {"id": "题一", "evidence": [{"doc_name": "甲", "evidence_page_num": n} for n in (0, 1)]})
        self.assertEqual(result["next_check"], "混合原因待复核")

    def test_recall_and_financial_review_need_different_evidence(self):
        a = chunk("甲", 2, "表格")
        gold = {"id": "题一", "evidence": [{"doc_name": "甲", "evidence_page_num": 1}]}
        wrong = {"answer_correct": False}
        positive = {"answer_correct": True}
        self.assertEqual(assess(example([[]], [], []), wrong, positive, gold)["next_check"], "先查页级召回")
        self.assertEqual(assess(example([[a]], [a], [a]), wrong, wrong, gold)["next_check"], "复查取数、题意或金标")
        self.assertEqual(assess(example([[a]], [a], [a]), wrong, {}, gold)["next_check"], "裁判未知")

    def test_uncaptured_direct_read_prevents_definite_recall_miss(self):
        a, b = chunk("甲", 1, "正文"), chunk("乙", 2, "表格")
        row = example([[a]], [b], [b])
        self.assertFalse(layers(row)["candidates"][1])
