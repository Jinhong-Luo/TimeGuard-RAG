"""最终节点快照的顺序、隔离、指纹和原版回放检查。"""
import unittest
from types import SimpleNamespace

from timeguard.final_snapshot import freeze, replay
from timeguard.snapshot_history import recover
from timeguard.common import digest
from timeguard.judge import citation_context


class SnapshotTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.model = SimpleNamespace(max_length=32768)
        self.tools = SimpleNamespace(chat_mdl=self.model, doc_scope=["报告甲"])
        self.state = {"question": "营业收入是多少", "slot_evidence": {"乙": {}, "甲": {}},
                      "kbinfos": {"chunks": [{"content": "第一块"}], "pre_summary": "研究摘要"},
                      "gold": "仅评测可见"}
        self.snapshot = freeze(self.state, self.tools, {}, {})

    async def test_replay_order_and_isolation(self):
        self.state["kbinfos"]["chunks"].clear()
        async def compose(state, tools, queue, conf):
            self.assertEqual(list(state["slot_evidence"]), ["乙", "甲"])
            self.assertNotIn("gold", state)
            self.assertEqual(tools.doc_scope, ["报告甲"])
            return state["kbinfos"]["chunks"].pop()["content"]
        for _ in range(2):
            self.assertEqual(await replay(self.snapshot, compose, self.model), "第一块")

    async def test_reject_tamper(self):
        self.snapshot["input"]["state"]["slot_evidence"] = {"甲": {}, "乙": {}}
        with self.assertRaises(ValueError):
            await replay(self.snapshot, None, self.model)

    def test_history_requires_all_three_keys(self):
        match = {"id": "块", "document_id": "甲", "content": "原文"}
        chunks = [{"id": "块", "document_id": doc, "content_sha256": digest(body)}
                  for doc, body in [("甲", "原文"), ("乙", "原文"), ("甲", "另一版")]]
        trace = {"retrievals": [{"call": 1, "channel": "检索", "chunks": chunks}], "answers": []}
        record = {"retrieved_chunks": [match], "native_trace": {"evidence_layers": trace}}
        self.assertEqual([c["content"] for c in recover(record)], ["原文", None, None])

    def test_native_numbers_follow_received_blocks(self):
        chunk = {"block_sha256": "指纹二", "id": "块二", "document_id": "报告", "content": "金额"}
        record = {"snapshot": {}, "block_hashes": ["指纹一", "指纹二"],
                  "requests": [{"status": 200, "complete_chunks": [chunk]}]}
        row = {"answer": "[ID:1][ID:2]", "reference": {"chunks": [chunk, chunk, chunk]},
               "native_trace": {"citation_origin": "model", "evidence_layers": {"answers": [record]}}}
        context = citation_context(row)
        self.assertEqual(context["unmapped_indexes"], [1])
        self.assertEqual(context["citations"][0]["chunk_id"], "块二")
