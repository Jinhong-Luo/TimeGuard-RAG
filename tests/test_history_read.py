"""验证按缺项回读时的范围、指纹、冲突和失败处理。"""
import unittest
from unittest.mock import Mock, patch

from timeguard.common import digest
from timeguard.retrieval.history_read import recover, recover_row


class HistoryReadTest(unittest.TestCase):
    question = "What was FY2022 capital expenditure in USD millions?"

    def chunk(self, ident="旧表", value=17, doc="甲"):
        text = ("Consolidated statement (USD millions)\n| Item | 2021 | 2022 |\n"
                f"| Capital expenditures | (15) | ({value}) |")
        return {"id": ident, "document_id": doc, "content": text}

    def run_case(self, old, visible=(), pool=(), reader=None, allow_tables=False):
        rows = {c["id"]: c for c in [*old, *visible]}
        calls = [{"chunks": [{**c, "content_sha256": digest(c["content"])}]} for c in old]
        return recover(self.question, visible, pool, calls, {"甲"}, reader or (lambda doc, ident: rows[ident]), allow_tables)

    def test_prior_search_restores_exact_original_once(self):
        c = self.chunk()
        r = self.run_case([c, c, self.chunk("其他公司", doc="乙")])
        self.assertEqual(r["chunks"], [c])
        self.assertEqual(r["audit"]["补入"], [("资本开支", 2022)])
        self.assertEqual(r["audit"]["读取"], 1)

    def test_present_fact_needs_no_history(self):
        r = self.run_case([self.chunk("早先")], [self.chunk("已送达")])
        self.assertEqual((r["chunks"], r["audit"]["缺项"], r["audit"]["读取"]), ([], [], 1))

    def test_final_pool_is_excluded(self):
        c = self.chunk()
        self.assertEqual(self.run_case([c], pool=[c])["audit"]["读取"], 0)

    def test_conflicting_amounts_are_not_recovered(self):
        r = self.run_case([self.chunk(), self.chunk("另表", 24)])
        self.assertEqual(r["chunks"], [])
        self.assertEqual(r["audit"]["冲突"], [("资本开支", 2022)])

    def test_source_change_and_missing_data(self):
        c = self.chunk()
        changed = self.run_case([c], reader=lambda *_: self.chunk(value=99))
        self.assertEqual((changed["chunks"], changed["audit"]["指纹不符"]), ([], 1))
        failed = self.run_case([c], reader=lambda *_: None)
        self.assertEqual((failed["chunks"], failed["audit"]["读取失败"]), ([], 1))

    def test_mismatched_source_is_rejected(self):
        with self.assertRaises(ValueError):
            self.run_case([self.chunk()], reader=lambda *_: self.chunk(doc="乙"))

    def test_acronym_must_spell_out_the_label(self):
        c = self.chunk()
        label = "Purchases of property, plant and equipment"
        for suffix, accepted in [(" (PP&E)", True), (" (domestic)", False), (" (net)", False)]:
            old = {**c, "content": c["content"].replace("Capital expenditures", label + suffix)}
            with self.subTest(suffix=suffix):
                self.assertEqual(bool(self.run_case([old], allow_tables=True)["chunks"]), accepted)

    def test_related_table_is_not_a_verified_amount(self):
        text = ("Consolidated cash flows <table><tr><td>Item</td><td>FY 2022</td></tr>"
                "<tr><td>Capital expenditures</td><td>(25)</td></tr></table>")
        old = {**self.chunk(), "content": text}
        r = self.run_case([old], allow_tables=True)
        self.assertEqual((r["chunks"], r["audit"]["补入"]), ([old], []))
        self.assertEqual(r["audit"]["待核原表"], [("资本开支", 2022)])
        for altered in (text.replace("2022","2021"), text.replace("Consolidated","Segment")):
            self.assertFalse(self.run_case([{**old, "content": altered}], allow_tables=True)["chunks"])

    def test_snapshot_from_another_index_or_outside_cutoff_is_rejected(self):
        answer = {"snapshot":{"input":{"identity":{"dataset_id":"资料库"},
                                       "tools":{"doc_scope":["截止日以后披露的报告"]}}},
                  "requests":[{"status":200}]}
        row = {"question": self.question, "as_of":"2022-12-31",
               "native_trace":{"evidence_layers":{"answers":[answer]}}}
        client = Mock()
        with self.assertRaises(ValueError):
            recover_row(row, client, [], "另一个资料库")
        with patch("timeguard.retrieval.history_read.scope", return_value={"companies":["甲"], "document_ids":[]}):
            with self.assertRaises(ValueError):
                recover_row(row, client, [], "资料库")
        client.request.assert_not_called()
