"""覆盖真实回放中出现的非连续摘录、截表与来源变更。"""
import unittest

from timeguard.common import digest
from timeguard.evidence.citation_spans import recover, visible_sections


class CitationSpansTests(unittest.TestCase):
    def setUp(self):
        self.anchor = {"id": "c", "document_id": "d", "document_name": "报表.pdf"}
        self.layer = {"packet": [self.anchor], "block_hashes": ["h"]}

    def test_discontinuous_with_repeated_tail(self):
        source = "2021 年营业收入为 100 百万元。其余说明。2022 年营业收入为 120 百万元。"
        visible = "2022 年营业收入为 120 百万元。\n\n2021 年营业收入为 100 百万元。"
        chunk = {**self.anchor, "block_sha256": "h", "content": visible, "source_sha256": digest(source)}
        raw = {"doc_id": "d", "content_with_weight": source}
        def check():
            return recover(1, self.layer, ("报表.pdf", visible), chunk, raw, set())
        self.assertEqual(len(check()["spans"]), 2)
        raw["content_with_weight"] += "修订"
        self.assertIsNone(check())

    def test_partial_table_excludes_cut_row(self):
        source = "<table><caption>年度经营情况</caption><tr><th>2024 年利润</th></tr><tr><td>450</td></tr></table>"
        exposed = source.partition("450")[0] + "4"
        message = "问题\nEvidence:\nID: 1\n├── Title: 报表.pdf\n└── Content:\n" + exposed
        section = visible_sections([{"role": "user", "content": message}])[1]
        raw = {"doc_id": "d", "content_with_weight": source}
        result = recover(1, self.layer, section, None, raw, {digest(source)})
        self.assertTrue(result["content"].endswith("</th></tr>"))
        self.assertIsNone(recover(1, self.layer, section, None, raw, set()))

    def test_ambiguous_number(self):
        header = "ID: 1\n├── Title: 甲\n└── Content:\n年度数据\n"
        self.assertEqual(visible_sections([{"role": "user", "content": "\nEvidence:\n" + header * 2}]), {})
