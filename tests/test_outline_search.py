"""目录页码、范围回退和自主选块的回归。"""
import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest.mock import Mock

from timeguard.evals.trials.four_blocks import pick
from timeguard.retrieval.outline_search import OutlineSearch, headings, rank_sections


class OutlineTests(unittest.TestCase):
    def test_physical_section_boundaries(self):
        pages = "Table of Contents\nItem 1. Business     25\fItem 1. Business\nOther text\fmore\fConsolidated Balance Sheets\f"
        sections = headings(pages)
        self.assertEqual(sections,[("Item 1. Business",2,3),("Consolidated Balance Sheets",4,4)])
        self.assertEqual(rank_sections("net receivables",{"a":sections})[0][2:4],(4,4))

    def test_scope_and_fallback(self):
        reader = OutlineSearch(Mock(),"ds",[{"id":"a"},{"id":"b"}])
        reader.read_outline = Mock(return_value=[])
        fallback = Mock(return_value=[])
        audit = {}
        with self.assertRaises(ValueError):
            reader.search("revenue",[],fallback,audit)
        reader.search("revenue",["a"],fallback,audit)
        reader.read_outline.assert_called_once_with("a")
        fallback.assert_called_once_with("revenue")
        self.assertEqual(audit["原生补搜次数"],1)

    def test_pdf_identity(self):
        with TemporaryDirectory() as tmp:
            Path(tmp,"a.pdf").write_bytes(b"123")
            reader = OutlineSearch(Mock(),"ds",[{"id":"a","local_path":"a.pdf","pdf_sha256":"wrong"}],Path(tmp))
            with self.assertRaises(ValueError):
                reader.search("revenue",["a"],Mock(),{})

    def test_adaptive_limits(self):
        for n in (1,3,4,8,12):
            ids = list(range(1,n+1))
            self.assertEqual(pick({"selected":ids},20,adaptive=True),ids)
        for ids in ([],[True],[21],[1,1],list(range(1,14))):
            with self.assertRaises(ValueError):
                pick({"selected":ids},20,adaptive=True)
        with self.assertRaises(ValueError):
            pick({"selected":[1]},20)
