"""检查证据排序范围、重复段落和互补指标。"""
import unittest

from timeguard.retrieval.page_select import select


class SelectionTest(unittest.TestCase):
    def test_complements_and_duplicates(self):
        def chunk(i,text,page):
            return {"chunk_id":i, "doc_id":"甲", "content":text, "positions":[[page]]}
        chunks = [chunk("1","Consolidated net revenue 2022",1),chunk("2","Consolidated net revenue 2022",1),
                  chunk("3","Revenue 2021",2),chunk("4","Other information",3)]
        selected = select("Consolidated revenue 2021 and 2022",chunks,["甲"],2)
        self.assertEqual([c["chunk_id"] for c in selected],["1","3"])

    def test_scope_failure(self):
        with self.assertRaises(ValueError):
            select("问题",[{"doc_id":"乙"}],["甲"])

    def test_unknown_page_and_stable_order(self):
        chunks = [{"doc_id":"甲","content":str(i)} for i in range(20)]
        self.assertEqual(select("问题",chunks,["甲"]),chunks[:12])
