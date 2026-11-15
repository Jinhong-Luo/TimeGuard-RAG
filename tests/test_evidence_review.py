"""证据审查不得接受越界、伪造和无依据的块。"""
import unittest
from timeguard.evidence_review import catalog,inspect_once,validated
from timeguard.final_snapshot import fingerprint


class ReviewTest(unittest.TestCase):
    def setUp(self):
        self.v = {"tools":{"doc_scope":["甲"]},"state":{"kbinfos":{"chunks":[
            {"doc_id":"甲","chunk_id":"行一","content":"2022 年销售额为 10 万元。"}]}}}
        self.snap = {"input":self.v,"sha256":fingerprint(self.v)}
        self.entries = catalog(self.snap)
        self.key = next(iter(self.entries))

    def test_quote_and_stable_id(self):
        reply = {"selected":[{"id":self.key,"fact":"销售额","quote":"10 万元"}],"missing":[]}
        self.assertEqual(validated(reply,self.entries),[self.key])
        reply["selected"][0]["quote"] = "20 万元"
        with self.assertRaises(ValueError):
            validated(reply,self.entries)
        reply["selected"][0]["id"] = "外部编号"
        with self.assertRaises(ValueError):
            validated(reply,self.entries)

    def test_timeout_only_once(self):
        called = []
        def timeout(*args):
            called.append(True)
            raise TimeoutError()
        record = inspect_once({"base_url":"","token":""},"销售额",self.snap,timeout)
        self.assertTrue(record["fallback"])
        self.assertEqual(len(called),1)
        self.assertIsNone(record["usage"])

    def test_wrong_structure_falls_back(self):
        def invalid(*args):
            return {"choices":[{"message":{"content":"[]"}}]}
        record = inspect_once({"base_url":"","token":""},"销售额",self.snap,invalid)
        self.assertTrue(record["fallback"])

    def test_scope_precedes_call(self):
        self.v["tools"]["doc_scope"] = ["乙"]
        self.snap["sha256"] = fingerprint(self.v)
        with self.assertRaises(ValueError):
            catalog(self.snap)
