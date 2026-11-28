"""验证事实来源、表格上下文和最终送达，避免只看页码。"""
import copy
import unittest
from timeguard.common import digest
from timeguard.evidence.fact_selection import excerpt, received_facts, validate, prepare


class FactSelectionTest(unittest.TestCase):
    def setUp(self):
        text = "Consolidated income statement USD millions\n| Item | 2022 | 2021 |\n| Revenue | 120 | 100 |\n| Costs | 80 | 70 |\nNote 1: continuing operations."
        self.chunks = [{"id":str(i),"document_id":"a","content":text,"content_with_weight":text} for i in range(4)]
        self.reply = {"selected":[1,2,3,4],"facts":[{"fact":"2021年合并收入","missing_index":1 if i==1 else None,
            "block":i,"lines":[3],"reason":"年份表头和收入行对应",
            "dimensions":{"metric":{"text":"Revenue","line":3},"period":{"text":"2021","line":2},
                          "scope":{"text":"Consolidated","line":1},"unit":{"text":"millions","line":1}}}
            for i in range(1,5)],"unresolved":[]}

    def test_validate_and_receive_requires_context(self):
        parsed = validate(self.reply,self.chunks,["2021 revenue"])
        self.assertEqual(len(received_facts(parsed["facts"],self.chunks)),4)
        incomplete = [{**c,"content":"| Revenue | 120 | 100 |"} for c in self.chunks]
        self.assertEqual(received_facts(parsed["facts"],incomplete),[])

    def test_reject_fabricated_period(self):
        self.reply["facts"][0]["dimensions"]["period"]["text"] = "2020"
        with self.assertRaisesRegex(ValueError,"全部"):
            validate(self.reply,self.chunks,["2021 revenue"])

    def test_unknown_remains_unknown(self):
        self.reply["facts"][0]["dimensions"]["unit"] = None
        parsed = validate(self.reply,self.chunks,["2021 revenue"])
        self.assertIsNone(parsed["facts"][0]["dimensions"]["unit"])

    def test_feedback_must_be_accounted_for(self):
        self.reply["facts"][0]["missing_index"] = None
        with self.assertRaisesRegex(ValueError,"逐项"):
            validate(self.reply,self.chunks,["2021 revenue"])

    def test_partial_bad_fact_does_not_discard_other_facts(self):
        bad = copy.deepcopy(self.reply["facts"][0])
        bad["lines"]=[99]
        self.reply["facts"].append(bad)
        result = validate(self.reply,self.chunks,["2021 revenue"])
        self.assertEqual(len(result["facts"]),4)
        self.assertEqual(len(result["rejected"]),1)

    def test_extract_keeps_header_unit_and_note(self):
        fact = validate(self.reply,self.chunks,["2021 revenue"])["facts"][0]
        part = excerpt(self.chunks[0],[fact])
        for wanted in ("2022", "2021", "USD millions", "Consolidated", "Note 1"):
            self.assertIn(wanted,part["content"])
        self.assertEqual(received_facts([fact],[part]),[0])

    def test_changed_source_or_company_fails(self):
        facts = validate(self.reply,self.chunks,["2021 revenue"])["facts"]
        changed = {**self.chunks[0],"content":"different"}
        with self.assertRaisesRegex(ValueError,"指纹"):
            excerpt(changed,[facts[0]])
        originals = {(c["document_id"],c["id"]):c for c in self.chunks}
        with self.assertRaisesRegex(ValueError,"范围"):
            prepare("revenue",self.chunks,facts,originals,{"b"})

    def test_merged_table_not_flattened(self):
        raw = {**self.chunks[0],"content":'<table><tr><td colspan="2">2021</td></tr></table>'}
        fact = {"source":["a","0"],"source_sha256":digest(raw["content"]),"lines":[1]}
        self.assertEqual(excerpt(raw,[fact]),raw)
