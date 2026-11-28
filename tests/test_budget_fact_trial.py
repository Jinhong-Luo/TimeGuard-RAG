"""预算对照必须保持原块、顺序及公司范围。"""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
from timeguard.evals.trials.budget_fact_trial import frozen_choice, totals


class FrozenBudgetTest(unittest.TestCase):
    def setUp(self):
        self.chunks=[{"id":str(i),"document_id":"d","content":"原表完整正文"} for i in range(4)]
        self.exp=SimpleNamespace(queries={"q":{"question":"问题"}},docs=[],rows={"q":{
            "native_trace":{"evidence_layers":{"answers":[{"snapshot":{"input":{"tools":{"doc_scope":["d"]}}}}]}}}})

    def call(self):
        with patch("timeguard.evals.trials.budget_fact_trial.read_json",return_value={"choices":{"fact_select":self.chunks}}), \
             patch("timeguard.evals.trials.budget_fact_trial.scope",return_value={"companies":["公司"],"document_ids":["d"]}):
            return frozen_choice(self.exp,Path("unused"),"q")

    def test_both_arms_share_exact_frozen_order_and_text(self):
        result=self.call()
        self.assertIs(result["narrow"],self.chunks)
        self.assertIs(result["wide"],self.chunks)
        self.assertEqual([c["id"] for c in result["wide"]],["0","1","2","3"])

    def test_excerpt_is_rejected(self):
        self.chunks[0]["fact_excerpt"]={"lines":[1]}
        with self.assertRaisesRegex(ValueError,"缩短"):
            self.call()

    def test_scope_change_is_rejected(self):
        self.chunks[0]["document_id"]="另一公司或截止日外报告"
        with self.assertRaisesRegex(ValueError,"范围"):
            self.call()


class BudgetCallCountTest(unittest.TestCase):
    def test_excludes_clarification_and_failed_attempt_from_successful_calls(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scope.json").write_text("{}", encoding="utf-8")
            (root / "request.failure-1.json").write_text("{}", encoding="utf-8")
            (root / "request.json").write_text(
                '{"usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}}',
                encoding="utf-8",
            )
            counts = totals(root, "*.json")
        self.assertEqual(counts["请求"], 1)
        self.assertEqual(counts["用量缺失"], 0)
        self.assertEqual(counts["total_tokens"], 10)
