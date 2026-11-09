"""检查原生矩阵的完整分母与结果来源边界。"""
import tempfile
from pathlib import Path
import unittest
from timeguard.common import write_json,write_jsonl
from timeguard.native_matrix import summarize_run


class NativeMatrixTests(unittest.TestCase):
    def test_missing_generation_and_judgement_stay_in_plan(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp:
            self.assertTrue(Path(temp).resolve().is_relative_to(Path.cwd().resolve()))
            root=Path(temp)
            write_json(root/"manifest.json",{"identity":{"query_ids":["q1","q2"]}})
            write_jsonl(root/"results.jsonl",[{"id":"q1","status":"request_error","elapsed_seconds":1}])
            result=summarize_run(root)
            self.assertEqual(result["planned"],2)
            self.assertEqual(result["missing"],1)
            self.assertEqual(result["judge_missing"],2)
            self.assertIsNone(result["full_accuracy"])
            self.assertEqual(result["correct_fraction_lower_bound"],0)

    def test_duplicate_results_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp:
            self.assertTrue(Path(temp).resolve().is_relative_to(Path.cwd().resolve()))
            root=Path(temp)
            write_json(root/"manifest.json",{"identity":{"query_ids":["q1"]}})
            write_jsonl(root/"results.jsonl",[{"id":"q1"},{"id":"q1"}])
            with self.assertRaises(ValueError):summarize_run(root)
