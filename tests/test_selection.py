"""阻止小样本、缺档、错误运行或变动索引产生最佳档位结论。"""

import copy
import unittest

from timeguard.common import digest
from timeguard.select import choose


class SelectionTests(unittest.TestCase):
    def records(self):
        ids = ["q1", "q2"]
        records = []
        for effort, seconds in (("low", 2.0), ("high", 4.0)):
            identity = {"config": {"model": "gpt-6.1-sol", "efforts_to_probe": ["low", "high"], "retrieval": {"page_size": 6}},
                        "query_ids": ids, "split": "dev", "effort": effort, "retrieval_only": False,
                        "scope": "完整分组评测", "full_split_count": 2, "full_split_query_ids_sha256": digest(ids),
                        "transport_proof": {"verified_efforts": [effort]}}
            records.append({"run": effort, "manifest": {"identity": identity, "index_snapshot": [{"id": "doc1", "chunk_count": 20}]},
                            "summary": {"completed": 2, "denominator": 2, "reviewed": 2, "answer_accuracy": 1.0,
                                        "mean_seconds": seconds, "status_counts": {"ok": 2}}})
        return records

    def test_selects_faster_complete_candidate(self):
        self.assertEqual(choose(self.records())["chosen_effort"], "low")

    def test_rejects_single_candidate(self):
        with self.assertRaises(ValueError):
            choose(self.records()[:1])

    def test_rejects_smoke_and_missing_full_split_proof(self):
        for field, value in (("scope", "小样本连通测试"), ("full_split_count", 3), ("full_split_query_ids_sha256", "wrong")):
            records = self.records()
            for record in records:
                record["manifest"]["identity"][field] = value
            with self.assertRaises(ValueError):
                choose(records)

    def test_rejects_missing_declared_effort_and_duplicate(self):
        records = self.records()
        for record in records:
            record["manifest"]["identity"]["config"]["efforts_to_probe"].append("ultra")
        with self.assertRaises(ValueError):
            choose(records)
        duplicate = self.records()
        duplicate.append(copy.deepcopy(duplicate[0]))
        with self.assertRaises(ValueError):
            choose(duplicate)

    def test_rejects_changed_index_and_incomplete_review(self):
        records = self.records()
        records[1]["manifest"]["index_snapshot"][0]["chunk_count"] = 21
        with self.assertRaises(ValueError):
            choose(records)
        records = self.records()
        records[1]["summary"]["reviewed"] = 1
        with self.assertRaises(ValueError):
            choose(records)

    def test_only_exact_model_effort_alias_is_normalized(self):
        records = self.records()
        for record in records:
            identity = record["manifest"]["identity"]
            identity["config"]["model"] += "@" + identity["effort"]
        self.assertEqual(choose(records)["chosen_effort"], "low")
        records[1]["manifest"]["identity"]["config"]["model"] = "gpt-6.1-sol@medium"
        with self.assertRaises(ValueError):
            choose(records)
        records = self.records()
        records[1]["manifest"]["identity"]["config"]["model"] = "different-provider/model@high"
        with self.assertRaises(ValueError):
            choose(records)

    def test_rejects_request_errors_and_invalid_duration(self):
        for field, value in (("status_counts", {"request_error": 2}), ("mean_seconds", None), ("mean_seconds", float("nan"))):
            records = self.records()
            records[1]["summary"][field] = value
            with self.assertRaises(ValueError):
                choose(records)

    def test_ragflow_composite_alias_preserves_instance_and_provider(self):
        records = self.records()
        for record in records:
            identity = record["manifest"]["identity"]
            identity["config"]["model"] = "gpt-6.1-sol@" + identity["effort"] + "@codex_local@OpenAI-API-compatible"
        self.assertEqual(choose(records)["chosen_effort"], "low")
        records[1]["manifest"]["identity"]["config"]["model"] = "gpt-6.1-sol@high@different_instance@OpenAI-API-compatible"
        with self.assertRaises(ValueError):
            choose(records)

    def test_rejects_test_set(self):
        records = self.records()
        for record in records:
            record["manifest"]["identity"]["split"] = "test"
        with self.assertRaises(ValueError):
            choose(records)


if __name__ == "__main__":
    unittest.main()
