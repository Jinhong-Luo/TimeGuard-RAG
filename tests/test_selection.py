"""阻止小样本、缺档、错误运行或变动索引产生最佳档位结论。"""

import copy
import json
from pathlib import Path
import unittest

from timeguard import judge
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


class ModelSelectionTests(unittest.TestCase):
    def records(self):
        ids = [f"q{i}" for i in range(39)]
        efforts = ["low", "medium", "high", "xhigh", "max", "ultra"]
        records = []
        for offset, effort in enumerate(efforts):
            seconds = float(offset + 1)
            identity = {"config": {"model": f"gpt-6.1-sol@{effort}@local@OpenAI-API-compatible", "efforts_to_probe": efforts},
                        "query_ids": ids, "split": "dev", "effort": effort, "retrieval_only": False,
                        "scope": "完整分组评测", "full_split_count": 39, "full_split_query_ids_sha256": digest(ids)}
            manifest = {"identity": identity, "index_snapshot": [{"id": "doc", "chunk_count": 20}]}
            generated = [{"id": ident, "status": "ok", "answer": "事实答案[ID:0]", "elapsed_seconds": seconds,
                          "reference": {"chunks": [{"content": "事实原文"}]}} for ident in ids]
            sources = {"manifest": digest(manifest), "results": digest(generated)}
            judge_identity = {"version": judge.JUDGE_VERSION, "judge_type": "model", "model": judge.MODEL,
                              "effort": judge.EFFORT, "transport": judge.TRANSPORT, "expected_ids": ids,
                              "generation_manifest_sha256": sources["manifest"], "generation_results_sha256": sources["results"],
                              "gold_sha256": digest("固定金标"), "prompt_sha256": digest(judge.INSTRUCTIONS),
                              "code_sha256": digest(Path(judge.__file__).read_bytes()), "schema": judge.SCHEMA}
            signature = digest(judge_identity)
            reviews = [{"id": ident, "judge_type": "model", "judge_status": "ok", "judge_run_signature": signature,
                        "judge_model": judge.MODEL, "judge_effort": judge.EFFORT, "answer_correct": True,
                        "numeric_and_scope_correct": True, "citation_support": True, "has_factual_claims": True,
                        "reviewer": "固定模型裁判", "judgement_reason": "材料一致",
                        "raw_judgement": json.dumps({"answer_correct": True, "numeric_and_scope_correct": True,
                                                     "citation_support": True, "has_factual_claims": True,
                                                     "judgement_reason": "材料一致", "failure_category": "未发现错误"}),
                        "citation_mapping": {key: value for key, value in judge.citation_context(generated[0]).items()
                                             if key != "citations"},
                        "bridge_metadata": {"transport": judge.TRANSPORT, "api_equivalent": False,
                                            "actual_model": judge.MODEL, "reasoning_effort": judge.EFFORT}} for ident in ids]
            records.append({"run": effort, "manifest": manifest,
                            "summary": {"completed": 39, "denominator": 39, "reviewed": 0, "answer_accuracy": None,
                                        "mean_seconds": seconds, "status_counts": {"ok": 39}},
                            "source_hashes": sources, "generated_rows": generated,
                            "judge_manifest": {"identity": judge_identity, "signature": signature}, "model_reviews": reviews})
        return copy.deepcopy(records)

    def test_model_mode_is_explicit_and_provisional(self):
        records = self.records()
        with self.assertRaises(ValueError):
            choose(records)
        result = choose(records, review_source="model")
        self.assertEqual(result["chosen_effort"], "low")
        self.assertEqual(result["review_source"], "model")
        self.assertIn("未完成人工抽查", result["selection_status"])
        self.assertIsNone(result["human_spot_check"])
        self.assertNotIn("generation_results_sha256", result["judge_configuration"])
        self.assertEqual(len({r["judge_manifest"]["signature"] for r in records}), 6)

    def test_unknown_secondary_judgements_keep_unknown_and_coverage(self):
        records = self.records()
        records[0]["model_reviews"][0]["citation_support"] = None
        records[0]["model_reviews"][1]["numeric_and_scope_correct"] = None
        for row in records[0]["model_reviews"][:2]:
            raw = json.loads(row["raw_judgement"])
            raw.update({key: row[key] for key in judge.FIELDS})
            row["raw_judgement"] = json.dumps(raw)
        result = choose(records, review_source="model")
        candidate = result["candidates"][0]
        self.assertIsNone(candidate["model_citation_support_rate"])
        self.assertEqual(candidate["model_citation_support_rate_unknown_count"], 1)
        self.assertEqual(candidate["model_citation_support_rate_known_count"], 38)
        self.assertAlmostEqual(candidate["model_citation_support_rate_coverage"], 38 / 39)
        self.assertEqual(candidate["model_numeric_and_scope_accuracy_unknown_count"], 1)

    def test_rejects_missing_effort_even_if_declaration_shrinks(self):
        records = self.records()[:-1]
        for record in records:
            record["manifest"]["identity"]["config"]["efforts_to_probe"] = ["low", "medium", "high", "xhigh", "max"]
        with self.assertRaises(ValueError):
            choose(records, review_source="model")

    def test_rejects_partial_unknown_failed_and_duplicate_reviews(self):
        for change in ("missing", "unknown", "failed", "duplicate"):
            records = self.records()
            rows = records[0]["model_reviews"]
            if change == "missing":
                rows.pop()
            elif change == "unknown":
                rows[0]["answer_correct"] = None
            elif change == "failed":
                rows[0]["judge_status"] = "failed"
            else:
                rows[-1] = copy.deepcopy(rows[0])
            with self.subTest(change=change), self.assertRaises(ValueError):
                choose(records, review_source="model")

    def test_rejects_changed_judge_identity_and_wrong_actual_effort(self):
        for change in ("prompt", "gold", "actual_effort", "run_signature"):
            records = self.records()
            record = records[0]
            if change in ("prompt", "gold"):
                record["judge_manifest"]["identity"][change + "_sha256"] = digest("不同配置")
                signature = digest(record["judge_manifest"]["identity"])
                record["judge_manifest"]["signature"] = signature
                for row in record["model_reviews"]:
                    row["judge_run_signature"] = signature
            elif change == "actual_effort":
                record["model_reviews"][0]["bridge_metadata"]["reasoning_effort"] = "high"
            else:
                record["model_reviews"][0]["judge_run_signature"] = "另一个运行"
            with self.subTest(change=change), self.assertRaises(ValueError):
                choose(records, review_source="model")

    def test_rejects_reviews_bound_to_other_generation(self):
        for source in ("manifest", "results"):
            records = self.records()
            records[0]["source_hashes"][source] = digest("其他生成记录")
            with self.subTest(source=source), self.assertRaises(ValueError):
                choose(records, review_source="model")

    def test_rejects_missing_raw_duration_and_inconsistent_summary(self):
        for change in ("duration", "summary", "failure"):
            records = self.records()
            if change == "duration":
                records[0]["generated_rows"][0]["elapsed_seconds"] = None
            elif change == "summary":
                records[0]["summary"]["mean_seconds"] = 0.1
            else:
                records[0]["generated_rows"][0]["status"] = "request_error"
            with self.subTest(change=change), self.assertRaises(ValueError):
                choose(records, review_source="model")

    def test_quality_is_recomputed_from_complete_reviews(self):
        records = self.records()
        records[0]["model_reviews"][0]["answer_correct"] = False
        raw = json.loads(records[0]["model_reviews"][0]["raw_judgement"])
        raw["answer_correct"] = False
        records[0]["model_reviews"][0]["raw_judgement"] = json.dumps(raw)
        records[0]["summary"]["model_answer_accuracy"] = 1.0
        result = choose(records, review_source="model")
        self.assertEqual(result["chosen_effort"], "medium")
        self.assertAlmostEqual(result["candidates"][0]["model_answer_accuracy"], 38 / 39)

    def test_rejects_final_fields_inconsistent_with_raw_judgement(self):
        records = self.records()
        records[0]["model_reviews"][0]["answer_correct"] = False
        with self.assertRaises(ValueError):
            choose(records, review_source="model")

    def test_rejects_different_generation_or_judge_concurrency(self):
        for target in ("generation", "judge"):
            records = self.records()
            for offset, record in enumerate(records):
                if target == "generation":
                    record["manifest"]["identity"]["concurrency"] = 1 if offset == 0 else 2
                else:
                    identity = record["judge_manifest"]["identity"]
                    identity["concurrency"] = 1 if offset == 0 else 2
                    signature = digest(identity)
                    record["judge_manifest"]["signature"] = signature
                    for row in record["model_reviews"]:
                        row["judge_run_signature"] = signature
            with self.subTest(target=target), self.assertRaises(ValueError):
                choose(records, review_source="model")


if __name__ == "__main__":
    unittest.main()
