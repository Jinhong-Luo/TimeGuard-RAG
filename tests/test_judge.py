"""验证离线模型裁判隔离、失败保留与模型身份约束，不调用真实模型。"""

import json
import unittest

from timeguard.judge import EFFORT, MODEL, TRANSPORT, build_request, citation_context, judge_one, parse_judgement, unique_rows


class JudgeTests(unittest.TestCase):
    def setUp(self):
        self.gold = {"id": "q1", "question": "收入是多少？", "answer": "100万元", "evidence": [{"evidence_text": "收入100万元", "doc_name": "A", "evidence_page_num": 3}]}
        self.generated = {"id": "q1", "status": "ok", "answer": "收入100万元[ID:0]", "reference": {"chunks": [{"content": "收入100万元"}, {"content": "未引用的候选，不能用来补证据"}]}, "retrieved_chunks": [{"content": "这是不同的诊断检索，不得替代真实引用"}]}
        self.judgement = {"answer_correct": True, "numeric_and_scope_correct": True, "citation_support": True,
                          "has_factual_claims": True,
                          "judgement_reason": "数值、单位及实际引用一致。", "failure_category": "未发现错误"}

    def response(self):
        return {"model": MODEL, "bridge_metadata": {"transport": TRANSPORT, "reasoning_effort": EFFORT, "api_equivalent": False, "actual_model": MODEL},
                "choices": [{"message": {"content": json.dumps(self.judgement, ensure_ascii=False)}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}}

    def test_only_actual_citations_enter_judge(self):
        request = build_request(self.gold, self.generated)
        material = json.loads(request["messages"][1]["content"])
        context = material["citation_context"]
        self.assertEqual(context["mapping_status"], "mapped")
        self.assertEqual([c["content"] for c in context["citations"]], ["收入100万元"])
        self.assertNotIn("未引用的候选", request["messages"][1]["content"])
        self.assertNotIn("actual_reference", material)
        self.assertNotIn("retrieved_chunks", material)
        self.assertEqual(request["model"], MODEL)
        self.assertEqual(request["reasoning_effort"], "medium")
        self.assertNotIn("tools", request)

    def test_success_is_explicitly_model_not_human(self):
        row = judge_one(self.gold, self.generated, "signature", "http://127.0.0.1/v1", "token", 10,
                        request_fn=lambda *args: self.response())
        self.assertEqual(row["judge_type"], "model")
        self.assertEqual(row["judge_status"], "ok")
        self.assertTrue(row["answer_correct"])
        self.assertIsNone(row["human_spot_check"])
        self.assertIsNone(row["cost_usd"])
        self.assertEqual(row["usage"]["total_tokens"], 120)
        self.assertEqual(row["judge_run_signature"], "signature")
        self.assertTrue(row["citation_support"])
        self.assertEqual(row["citation_mapping"]["marker_indexes"], [0])

    def judge(self):
        return judge_one(self.gold, self.generated, "sig", "url", "token", 1,
                         request_fn=lambda *args: self.response())

    def test_uncited_candidates_cannot_make_factual_answer_supported(self):
        self.generated["answer"] = "收入100万元。"
        request = build_request(self.gold, self.generated)
        self.assertEqual(json.loads(request["messages"][1]["content"])["citation_context"]["citations"], [])
        row = self.judge()
        self.assertFalse(row["citation_support"])
        self.assertTrue(row["citation_support_model"])
        self.assertTrue(json.loads(row["raw_judgement"])["citation_support"])
        self.assertIn("没有显式引用标记", row["judgement_reason"])

    def test_abstention_without_factual_claim_is_not_forced_false(self):
        self.generated["answer"] = "所给材料不足，无法回答。"
        self.judgement["has_factual_claims"] = False
        self.assertIsNone(self.judge()["citation_support"])

    def test_out_of_range_or_partial_mapping_is_unknown(self):
        self.generated["answer"] = "收入100万元[ID:0][ID:9]"
        row = self.judge()
        self.assertIsNone(row["citation_support"])
        self.assertEqual(row["citation_mapping"]["unmapped_indexes"], [9])

    def test_missing_original_text_is_unknown_even_with_metadata(self):
        self.generated["reference"]["chunks"][0] = {"document_id": "A", "positions": [[4, 0, 0, 1, 1]], "content_ltks": "收入100万元"}
        row = self.judge()
        self.assertIsNone(row["citation_support"])
        self.assertEqual(row["citation_mapping"]["missing_text_indexes"], [0])

    def test_dictionary_chunks_are_not_silently_reordered(self):
        self.generated["reference"]["chunks"] = {"opaque-id": {"content": "收入100万元"}}
        self.assertIsNone(self.judge()["citation_support"])

    def test_zero_based_markers_and_repeated_markers(self):
        self.generated["answer"] = "甲[ID:1]乙[0]丙[ID:1]丁[ID:١]"
        context = citation_context(self.generated)
        self.assertEqual(context["marker_indexes"], [1, 0])
        self.assertEqual([c["content"] for c in context["citations"]], ["未引用的候选，不能用来补证据", "收入100万元"])

    def test_malformed_explicit_marker_is_unknown(self):
        self.generated["answer"] = "收入100万元[ID:unknown]"
        self.assertIsNone(self.judge()["citation_support"])

    def test_thinking_citations_do_not_count_as_answer_citations(self):
        self.generated["answer"] = "<think>候选[ID:0]</think>收入100万元。"
        request = build_request(self.gold, self.generated)
        self.assertNotIn("<think>", request["messages"][1]["content"])
        self.assertFalse(self.judge()["citation_support"])

    def test_raw_reference_original_text_field_is_supported(self):
        self.generated["reference"]["chunks"][0] = {"content_with_weight": "收入100万元"}
        self.assertEqual(citation_context(self.generated)["citations"][0]["content"], "收入100万元")

    def test_missing_generation_is_not_fake_model_score(self):
        def forbidden(*args):
            raise AssertionError("不应调用模型")
        row = judge_one(self.gold, {"status": "request_error"}, "sig", "url", "token", 1, request_fn=forbidden)
        self.assertEqual(row["judge_status"], "skipped_generation_failure")
        self.assertFalse(row["model_call_attempted"])
        self.assertIsNone(row["answer_correct"])

    def test_failed_judge_retains_row_and_unknown_score(self):
        def fail(*args):
            raise RuntimeError("调用失败，token")
        row = judge_one(self.gold, self.generated, "sig", "url", "token", 1, request_fn=fail)
        self.assertEqual(row["id"], "q1")
        self.assertEqual(row["judge_status"], "failed")
        self.assertIsNone(row["answer_correct"])
        self.assertNotIn("token", row["error"])

    def test_invalid_output_preserves_usage(self):
        response = self.response()
        response["choices"][0]["message"]["content"] = '{"answer_correct": "true"}'
        row = judge_one(self.gold, self.generated, "sig", "url", "token", 1, request_fn=lambda *args: response)
        self.assertEqual(row["judge_status"], "failed")
        self.assertEqual(row["usage"]["total_tokens"], 120)
        self.assertIn("answer_correct", row["raw_judgement"])

    def test_rejects_wrong_effort_model_and_tools(self):
        response = self.response()
        response["bridge_metadata"]["reasoning_effort"] = "high"
        with self.assertRaises(ValueError):
            parse_judgement(response)
        response = self.response()
        response["model"] = "another-model"
        with self.assertRaises(ValueError):
            parse_judgement(response)
        response = self.response()
        response["choices"][0]["message"]["tool_calls"] = [{"function": {"name": "search"}}]
        with self.assertRaises(ValueError):
            parse_judgement(response)

    def test_unknown_judgement_is_not_forced_to_false(self):
        self.judgement["numeric_and_scope_correct"] = None
        parsed, _ = parse_judgement(self.response())
        self.assertIsNone(parsed["numeric_and_scope_correct"])

    def test_duplicate_ids_rejected(self):
        with self.assertRaises(ValueError):
            unique_rows([self.gold, self.gold], "金标")


if __name__ == "__main__":
    unittest.main()
