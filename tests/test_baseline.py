"""验证数据隔离、评分分母及原生参数传递的关键约束。"""

import copy
import unittest

from timeguard.data import prepare
from timeguard.run import chat_body, retrieval_body
from timeguard.score import evidence_metrics, summarize


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.gold = {"id": "q1", "question": "营业收入是多少？", "answer": "100", "evidence": [
            {"doc_name": "A_2024", "evidence_page_num": 9, "evidence_text": "Revenue 100 million"}]}
        self.chunk = {"document_name": "A_2024.pdf", "positions": [[10, 0, 0, 0, 0]], "content": "Revenue 100 million"}

    def test_zero_retrieval_stays_in_denominator(self):
        gold = {"q1": self.gold, "q2": {**self.gold, "id": "q2"}}
        rows = [{"id": "q1", "status": "ok", "retrieved_chunks": [self.chunk]},
                {"id": "q2", "status": "ok", "retrieved_chunks": []}]
        report, _, _ = summarize(["q1", "q2"], gold, rows)
        self.assertEqual(report["denominator"], 2)
        self.assertEqual(report["page_hit_recall_at_k_proxy"], 0.5)
        self.assertEqual(report["zero_retrieval_count"], 1)

    def test_missing_and_failed_requests_remain(self):
        gold = {"q1": self.gold, "q2": {**self.gold, "id": "q2"}}
        report, _, _ = summarize(["q1", "q2"], gold, [{"id": "q1", "status": "request_error"}])
        self.assertEqual(report["denominator"], 2)
        self.assertEqual(report["status_counts"], {"request_error": 1, "missing": 1})
        self.assertEqual(report["page_hit_recall_at_k_proxy"], 0)

    def test_wrong_document_cannot_hit_same_page(self):
        chunk = {**self.chunk, "document_name": "B_2024.pdf"}
        self.assertEqual(evidence_metrics(self.gold, [chunk], 6)["page_hit_recall"], 0)

    def test_page_base_and_k_boundary(self):
        wrong = {**self.chunk, "positions": [[9, 0, 0, 0, 0]]}
        self.assertEqual(evidence_metrics(self.gold, [wrong, self.chunk], 1)["page_hit_recall"], 0)
        self.assertEqual(evidence_metrics(self.gold, [wrong, self.chunk], 2)["page_hit_recall"], 1)

    def test_partial_review_does_not_claim_accuracy(self):
        gold = {"q1": self.gold, "q2": {**self.gold, "id": "q2"}}
        report, _, _ = summarize(["q1", "q2"], gold, [], reviews=[{"id": "q1", "answer_correct": True, "reviewer": "复核人"}])
        self.assertIsNone(report["answer_accuracy"])
        self.assertEqual(report["reviewed"], 1)

    def test_duplicate_runs_rejected(self):
        with self.assertRaises(ValueError):
            summarize(["q1"], {"q1": self.gold}, [{"id": "q1"}, {"id": "q1"}])

    def test_no_gold_fields_reach_retrieval(self):
        config = {"dataset_ids": ["dataset"], "retrieval": {"page_size": 6}}
        payload = retrieval_body(config, {**self.gold, "doc_name": "SECRET_DOC"})
        self.assertEqual(payload, {"page_size": 6, "dataset_ids": ["dataset"], "question": self.gold["question"]})

    def test_effort_in_native_setting_only(self):
        config = {"dataset_ids": ["d"], "model": "m", "llm_setting": {},
                  "retrieval": {"page_size": 6, "top_k": 1024, "similarity_threshold": 0.2, "vector_similarity_weight": 0.3}}
        initial = copy.deepcopy(config)
        body = chat_body(config, "high", "test")
        self.assertEqual(body["llm_setting"]["reasoning_effort"], "high")
        self.assertFalse(body["prompt_config"]["reasoning"])
        self.assertEqual(config, initial)

    def test_company_split_and_no_answer_in_queries(self):
        source = [{"financebench_id": f"q{i}", "company": f"C{i // 5}", "question": "问题", "answer": "保密答案",
                   "doc_name": f"D{i // 5}", "evidence": [{"doc_name": f"D{i // 5}"}]} for i in range(150)]
        docs = [{"doc_name": f"D{i}"} for i in range(30)]
        queries, gold, _, allocation = prepare(source, docs)
        self.assertEqual(set(allocation.values()), {"tune", "dev", "test"})
        self.assertTrue(all(set(q) == {"id", "question", "split"} for q in queries))
        for company in allocation:
            self.assertEqual(len({r["split"] for r in gold if r["company"] == company}), 1)


if __name__ == "__main__":
    unittest.main()
