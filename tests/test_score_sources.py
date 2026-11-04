"""确保模型裁判不会被报告成已完成人工核验。"""

import unittest
from timeguard.score import summarize


class ReviewSourceTests(unittest.TestCase):
    def setUp(self):
        self.gold = {"q": {"id": "q", "question": "问题", "answer": "答案", "evidence": []}}
        self.result = [{"id": "q", "status": "ok", "retrieved_chunks": [], "answer": "答案"}]
        self.review = {"id": "q", "answer_correct": True, "numeric_and_scope_correct": True,
                       "citation_support": False, "reviewer": "模型裁判", "judge_type": "model", "judge_status": "ok"}

    def test_model_review_keeps_human_accuracy_empty(self):
        report, _, _ = summarize(["q"], self.gold, self.result, reviews=[self.review])
        self.assertEqual(report["model_answer_accuracy"], 1)
        self.assertEqual(report["model_citation_support_rate"], 0)
        self.assertEqual(report["model_reviewed"], 1)
        self.assertEqual(report["human_reviewed"], 0)
        self.assertIsNone(report["answer_accuracy"])
        self.assertIsNone(report["human_spot_check"])

    def test_failed_model_judge_does_not_become_valid(self):
        report, _, _ = summarize(["q"], self.gold, self.result,
                                 reviews=[{**self.review, "judge_status": "request_error"}])
        self.assertIsNone(report["model_answer_accuracy"])
        self.assertEqual(report["model_reviewed"], 0)

    def test_human_score_remains_separate(self):
        review = {**self.review, "judge_type": "human", "reviewer": "人工复核人"}
        report, _, _ = summarize(["q"], self.gold, self.result, reviews=[review])
        self.assertEqual(report["answer_accuracy"], 1)
        self.assertIsNone(report["model_answer_accuracy"])


if __name__ == "__main__":
    unittest.main()
