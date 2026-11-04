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

    def test_unknown_model_fields_preserve_full_denominator(self):
        gold = {ident: {**self.gold["q"], "id": ident} for ident in ("q", "r")}
        rows = [{**self.result[0], "id": ident} for ident in gold]
        reviews = [self.review, {**self.review, "id": "r", "numeric_and_scope_correct": None,
                                "citation_support": None}]
        report, _, _ = summarize(list(gold), gold, rows, reviews=reviews)
        self.assertEqual(report["model_answer_accuracy"], 1)
        for target in ("model_numeric_and_scope_accuracy", "model_citation_support_rate"):
            self.assertIsNone(report[target])
            self.assertEqual(report[target + "_known_count"], 1)
            self.assertEqual(report[target + "_unknown_count"], 1)
            self.assertEqual(report[target + "_coverage"], 0.5)

    def test_failed_and_missing_judgements_do_not_improve_coverage(self):
        gold = {ident: {**self.gold["q"], "id": ident} for ident in ("q", "r", "s")}
        reviews = [self.review, {**self.review, "id": "r", "judge_status": "failed"}]
        report, _, _ = summarize(list(gold), gold, self.result, reviews=reviews)
        self.assertEqual(report["denominator"], 3)
        self.assertIsNone(report["model_answer_accuracy"])
        for target in ("model_numeric_and_scope_accuracy", "model_citation_support_rate"):
            self.assertIsNone(report[target])
            self.assertEqual(report[target + "_known_count"], 1)
            self.assertEqual(report[target + "_unknown_count"], 2)
            self.assertEqual(report[target + "_coverage"], 1 / 3)

    def test_partial_human_and_absent_reviews_keep_unknown_counts(self):
        gold = {ident: {**self.gold["q"], "id": ident} for ident in ("q", "r")}
        review = {**self.review, "judge_type": "human", "reviewer": "人工复核人"}
        for reviews, known in (([review], 1), (None, 0)):
            report, _, _ = summarize(list(gold), gold, self.result, reviews=reviews)
            for target in ("numeric_and_scope_accuracy", "citation_support_rate"):
                self.assertIsNone(report[target])
                self.assertEqual(report[target + "_known_count"], known)
                self.assertEqual(report[target + "_unknown_count"], 2 - known)
                self.assertEqual(report[target + "_coverage"], known / 2)
            self.assertEqual(report["model_citation_support_rate_coverage"], 0)


if __name__ == "__main__":
    unittest.main()
