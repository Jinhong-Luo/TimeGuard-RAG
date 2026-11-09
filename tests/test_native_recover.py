"""恢复仅按请求状态选取完整行，不依据答案对错挑选来源。"""
import unittest
from timeguard.native_recover import recovery_plan


class NativeRecoveryTests(unittest.TestCase):
    def test_only_failed_and_unrun_questions_are_sent(self):
        queries=[{"id":str(i),"question":"题"+str(i)} for i in range(39)]
        rows=[{**q,"status":"ok","answer":"保留答案"} for q in queries[:30]]
        rows.extend({**q,"status":"request_error"} for q in queries[30:32])
        good,pending=recovery_plan(queries,rows)
        self.assertEqual(len(good),30)
        self.assertEqual(len(pending),9)
        self.assertEqual({q["id"] for q in pending},{str(i) for i in range(30,39)})
        good["0"]["answer"]="副本变化"
        self.assertEqual(rows[0]["answer"],"保留答案")

    def test_changed_question_rejected(self):
        with self.assertRaises(ValueError):recovery_plan([{"id":"1","question":"原题"}],[{"id":"1","question":"新题","status":"ok"}])
