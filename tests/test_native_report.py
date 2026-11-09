"""验证原生报告的未知与缺失分母，以及模型审计窗口绑定。"""
import json
import unittest
from timeguard.common import digest
from timeguard.native_report import judgement_counts,window_records


class NativeReportTests(unittest.TestCase):
    def test_unknown_and_missing_cannot_become_complete_accuracy(self):
        result=judgement_counts(["a","b","c"],[{"id":"a","answer_correct":True},{"id":"b","answer_correct":None}])
        self.assertEqual(result["planned"],3)
        self.assertEqual(result["correct"],1)
        self.assertEqual(result["unknown"],2)
        self.assertEqual(result["missing_reviews"],1)
        self.assertIsNone(result["full_accuracy"])
        self.assertEqual(result["known_subset_accuracy"],1)

    def test_audit_window_requires_exact_bytes(self):
        raw=(json.dumps({"request_id":"one"})+"\n").encode()
        window={"audit_byte_start":0,"audit_byte_end":len(raw),"audit_sha256":digest(raw)}
        self.assertEqual(window_records(raw,window),[{"request_id":"one"}])
        with self.assertRaises(ValueError):window_records(raw.replace(b"one",b"two"),window)

    def test_duplicate_judgements_rejected(self):
        with self.assertRaises(ValueError):judgement_counts(["a"],[{"id":"a"},{"id":"a"}])
