"""离线说明必须绑定实际理由，不能在未来重跑中沿用旧摘要。"""

import unittest

from timeguard.common import digest
from timeguard.report_baseline import bound_reason


class ReasonBindingTests(unittest.TestCase):
    def test_changed_reason_or_missing_annotation_uses_generic_summary(self):
        annotation = (digest("原始判断理由"), "本次整理摘要")
        exact = bound_reason({"judgement_reason": "原始判断理由"}, annotation)
        self.assertTrue(exact["summary_source_matched"])
        self.assertEqual(exact["reason_summary"], "本次整理摘要")
        changed = bound_reason({"judgement_reason": "新的判断理由"}, annotation)
        self.assertFalse(changed["summary_source_matched"])
        self.assertNotEqual(changed["reason_summary"], "本次整理摘要")
        self.assertEqual(changed["judgement_reason_sha256"], digest("新的判断理由"))
        absent = bound_reason({"judgement_reason": "新的判断理由"}, None)
        self.assertEqual(changed, absent)


if __name__ == "__main__":
    unittest.main()
