"""开发集配对报告的分母、损失和共享请求核验。"""
import unittest

from timeguard.full_feedback import packet_key
from timeguard.full_feedback_report import transitions,STAGES


class FullFeedbackTest(unittest.TestCase):
    def test_gross_and_net_cost(self):
        def row(before,after):
            return {"甲":{"达标":before},"乙":{"达标":after},
                    "阶段":{"主模型反馈":{"次数":1},"定向补搜":{"次数":1},"补搜后筛选":{"次数":1}}}
        rows = {"一":row(False,True),"二":row(False,True),"三":row(True,False),"四":row(True,True)}
        r = transitions(rows,"甲","乙",STAGES)
        self.assertEqual((r["救回"],r["损失"],r["净增"]),(["一","二"],["三"],1))
        self.assertEqual(r["每救回一道"]["合计"],6)
        self.assertEqual(r["每净增一道"]["合计"],12)
        self.assertIsNone(transitions({"三":rows["三"]},"甲","乙",STAGES)["每净增一道"])

    def test_same_text_with_different_source_is_not_shared(self):
        p = {"body":{"messages":["相同文字"]},"block_hashes":["原文"],"visible":[{"id":1}]}
        self.assertNotEqual(packet_key(p),packet_key({**p,"visible":[{"id":2}]}))
