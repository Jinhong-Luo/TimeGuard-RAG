"""分流、补查失败、引用和空范围的回归。"""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from timeguard.common import digest,write_json
from timeguard.final_report import generation_cost
from timeguard.finance_route import simple_metric,simple_lookup,needs_review
from timeguard.filing_pages import FilingPages,pages_for
from timeguard.scope import intersect_scope
from timeguard.judge import citation_context


class RoutingTest(unittest.TestCase):
    def test_simple_success_needs_no_model(self):
        text="Consolidated statement (USD millions)\n| Item | 2021 | 2022 |\n| Capital expenditures | (15) | (17) |"
        c={"id":"块","document_id":"甲","content":text}
        result=simple_lookup("What was FY2022 capital expenditure in USD millions?",[c],["甲"],lambda _:c)
        self.assertIn("17",result["answer"])
        self.assertEqual(citation_context(result)["mapping_status"],"mapped")
        self.assertEqual(citation_context(result)["citation_origin"],"rule")

    def test_no_guess_on_ambiguous_currency(self):
        text="Capital expenditures 2022 15"
        c={"id":"块","document_id":"甲","content":text}
        self.assertIsNone(simple_lookup("FY2022 capital expenditure in USD millions?",[c],["甲"],lambda _:c))
        self.assertIsNone(simple_metric("Revenue growth from 2021 to 2022, in USD millions"))

    def test_multi_period_and_gap(self):
        self.assertEqual(needs_review("Revenue in 2021 and 2022",[]),"多期")
        self.assertTrue(needs_review("Revenue in 2022",[{"content":"Revenue in 2020"}]))
        self.assertIsNone(needs_review("Revenue in 2022",[{"content":"Revenue in 2022"}]))

    def test_supplement_timeout_keeps_first_pages(self):
        c={"id":"块","document_id":"甲","content":"Revenue in 2021","positions":[[1]]}
        idx=FilingPages(None,"库",[{"id":"甲","company":"Example","doc_name":"年报"}])
        idx.cache["甲"]=[c]
        with patch("timeguard.filing_pages.pages_for",side_effect=[([c],[("甲",1)]),TimeoutError()]) as search:
            result=idx.retrieve("Example revenue in 2022",{"input":{"tools":{"doc_scope":["甲"]}}})
        self.assertEqual(search.call_count,2)
        self.assertEqual(result["extra_error"],"TimeoutError")
        self.assertEqual(result["supplement"],[c])

    def test_empty_or_foreign_filter_stays_closed(self):
        self.assertEqual(intersect_scope([],None),[])
        self.assertEqual(intersect_scope(["甲"],["乙"]),[])
        self.assertEqual(intersect_scope(["甲"],None),["甲"])

    def test_recovery_counts_every_attempt(self):
        with TemporaryDirectory() as tmp:
            initial=Path(tmp,"首次")
            resumed=Path(tmp,"恢复")
            initial.mkdir()
            resumed.mkdir()
            raw=b"{}\n"
            (initial/"results.jsonl").write_bytes(raw)
            record={"actual_calls":3,"failed_calls":1,"tokens":45,"wall_seconds":4,"model_identity_verified":True}
            write_json(initial/"execution.json",record)
            proof={"source_run":initial.name,"results_sha256":digest(raw)}
            write_json(resumed/"execution.json",{**record,"failed_calls":0,"recovery":proof})
            result=generation_cost(resumed)
            self.assertEqual((result["actual_calls"],result["failed_calls"],result["tokens"]),(6,1,90))
            (initial/"results.jsonl").write_bytes(b"[]\n")
            with self.assertRaises(ValueError):
                generation_cost(resumed)
