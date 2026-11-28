"""财务校验的跨期、单位、符号、定义与缺项回归。"""
import unittest
from timeguard.research.financial_agent import Operand
from timeguard.evidence.formula_guard import formula,requirements,check


def money(value,year=2022,scale="millions",label="revenue"):
    return Operand(chunk_id="原表",line_number=2,number=value,year=year,period="FY",currency="USD",scale=scale,kind="money",row_label=label)


class FormulaGuardTest(unittest.TestCase):
    def test_comparative_quick_ratio(self):
        name,needs=requirements("Has Example's quick ratio improved between FY2022 and FY2023?")
        self.assertEqual(len(needs),6)
        values=[money(v,y) for (m,y),v in zip(needs,("100","20","40","120","30","45"))]
        self.assertIn("2022 年",formula(name,values))
        self.assertIn("2023 年",formula(name,values))
        self.assertIn("2.0000",formula(name,values))

    def test_turnover_and_unit(self):
        name,needs=requirements("FY2022 fixed asset turnover, based on FY2021 and FY2022.")
        self.assertEqual(formula(name,[money("120"),money("50",2021),money("70")]).split()[-1],"2.0000")
        with self.assertRaisesRegex(ValueError,"单位"):
            formula(name,[money("120"),money("50",2021),money("70",scale="billions")])
        self.assertEqual(len(needs),3)

    def test_sign_and_missing_period(self):
        self.assertIn("-15",formula("资本开支",[money("(15)")]))
        result=check("FY2022 fixed asset turnover",[],[],None)
        self.assertEqual(result["verified"],0)
        self.assertEqual(len(result["missing"]),3)

    def test_percentage_points(self):
        name,needs=requirements("Change in gross margin in percentage points from 2021 to 2022")
        operands=[money(v,y,label="Gross margin").model_copy(update={"kind":"percentage","scale":"ones","currency":"XXX"})
                  for (_,y),v in zip(needs,("9.5","10.0"))]
        self.assertEqual(formula(name,operands),"-0.5 个百分点")

    def test_reject_other_document(self):
        with self.assertRaisesRegex(ValueError,"越界"):
            check("2022 capital expenditure",[{"document_id":"乙"}],["甲"],None)
    def test_pipe_table_checks_year_column(self):
        text="Consolidated statement (USD millions)\n| Item | 2021 | 2022 |\n| Net sales | 50 | 60 |"
        c={"id":"表","document_id":"甲","content":text}
        r=check("Net sales growth from 2021 to 2022",[c],["甲"],lambda _:c)
        self.assertEqual(r["verified"],2)
        self.assertIn("20%",r["instruction"])
        self.assertEqual([f["number"] for f in r["facts"]],["60","50"])
    def test_fused_scale_is_not_a_second_valid_unit(self):
        text="<table><caption>Dollarsinmillions,exceptper share</caption><tr><th></th><th>2021</th><th>2022</th></tr><tr><td>Netsales</td><td>$ 50</td><td>$ 60</td></tr></table>"
        c={"id":"表","document_id":"甲","content":text}
        result=check("net sales growth from FY2021 to FY2022",[c],["甲"],lambda _:c)
        self.assertEqual(result["verified"],2)
        self.assertIn("20%",result["instruction"])
    def test_segment_request_rejects_consolidated_numbers(self):
        text="Consolidated sales (USD millions)\n| Item | 2021 | 2022 |\n| Net sales | 50 | 60 |"
        c={"id":"表","document_id":"甲","content":text}
        result=check("Domestic net sales growth from FY2021 to FY2022",[c],["甲"],lambda _:c)
        self.assertEqual(result["verified"],0)
