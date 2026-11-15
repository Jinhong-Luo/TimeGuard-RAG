import unittest
from timeguard.filing_pages import FilingPages,missing_query,pages_for


class FilingPageTest(unittest.TestCase):
    def test_search_respects_report_pages(self):
        chunks=[{"id":str(i),"content":body,"document_id":"甲","positions":[[i]]} for i,body in enumerate(
            ["","litigation and legal proceedings","operating margins","cash"],start=0)]
        best,first=pages_for("legal battles 2022",chunks,count=1)
        self.assertEqual(first,[("甲",1)])
        self.assertEqual(best[0]["id"],"1")
        self.assertIn("2022",missing_query("legal battles 2022",best))

    def test_no_future_report_reads(self):
        doc={"id":"甲","company":"Example","published_at":"2023-01-01","doc_name":"报告"}
        index=FilingPages(None,"库",[doc])
        result=index.retrieve("Example revenue as of 2022-01-01",{"input":{"tools":{"doc_scope":["甲"]}}})
        self.assertEqual(result["page"],[])
        self.assertEqual(result["index_reads"],0)
        self.assertEqual(result["extra_searches"],0)
    def test_supplement_searches_once_within_intersection(self):
        docs=[{"id":d,"company":co,"doc_name":d} for d,co in (("甲","Alpha"),("乙","Beta"))]
        index=FilingPages(None,"库",docs)
        index.cache["甲"]=[{"document_id":"甲","id":str(i),"content":"Revenue in 2021 was 8.",
                            "positions":[[i]]} for i in range(1,8)]
        snap={"input":{"tools":{"doc_scope":["甲","乙"]}}}
        missing=index.retrieve("Alpha revenue 2022 and 2021",snap)
        self.assertEqual(missing["extra_searches"],1)
        self.assertEqual(missing["allowed"],["甲"])
        self.assertTrue(all(c["document_id"]=="甲" for c in missing["supplement"]))
        complete=index.retrieve("Alpha revenue 2021",snap)
        self.assertEqual(complete["extra_searches"],0)
        self.assertEqual(complete["page"],complete["supplement"])
