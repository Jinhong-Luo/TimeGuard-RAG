"""验证先回看、缺项才补搜、最多一次和失败回退。"""
import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from timeguard.evals.trials.four_feedback import feedback
from timeguard.evals.trials.paired_trial import run_packets


class FeedbackTest(unittest.TestCase):
    def setUp(self):
        self.chunks = [{"id":str(n),"document_id":"甲","document_name":"甲年报",
                        "positions":[[1]],"content":str(n)+" 行"} for n in range(20)]
        self.query = {"id":"调优题","question":"本年收入？"}
        self.missing = {"missing":["本年合并收入"]}
        self.old = {"selected":[2,3,4,8],"checks":[None],"query":"consolidated revenue"}

    def execute(self, replies, search):
        with patch("timeguard.evals.trials.four_feedback.chat",side_effect=replies):
            return feedback({},self.query,{"chunks":self.chunks},[1,2,3,4],self.chunks[:4],search)

    def test_sufficient_evidence_skips_review_and_search(self):
        search = Mock()
        chosen,record = self.execute([{"missing":[]}],search)
        self.assertEqual(chosen["feedback"],self.chunks[:4])
        self.assertEqual(record["补搜次数"],0)
        search.assert_not_called()

    def test_old_quote_avoids_new_search(self):
        search = Mock()
        reply = {**self.old,"checks":[{"id":8,"quote":"7 行"}]}
        chosen,_ = self.execute([self.missing,reply],search)
        self.assertEqual(chosen["feedback"][-1],self.chunks[7])
        search.assert_not_called()

    def test_one_search_and_failed_second_review_keeps_history_choice(self):
        search = Mock(return_value=[{**self.chunks[0],"id":"新增"}])
        chosen,r = self.execute([self.missing,self.old,{"bad":True}],search)
        search.assert_called_once_with("consolidated revenue")
        self.assertEqual((r["补搜次数"],r["回退阶段"]),(1,"补搜后筛选"))
        self.assertEqual(chosen["feedback"][-1],self.chunks[7])

    def test_invalid_old_quote_does_not_trigger_search(self):
        search = Mock()
        bad = {**self.old,"checks":[{"id":5,"quote":"无效"}]}
        chosen,r = self.execute([self.missing,bad],search)
        self.assertEqual(r["回退阶段"],"旧池复查")
        self.assertEqual(chosen["feedback"],chosen["no_feedback"])
        search.assert_not_called()

    def test_search_failure_is_counted(self):
        _,r = self.execute([self.missing,self.old],Mock(side_effect=TimeoutError()))
        self.assertEqual((r["补搜次数"],r["补搜失败"],r["回退阶段"]),(1,1,"定向补搜"))

    def test_new_source_can_replace_one_old_block(self):
        extra = {**self.chunks[0],"id":"新增","content":"可逐字核对的新依据"}
        final = {"selected":[2,3,4,21],"checks":[{"id":21,"quote":"新依据"}],"query":""}
        search = Mock(return_value=[self.chunks[2],extra])
        chosen,r = self.execute([self.missing,self.old,final],search)
        self.assertEqual(chosen["history_only"][-1]["id"],"7")
        self.assertEqual(chosen["feedback"][-1]["id"],"新增")
        self.assertEqual((r["新增候选数"],r["最终仍缺项数"]),(1,0))
        search.assert_called_once()

    def test_recovery_only_reuses_identical_model_request(self):
        row = {"id":"题","question":"问题","split":"调优"}
        body = {"messages":[{"role":"user","content":"已冻结的请求"}]}
        def record():
            return {"snapshot":{"sha256":"候选"},"selected":[],"block_hashes":[],"requests":[]}
        with TemporaryDirectory() as tmp:
            source = Path(tmp,"首次")
            (source/"对照").mkdir(parents=True)
            (source/"results.jsonl").write_text(json.dumps(row),encoding="utf-8")
            saved = record()
            saved["requests"] = [{"body":body,"status":200}]
            old = {**row,"answer":"已完成","status":"ok","native_trace":{"evidence_layers":{"answers":[saved]}}}
            (source/"对照"/"results.jsonl").write_text(json.dumps(old),encoding="utf-8")
            for changed in (False,True):
                out = Path(tmp,str(changed))
                out.mkdir()
                packet = {**record(),"body":{} if changed else body,"visible":[]}
                with patch("timeguard.evals.trials.paired_trial.generate") as generate, patch("timeguard.evals.trials.paired_trial.judge",return_value=0):
                    if changed:
                        with self.assertRaises(ValueError):
                            run_packets(source,out,["对照"],[row],[{"对照":packet}],{},source)
                    else:
                        run_packets(source,out,["对照"],[row],[{"对照":packet}],{},source)
                        result = json.loads((out/"对照"/"results.jsonl").read_text(encoding="utf-8"))
                        self.assertEqual(result["恢复来源"],"首次")
                    generate.assert_not_called()
