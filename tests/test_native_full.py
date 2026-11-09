"""验证原生完整基线复用保留未知、不改变输入且不重复发送已判题。"""
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from timeguard.common import digest,read_jsonl,write_json,write_jsonl
from timeguard import judge
from timeguard.native_full import reuse_reviews
from timeguard import native_full,native_baseline


class NativeFullTests(unittest.TestCase):
    def fixture(self,with_citation=False):
        generated={"q1":{"id":"q1","status":"ok","answer":"原答案[ID:0]" if with_citation else "材料不足",
                   "reference":{"chunks":[{"id":"c1","document_id":"d1","content":"原文"}] if with_citation else []}}}
        gold={"q1":{"id":"q1","question":"问题1","answer":"标准答案","evidence":[]}}
        payload={"answer_correct":None,"numeric_and_scope_correct":None,"citation_support":True if with_citation else None,
                 "has_factual_claims":with_citation,"judgement_reason":"证据需要核对。","failure_category":"未知"}
        metadata={"transport":judge.TRANSPORT,"reasoning_effort":judge.EFFORT,"api_equivalent":False,"actual_model":judge.MODEL}
        context=judge.citation_context(generated["q1"])
        row={**judge.apply_citation_policy(payload,context),"id":"q1","judge_status":"ok","judge_type":"model",
             "judge_run_signature":"old","judge_model":judge.MODEL,"bridge_metadata":metadata,"elapsed_seconds":1.0,
             "judge_input_sha256":digest(judge.build_request(gold["q1"],generated["q1"])),"raw_judgement":json.dumps(payload),
             "citation_mapping":{k:v for k,v in context.items() if k!="citations"}}
        line=json.dumps(row,ensure_ascii=False).encode()
        response={"model":judge.MODEL,"bridge_metadata":metadata,"choices":[{"message":{"content":json.dumps(payload)}}]}
        return gold,generated,row,line,response

    def test_unknown_judgement_and_original_signature_preserved(self):
        gold,generated,original,line,_=self.fixture()
        result=reuse_reviews([line],"old","new",gold,generated)[0]
        self.assertIsNone(result["answer_correct"])
        self.assertEqual(result["judge_run_signature"],"new")
        self.assertEqual(result["reuse_origin"]["judge_run_signature"],"old")
        self.assertEqual(original["judge_run_signature"],"old")

    def test_changed_answer_or_cited_evidence_rejected(self):
        for citation in (False,True):
            gold,generated,_,line,_=self.fixture(citation);changed=deepcopy(generated)
            if citation:changed["q1"]["reference"]["chunks"][0]["content"]="改变后的原文"
            else:changed["q1"]["answer"]="改变后的答案"
            with self.assertRaises(ValueError):reuse_reviews([line],"old","new",gold,changed)

    def test_reused_judgement_does_not_send_another_request(self):
        gold,generated,_,line,response=self.fixture()
        reused=reuse_reviews([line],"old","new",gold,generated)
        gold["q2"]={"id":"q2","question":"问题2","answer":"标准答案2","evidence":[]}
        generated["q2"]={"id":"q2","status":"ok","answer":"材料不足","reference":{"chunks":[]}}
        calls=[]
        def request(*args):calls.append(args[1]);return response
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp:
            self.assertTrue(Path(temp).resolve().is_relative_to(Path.cwd().resolve()))
            output=Path(temp)/"reviews.jsonl"
            ok=judge.judge_queries(["q1","q2"],gold,generated,"new",output,"unused","token",1,
                                   concurrency=1,request_fn=request,reuse_rows=reused)
            self.assertTrue(ok)
            self.assertEqual(len(calls),1)
            self.assertEqual(len(read_jsonl(output)),2)

    def test_generation_only_sends_new111_and_never_reads_judge_material(self):
        previous=Path.cwd()
        with tempfile.TemporaryDirectory(dir=previous) as temp:
            self.assertTrue(Path(temp).resolve().is_relative_to(previous.resolve()))
            try:
                os.chdir(temp)
                queries=[{"id":str(i),"question":"问题"+str(i),"split":"dev" if i<39 else "tune" if i<127 else "test"} for i in range(150)]
                query_path=Path("data/financebench/queries.jsonl");write_jsonl(query_path,queries)
                root=Path("runs");source=root/"source";source.mkdir(parents=True)
                spec={"queries":queries[:39]};write_json(source/"spec.json",spec)
                original=[{**q,"status":"ok","answer":"已有答案","reference":{},"elapsed_seconds":1} for q in queries[:39]]
                write_jsonl(source/"results.jsonl",original)
                write_json(source/"manifest.json",{"identity":{"query_ids":[q["id"] for q in queries[:39]],"split":"dev",
                    "queries_sha256":digest(query_path.read_bytes()),"source_sha256":digest(Path(native_baseline.__file__).read_bytes()),
                    "freeze":{"same":True},"spec_sha256":digest(spec)}})
                write_json(source/"execution.json",{"wall_seconds":10,"actual_calls":39,"tokens":390})
                chosen={"run":"source","manifest_sha256":digest((source/"manifest.json").read_bytes()),
                        "results_sha256":digest((source/"results.jsonl").read_bytes()),"reviews_sha256":"a"*64,"reviews_manifest_sha256":"b"*64}
                write_json(root/"selection-core.json",{"chosen":chosen})
                private=Path("private");private.mkdir();(private/"bridge-audit.jsonl").write_bytes(b"")
                before=(source/"results.jsonl").read_bytes();requests=[]
                def worker(fresh_spec,out):
                    requests.extend(fresh_spec["queries"])
                    self.assertNotIn("已有答案",json.dumps(fresh_spec,ensure_ascii=False))
                    write_jsonl(out/"results.jsonl",[{**q,"status":"ok","answer":"新增答案","elapsed_seconds":1} for q in fresh_spec["queries"]])
                    return 0
                with patch("timeguard.matrix.frozen_index",return_value={"same":True}),patch.object(native_full,"private_runtime",return_value=(private,{})),patch.object(native_full,"container_worker",side_effect=worker):
                    self.assertEqual(native_full.generate(root,source,root/"full"),0)
                self.assertEqual(len(requests),111)
                self.assertFalse({q["id"] for q in requests}&{q["id"] for q in queries[:39]})
                self.assertEqual(len(read_jsonl(root/"full/results.jsonl")),150)
                self.assertEqual((source/"results.jsonl").read_bytes(),before)
                self.assertFalse((source/"model_reviews.jsonl").exists())
            finally:
                os.chdir(previous)
