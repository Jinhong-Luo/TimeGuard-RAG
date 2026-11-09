"""验证原生报告的未知与缺失分母，以及模型审计窗口绑定。"""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from timeguard.common import digest,write_json
from timeguard.native_report import judgement_counts,window_records,build


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

    def test_pending_scope_keeps_recovery_and_original_failure_separate(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp:
            root=Path(temp)
            names={"plain":"plain-dev","keyword":"keyword-core-dev","rerank":"rerank-recovered-core-dev","metadata_auto":"metadata_auto-core-dev"}
            write_json(root/"scope-current.json",{"core_arms":list(names),"run_paths":names,
                "full150_requested":None,"status":"awaiting_scope_decision"})
            for name in names.values():
                write_json(root/name/"manifest.json",{"identity":{"source_sha256":"相同源码","queries_sha256":"相同题源","upstream":"同版","freeze":"同索引"}})
            original=root/"rerank-core-dev";original.mkdir();(original/"results.jsonl").write_bytes(b"original\n")
            proof={"source_run":original.name,"results_sha256":digest((original/"results.jsonl").read_bytes())}
            write_json(root/"metadata-restore.json",{"verified":True,"documents":84})
            write_json(root/"supervisor-finish.json",{"awaiting_scope_decision":True})
            def details(path):
                if path.name==original.name:
                    return {"run":path.name,"planned":39,"completed":32,"statuses":{"ok":30,"request_error":2},"missing":7,"complete":False}
                return {"run":path.name,"complete":True,**({"recovery":proof} if path.name==names["rerank"] else {})}
            with patch("timeguard.native_report.run_details",side_effect=details),patch("timeguard.native_report.audit_summary",return_value={"available":True,"source_fingerprint_stable":True}):
                result=build(root,root/"unused-audit")
            self.assertEqual(result["status"],"awaiting_scope_decision")
            self.assertIn(names["rerank"],{r["run"] for r in result["core_runs"]})
            self.assertEqual(result["interrupted_runs"][0]["missing"],7)
            self.assertEqual(result["interrupted_runs"][0]["statuses"]["request_error"],2)
            self.assertIsNone(result["full150"])

    def test_core_source_cannot_escape_current_experiment(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp:
            root=Path(temp)
            write_json(root/"scope-current.json",{"core_arms":["plain"],"run_paths":{"plain":"../other"}})
            with self.assertRaises(ValueError):build(root,root/"unused-audit")
