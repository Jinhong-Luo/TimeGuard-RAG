"""完整84分母、任务记账与原生分页的离线测试，不调用模型或真实REST。"""

import contextlib
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from timeguard import audit_index
from timeguard.common import digest


class IndexAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = {"chunk_token_num": 256, "layout_recognize": "DeepDOC"}
        self.ledger, self.docs, entries, groups, pages = [], [], {}, [], []
        for index in range(84):
            name, ident = f"report{index}", f"doc{index}"
            raw = b"%PDF-1.7\n" + str(index).encode()
            (self.root / (name + ".pdf")).write_bytes(raw)
            self.ledger.append({"doc_name": name, "local_path": name + ".pdf", "pdf_sha256": digest(raw)})
            entries[name] = {"id": ident, "sha256": digest(raw), "parse_requested": True}
            self.docs.append({"id": ident, "name": name + ".pdf", "run": "DONE", "chunk_count": 3,
                              "size": len(raw), "update_time": 1, "chunk_method": "naive", "parser_config": self.config})
            groups.append({"document_id": ident, "tasks": [
                {"id": ident + "-task1", "doc_id": ident, "from_page": 0, "to_page": 12, "progress": 1,
                 "retry_count": 1, "chunk_ids": ident + "-a " + ident + "-a"},
                {"id": ident + "-task2", "doc_id": ident, "from_page": 12, "to_page": 13, "progress": 1,
                 "retry_count": 1, "chunk_ids": ident + "-a " + ident + "-b"}]})
            pages.append({"name": name + ".pdf", "pages": 13})
        self.state = {"dataset_id": "dataset", "embedding_model": "fixed-embedding", "corpus_sha256": digest(self.ledger),
                      "documents": entries, "dataset_config": {"chunk_method": "naive", "parser_config": self.config},
                      "parse_profile": {"device": "gpu", "table_auto_rotate": True}}
        self.pages = {"documents": 84, "failed_documents": 0, "total_pages": 84 * 13, "document_pages": pages}
        self.tasks = {"dataset_id": "dataset", "recorded_at_utc": "测试时间", "documents": groups}

    def client(self, faults=None, drift=False):
        outer, faults = self, faults or {}
        class FakeClient:
            def __init__(self):
                self.document_reads = 0

            def request(self, method, path):
                outer.assertEqual(method, "GET")
                if path == "/datasets/dataset":
                    return {"id": "dataset", "embedding_model": "fixed-embedding", "chunk_method": "naive",
                            "parser_config": outer.config}
                if "/chunks?" not in path:
                    self.document_reads += 1
                    docs = copy.deepcopy(outer.docs)
                    if drift and self.document_reads > 1:
                        docs[0]["update_time"] = 2
                    return {"docs": docs}
                ident = path.split("/documents/")[1].split("/")[0]
                rows = [{"id": ident + "-a", "document_id": ident, "dataset_id": "dataset", "available": True,
                         "content": "仅用于本地校验的正文，不得出现在报告", "positions": [[1, 2, 3, 4, 5]], "doc_type_kwd": "text"},
                        {"id": ident + "-b", "document_id": ident, "dataset_id": "dataset", "available": True,
                         "content": "", "positions": [[1, 2, 3, 4, 5]], "doc_type_kwd": "image", "image_id": "dataset-" + ident + "-b"}]
                fault = faults.get(ident)
                if fault == "missing":
                    rows.pop()
                elif fault == "empty_text":
                    rows[1]["doc_type_kwd"] = "text"
                elif fault == "wrong_owner":
                    rows[0]["document_id"] = "another"
                return {"total": 2, "doc": {"id": ident}, "chunks": rows}
        return FakeClient()

    def evaluate(self, client=None):
        with contextlib.redirect_stdout(io.StringIO()):
            return audit_index.audit(self.state, self.ledger, self.pages, self.tasks, client or self.client(), self.root)

    def test_full84_keeps_native_cumulative_unique_and_image_counts_separate(self):
        result = self.evaluate()
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["planned_denominator"], 84)
        self.assertEqual(result["passed_documents"], 84)
        totals = result["verified_document_totals"]
        self.assertEqual(totals["metadata_chunk_count"], 252)
        self.assertEqual(totals["unique_available_chunk_count"], 168)
        self.assertEqual(totals["text_chunk_count"], 84)
        self.assertEqual(totals["nontext_image_chunk_count"], 84)
        self.assertEqual(totals["cross_task_duplicate_entries"], 84)
        self.assertNotIn("仅用于本地校验的正文", str(result))

    def test_failed_documents_remain_in84_and_other_documents_continue(self):
        result = self.evaluate(self.client({"doc0": "missing", "doc1": "empty_text", "doc2": "wrong_owner"}))
        self.assertEqual(result["status"], "failed")
        self.assertEqual(len(result["documents"]), 84)
        self.assertEqual(result["planned_denominator"], 84)
        self.assertEqual(result["failed_documents"], 3)
        self.assertEqual(result["passed_documents"], 81)
        self.assertEqual(result["verified_document_totals"]["unique_available_chunk_count"], 162)

    def test_page_gap_and_pdf_change_are_not_hidden(self):
        self.tasks["documents"][0]["tasks"][1]["from_page"] = 11
        (self.root / "report1.pdf").write_bytes("%PDF-更改".encode())
        result = self.evaluate()
        self.assertEqual(result["failed_documents"], 2)
        self.assertEqual(result["passed_documents"], 82)

    def test_snapshot_drift_prevents_global_pass(self):
        result = self.evaluate(self.client(drift=True))
        self.assertEqual(result["passed_documents"], 84)
        self.assertEqual(result["status"], "failed")
        self.assertIn("审计期间", result["global_errors"][0])


class IndexAuditPlanTests(unittest.TestCase):
    def test_all_pages_and_duplicate_ids_are_checked(self):
        class Pages:
            def __init__(self, duplicate=False):
                self.calls = 0
                self.duplicate = duplicate

            def request(self, method, path):
                self.calls += 1
                values = range(100) if self.calls == 1 else [0 if self.duplicate else 100]
                rows = [{"id": str(value), "document_id": "doc", "dataset_id": "dataset", "available": True,
                         "content": "正文", "positions": [], "doc_type_kwd": "text"} for value in values]
                return {"total": 101, "doc": {"id": "doc"}, "chunks": rows}
        client = Pages()
        result = audit_index.inspect_chunks(client, "dataset", "doc", set(map(str, range(101))))
        self.assertEqual(client.calls, 2)
        self.assertEqual(result["unique_available_chunk_count"], 101)
        with self.assertRaises(audit_index.AuditFailure):
            audit_index.inspect_chunks(Pages(duplicate=True), "dataset", "doc", set(map(str, range(101))))

    def test_plan_does_not_require_private_snapshot_or_make_requests(self):
        with patch("sys.argv", ["audit", "--tasks", "尚不存在的本机任务快照.json"]), \
                patch.object(audit_index, "Client") as client, contextlib.redirect_stdout(io.StringIO()):
            audit_index.main()
        client.assert_not_called()


if __name__ == "__main__":
    unittest.main()
