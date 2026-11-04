"""用小型本地PDF和假原生接口验证扩充审计，不调用模型或修改服务。"""

import copy
from pathlib import Path
import tempfile
import unittest

from timeguard.common import digest, read_json, write_json, write_jsonl
from timeguard.extend_corpus import FINANCEBENCH_COMMIT, MODEL_FIELDS, materialize, prepare_inputs, verify_native


class ExtendCorpusTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.full_root, self.old_root = self.root / "full", self.root / "old"
        self.full_root.mkdir()
        self.old_root.mkdir()
        self.full = []
        for index in range(84):
            content = b"%PDF-1.7\n" + str(index).encode()
            name = f"document{index}"
            (self.full_root / (name + ".pdf")).write_bytes(content)
            self.full.append({"doc_name": name, "source_commit": FINANCEBENCH_COMMIT,
                              "pdf_sha256": digest(content), "local_path": name + ".pdf"})
        self.old = [{**self.full[0], "local_path": "../full/document0.pdf"}]
        self.full_ledger, self.old_ledger = self.full_root / "documents.jsonl", self.old_root / "documents.jsonl"
        write_jsonl(self.full_ledger, self.full)
        write_jsonl(self.old_ledger, self.old)
        self.profile = {**{key: "a" * 64 for key in MODEL_FIELDS}, "parser": "DeepDOC", "device": "gpu",
                        "onnx_provider": "CUDAExecutionProvider", "visible_gpu_count": 1, "ocr_gpu_mem_limit_mb": 8192,
                        "ocr_arena_extend_strategy": "kSameAsRequested", "ocr_gpumem_arena_shrinkage": True,
                        "cpus": 16, "memory_gib": 24, "task_executors": 1, "max_concurrent_tasks": 4,
                        "max_concurrent_chunk_builders": 4}
        self.config = {"chunk_token_num": 256, "layout_recognize": "DeepDOC"}
        self.state = {"dataset_id": "dataset", "embedding_model": "fixed-embedding", "corpus_sha256": digest(self.old),
                      "parse_profile": self.profile, "dataset_config": {"chunk_method": "naive", "parser_config": self.config},
                      "documents": {"document0": {"id": "doc", "sha256": self.full[0]["pdf_sha256"], "parse_requested": True}}}
        self.source = self.root / "source.json"
        write_json(self.source, self.state)
        self.target_profile = self.root / "final-profile.json"
        write_json(self.target_profile, {**self.profile, "table_auto_rotate": True, "memory_gib": 48,
                                         "task_executors": 8, "max_concurrent_tasks": 1, "max_concurrent_chunk_builders": 1})
        self.evidence = self.root / "resource-stages.json"
        write_json(self.evidence, [{"stage": "原文档资源加进程", "algorithm_changed": False}])
        models = {"image_config_digest": "a" * 64,
                  "files": [{"name": name, "bytes": 4, "sha256": "b" * 64} for name in
                            ("det.onnx", "rec.onnx", "layout.onnx", "tsr.onnx", "ocr.res", "updown_concat_xgb.model")]}
        parser = {"container_image_config": "a" * 64, "upstream_commit": "8090ffd8ba6098813d7f7ec2c4b8774f36c8423a",
                  "pdf_parser_source_sha256": "c" * 64, "table_auto_rotate_effective": True,
                  "table_auto_rotate_env": None, "default_verified_in_fixed_source": True, "ocr_environment": {"OCR": "same"}}
        self.artifacts = {key: self.root / (key + ".json") for key in ("source_models", "final_models", "source_parser", "final_parser")}
        for key, path in self.artifacts.items():
            value = models if key.endswith("models") else {**parser, **({"table_auto_rotate_env": "true"} if key == "final_parser" else {})}
            write_json(path, value)
        self.target, self.profile_output, self.report = self.root / "target.json", self.root / "corpus-profile.json", self.root / "report.json"

    def prepare(self):
        return prepare_inputs(self.source, self.old_ledger, self.full_ledger, self.target_profile, [self.evidence], self.artifacts)

    def client(self, fault=None):
        outer = self
        class FakeClient:
            def __init__(self):
                self.calls = []

            def request(self, method, path):
                self.calls.append((method, path))
                outer.assertEqual(method, "GET")
                if path == "/datasets/dataset":
                    return {"id": "dataset", "embedding_model": "changed" if fault == "embedding" else "fixed-embedding",
                            "chunk_method": "naive", "parser_config": outer.config}
                doc = {"id": "doc", "name": "document0.pdf", "chunk_count": 102 if fault == "cross_task" else 101, "run": "DONE", "update_time": 1,
                       "chunk_method": "naive", "parser_config": outer.config}
                if "/chunks?" not in path:
                    return {"docs": [doc] + ([dict(doc, id="unexpected")] if fault == "extra_document" else [])}
                first = "page=1&" in path
                chunks = [{"id": str(i), "document_id": "doc", "dataset_id": ["dataset"], "content": "原生块正文",
                           "positions": [[1, 2, 3, 4, 5]], "available": True} for i in (range(100) if first else [100])]
                if fault == "duplicate" and not first:
                    chunks[0]["id"] = "0"
                if fault == "ownership":
                    chunks[0]["dataset_id"] = ["other"]
                if fault in ("image", "empty_text", "image_without_id", "image_without_position"):
                    chunks[0].update(content="", doc_type_kwd="image", image_id="dataset-" + chunks[0]["id"])
                    if fault == "empty_text":
                        chunks[0]["doc_type_kwd"] = "text"
                    elif fault == "image_without_id":
                        chunks[0]["image_id"] = ""
                    elif fault == "image_without_position":
                        chunks[0]["positions"] = []
                return {"total": 102 if fault == "total" else 101, "chunks": chunks, "doc": doc}
        return FakeClient()

    def test_extension_keeps_original_and_only_prepares_83_new_documents(self):
        prepared = self.prepare()
        original = self.source.read_bytes()
        client = self.client()
        native = verify_native(prepared, client)
        self.assertEqual(native["chunk_count"], 101)
        self.assertEqual(sum("/chunks?" in path for _, path in client.calls), 2)
        materialize(prepared, native, self.target, self.profile_output, self.report)
        output = read_json(self.target)
        self.assertEqual(output["corpus_sha256"], digest(self.full))
        self.assertEqual(output["documents"], self.state["documents"])
        self.assertEqual(self.source.read_bytes(), original)
        self.assertEqual(output["parse_profile"], read_json(self.profile_output))
        self.assertEqual(output["parse_profile"]["resource_schedule"]["reused_document_original_profile"]["memory_gib"], 24)
        self.assertEqual(output["parse_profile"]["memory_gib"], 48)
        self.assertEqual(read_json(self.report)["source_files"]["source_state"]["sha256"], digest(original))
        self.assertNotIn("原生块正文", self.report.read_text(encoding="utf-8"))
        with self.assertRaises(ValueError):
            materialize(prepared, native, self.target, self.profile_output, self.report)

    def test_rejects_rotation_and_model_weight_changes(self):
        final = read_json(self.target_profile)
        write_json(self.target_profile, {**final, "table_auto_rotate": False})
        with self.assertRaises(ValueError):
            self.prepare()
        write_json(self.target_profile, final)
        models = read_json(self.artifacts["final_models"])
        models["files"][0]["sha256"] = "d" * 64
        write_json(self.artifacts["final_models"], models)
        with self.assertRaises(ValueError):
            self.prepare()

    def test_rejects_changed_pdf_or_nonmember_document(self):
        (self.full_root / "document0.pdf").write_bytes(b"%PDF-different")
        with self.assertRaises(ValueError):
            self.prepare()
        changed = copy.deepcopy(self.old)
        changed[0]["doc_name"] = "not-in-84"
        write_jsonl(self.old_ledger, changed)
        with self.assertRaises(ValueError):
            self.prepare()

    def test_rejects_native_duplicates_wrong_ownership_counts_and_embedding(self):
        prepared = self.prepare()
        for fault in ("duplicate", "ownership", "total", "extra_document", "embedding"):
            with self.subTest(fault=fault), self.assertRaises(ValueError):
                verify_native(prepared, self.client(fault))

    def test_rejects_input_mutation_after_validation_before_writing(self):
        prepared = self.prepare()
        native = verify_native(prepared, self.client())
        self.evidence.write_bytes(self.evidence.read_bytes() + b"\n")
        with self.assertRaises(ValueError):
            materialize(prepared, native, self.target, self.profile_output, self.report)
        self.assertFalse(self.target.exists())

    def test_only_exact_raw_task_accounting_accepts_cross_task_duplicates(self):
        prepared = self.prepare()
        with self.assertRaises(ValueError):
            verify_native(prepared, self.client("cross_task"))
        tasks = {"dataset_id": "dataset", "document_id": "doc", "tasks": [
            {"id": "first", "doc_id": "doc", "from_page": 0, "to_page": 80, "progress": 1, "retry_count": 1,
             "chunk_ids": " ".join(map(str, range(100))) + " 0"},
            {"id": "second", "doc_id": "doc", "from_page": 80, "to_page": 160, "progress": 1, "retry_count": 1,
             "chunk_ids": "99 100"}]}
        prepared["task_accounting"] = tasks
        report = verify_native(prepared, self.client("cross_task"))
        self.assertEqual(report["native_document_chunk_count"], 102)
        self.assertEqual(report["unique_available_chunk_count"], 101)
        self.assertEqual(report["task_count_reconciliation"]["within_task_duplicate_entries"], 1)
        self.assertEqual(report["task_count_reconciliation"]["cross_task_duplicate_entries"], 1)
        for field, value in (("chunk_ids", "99 missing"), ("from_page", 81), ("progress", 0.9)):
            changed = copy.deepcopy(prepared)
            changed["task_accounting"]["tasks"][1][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                verify_native(changed, self.client("cross_task"))

    def test_keeps_native_image_blocks_and_rejects_other_empty_content(self):
        prepared = self.prepare()
        report = verify_native(prepared, self.client("image"))
        self.assertEqual(report["text_chunk_count"], 99)
        self.assertEqual(report["nontext_image_chunk_count"], 2)
        self.assertEqual(report["unique_available_chunk_count"], 101)
        for fault in ("empty_text", "image_without_id", "image_without_position"):
            with self.subTest(fault=fault), self.assertRaises(ValueError):
                verify_native(prepared, self.client(fault))


if __name__ == "__main__":
    unittest.main()
