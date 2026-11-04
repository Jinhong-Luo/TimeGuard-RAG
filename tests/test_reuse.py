"""用本地假服务验证39题开发集到150题全量的严格复用，不调用真实模型。"""

import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from timeguard import judge, run
from timeguard.common import digest, read_json, read_jsonl, write_json, write_jsonl


class ReuseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dev, self.all = self.root / "dev", self.root / "all"
        self.queries = [{"id": f"q{i}", "question": f"问题{i}",
                         "split": "dev" if i < 39 else "tune" if i < 127 else "test"} for i in range(150)]
        config = {"base_url": "http://example.invalid", "model": judge.MODEL, "dataset_ids": ["dataset"],
                  "efforts_to_probe": ["low", "medium", "high", "xhigh", "max", "ultra"],
                  "retrieval": {"page_size": 6, "top_k": 10, "similarity_threshold": 0.2, "vector_similarity_weight": 0.3}}
        write_json(self.root / "config.json", config)
        write_json(self.root / "ingestion.json", {"dataset_id": "dataset", "corpus_sha256": "fixed-corpus",
                                                 "embedding_model": "fixed-embedding", "documents": {"doc": {"id": "doc"}}})
        write_jsonl(self.root / "queries.jsonl", self.queries)
        write_jsonl(self.root / "gold.jsonl", [{**query, "answer": "标准答案"} for query in self.queries])
        write_json(self.root / "proof.json", {"model": judge.MODEL, "verified_efforts": ["low"]})
        self.calls = []
        outer = self

        class FakeClient:
            def __init__(self, *args, **kwargs):
                pass

            def request(self, method, path, body=None):
                if method == "GET":
                    outer.assertIn("page_size=100", path)
                    return [{"id": "doc", "run": "DONE", "chunk_count": 1, "update_time": "frozen"}]
                if path == "/chats":
                    return {"id": "chat", "llm_setting": body["llm_setting"]}
                if path == "/retrieval":
                    return {"chunks": []}
                outer.calls.append(body)
                return {"answer": "答案" + body["question"] + "[ID:0]", "session_id": body["question"],
                        "reference": {"chunks": [{"content": "对应原文"}]}}

        self.FakeClient = FakeClient
        self.generate("dev", self.dev)
        self.selection = self.root / "selection.json"
        candidates = [{"effort": effort, "completed": 39, "denominator": 39, "status_counts": {"ok": 39},
                       "model_reviewed": 39, "reviewed": 0, "model_answer_accuracy": 1.0, "mean_seconds": offset + 1}
                      for offset, effort in enumerate(config["efforts_to_probe"])]
        write_json(self.selection, {"chosen_effort": "low", "review_source": "model",
                                    "selection_status": "模型裁判暂定推荐，未完成人工抽查", "quality_tolerance": 0.02,
                                    "candidates": candidates,
                                    "source_runs": [{"effort": "low", "manifest": digest((self.dev / "manifest.json").read_bytes()),
                                                     "results": digest((self.dev / "results.jsonl").read_bytes())}]})
        self.calls.clear()

    def generate(self, split, output, extra=()):
        argv = ["run", "--config", str(self.root / "config.json"), "--queries", str(self.root / "queries.jsonl"),
                "--ingestion", str(self.root / "ingestion.json"), "--transport-proof", str(self.root / "proof.json"),
                "--effort", "low", "--split", split, "--concurrency", "2", "--output", str(output), *extra]
        def read(path):
            return {"version": "fixed"} if str(path) == "upstream.lock.json" else read_json(path)
        with patch("sys.argv", argv), patch.object(run, "read_json", side_effect=read), \
                patch.object(run, "Client", self.FakeClient), contextlib.redirect_stdout(io.StringIO()):
            run.main()

    def generate_all(self, extra=()):
        self.generate("all", self.all, ["--reuse-run", str(self.dev), "--selection", str(self.selection), *extra])

    def response(self, *args):
        judgement = {"answer_correct": True, "numeric_and_scope_correct": None, "citation_support": True,
                     "has_factual_claims": True, "judgement_reason": "答案和引用一致。", "failure_category": "未发现错误"}
        return {"model": judge.MODEL, "bridge_metadata": {"transport": judge.TRANSPORT, "reasoning_effort": judge.EFFORT,
                                                           "actual_model": judge.MODEL, "api_equivalent": False},
                "choices": [{"message": {"content": json.dumps(judgement)}}], "usage": {"total_tokens": 100}}

    def evaluate(self, root, extra=()):
        argv = ["judge", str(root), "--gold", str(self.root / "gold.jsonl"), "--concurrency", "2", "--execute", *extra]
        with patch("sys.argv", argv), patch.object(judge, "post_completion", side_effect=self.response) as request, \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as exited:
                judge.main()
            self.assertEqual(exited.exception.code, 0)
            if root == self.dev:
                # 模拟选档时冻结第一次完成的裁判文件，后续重跑不得更新旧选择。
                selection = read_json(self.selection)
                selected = selection["source_runs"][0]
                if "model_reviews" not in selected:
                    selected.update(model_reviews=digest((root / "model_reviews.jsonl").read_bytes()),
                                    model_reviews_manifest=digest((root / "model_reviews.manifest.json").read_bytes()))
                    write_json(self.selection, selection)
            return request.call_count

    def test_generation_reuses_39_only_calls_111_and_keeps_original_costs(self):
        source_bytes = (self.dev / "results.jsonl").read_bytes()
        self.generate_all()
        self.assertEqual(len(self.calls), 111)
        rows = read_jsonl(self.all / "results.jsonl")
        self.assertEqual(len(rows), 150)
        reused = [row for row in rows if "reuse_origin" in row]
        self.assertEqual(len(reused), 39)
        source_rows = read_jsonl(self.dev / "results.jsonl")
        source_by_id = {row["id"]: row for row in source_rows}
        for row in reused:
            self.assertEqual({key: value for key, value in row.items() if key != "reuse_origin"}, source_by_id[row["id"]])
        source_hashes = {digest(line) for line in source_bytes.splitlines()}
        self.assertEqual({row["reuse_origin"]["row_sha256"] for row in reused}, source_hashes)
        manifest = read_json(self.all / "manifest.json")
        self.assertEqual(len(manifest["identity"]["query_ids"]), 150)
        self.assertEqual(manifest["index_snapshot_sha256"], digest(manifest["index_snapshot"]))
        self.assertGreater(manifest["identity"]["reuse"]["source_wall_time_seconds"], 0)
        execution = read_jsonl(self.all / "execution_history.jsonl")[0]
        self.assertEqual(execution["new_result_count"], 111)
        self.assertEqual(execution["reused_result_count"], 39)
        self.calls.clear()
        self.generate_all()
        self.assertEqual(self.calls, [])
        self.assertEqual(source_bytes, (self.dev / "results.jsonl").read_bytes())
        self.assertEqual(len(read_jsonl(self.all / "results.jsonl")), 150)

    def test_generation_rejects_changed_selection_concurrency_or_source(self):
        original = read_json(self.selection)
        for mutation in ("choice", "candidate", "hash"):
            selection = copy.deepcopy(original)
            if mutation == "choice":
                selection["chosen_effort"] = "high"
            elif mutation == "candidate":
                selection["candidates"][0]["model_reviewed"] = 38
            else:
                selection["source_runs"][0]["results"] = digest("其他结果")
            write_json(self.selection, selection)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.generate_all()
        write_json(self.selection, original)
        with self.assertRaises(ValueError):
            self.generate_all(["--concurrency", "1"])
        self.assertEqual(self.calls, [])

    def test_judge_reuses_exact_input_and_rebinds_39_without_new_calls(self):
        self.assertEqual(self.evaluate(self.dev), 39)
        source = self.dev / "model_reviews.jsonl"
        source_bytes = source.read_bytes()
        self.generate_all()
        extra = ["--reuse-reviews", str(source)]
        self.assertEqual(self.evaluate(self.all, extra), 111)
        rows = read_jsonl(self.all / "model_reviews.jsonl")
        manifest = read_json(self.all / "model_reviews.manifest.json")
        self.assertEqual(len(rows), 150)
        self.assertEqual(sum("reuse_origin" in row for row in rows), 39)
        self.assertTrue(all(row["judge_run_signature"] == manifest["signature"] for row in rows))
        self.assertTrue(all(row["usage"]["total_tokens"] == 100 for row in rows))
        summary = read_json(self.all / "model_reviews.summary.json")
        self.assertEqual(summary["denominator"], 150)
        self.assertEqual(summary["reused_count"], 39)
        execution = read_jsonl(self.all / "model_reviews.execution.jsonl")[0]
        self.assertEqual(execution["new_model_call_count"], 111)
        self.assertEqual(execution["reused_result_count"], 39)
        self.assertEqual(self.evaluate(self.all, extra), 0)
        self.assertEqual(source.read_bytes(), source_bytes)
        generated = read_jsonl(self.all / "results.jsonl")
        next(row for row in generated if row["id"] == "q0")["answer"] = "被改变的答案"
        write_jsonl(self.all / "results.jsonl", generated)
        with self.assertRaises(ValueError):
            self.evaluate(self.all, extra)

    def test_judge_rejects_changed_judge_concurrency(self):
        self.evaluate(self.dev)
        self.generate_all()
        with self.assertRaises(ValueError):
            self.evaluate(self.all, ["--reuse-reviews", str(self.dev / "model_reviews.jsonl"), "--concurrency", "1"])

    def test_judge_timeout_is_signed_and_cannot_change_on_resume_or_reuse(self):
        self.evaluate(self.dev)
        identity = read_json(self.dev / "model_reviews.manifest.json")["identity"]
        self.assertEqual(identity["request_timeout_seconds"], 360)
        with self.assertRaises(ValueError):
            self.evaluate(self.dev, ["--timeout", "420"])
        self.generate_all()
        with self.assertRaises(ValueError):
            self.evaluate(self.all, ["--reuse-reviews", str(self.dev / "model_reviews.jsonl"), "--timeout", "420"])

    def test_reuse_rejects_self_consistent_review_replacement_after_selection(self):
        self.evaluate(self.dev)
        original_generation = (self.dev / "results.jsonl").read_bytes()
        source = self.dev / "model_reviews.jsonl"
        rows = read_jsonl(source)
        rows[0]["answer_correct"] = False
        parsed = json.loads(rows[0]["raw_judgement"])
        parsed["answer_correct"] = False
        rows[0]["raw_judgement"] = json.dumps(parsed)
        write_jsonl(source, rows)
        # 生成端只核对选择元数据，不读取裁判；真正的旧裁判文件绑定在裁判端核验。
        self.generate_all()
        self.assertEqual(original_generation, (self.dev / "results.jsonl").read_bytes())
        with self.assertRaisesRegex(ValueError, "冻结选档"):
            self.evaluate(self.all, ["--reuse-reviews", str(source)])

    def test_reuse_rejects_changed_review_manifest_bytes(self):
        self.evaluate(self.dev)
        source = self.dev / "model_reviews.jsonl"
        manifest_path = source.with_suffix(".manifest.json")
        manifest = read_json(manifest_path)
        manifest["created_at_utc"] = "另一次自洽裁判运行"
        write_json(manifest_path, manifest)
        self.generate_all()
        with self.assertRaisesRegex(ValueError, "冻结选档"):
            self.evaluate(self.all, ["--reuse-reviews", str(source)])

    def test_bad_saved_chat_effort_is_rejected_on_resume(self):
        manifest_path = self.dev / "manifest.json"
        manifest = read_json(manifest_path)
        manifest["chat_config"]["llm_setting"]["reasoning_effort"] = "high"
        write_json(manifest_path, manifest)
        with self.assertRaises(ValueError):
            self.generate("dev", self.dev)
        self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main()
