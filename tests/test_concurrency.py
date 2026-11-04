"""验证有限并发的题目隔离、失败留存及签名冻结；不连接真实服务。"""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

from timeguard import judge, run
from timeguard.common import digest, read_json, read_jsonl, write_json, write_jsonl


class ConcurrencyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = {"base_url": "http://example.invalid", "dataset_ids": ["dataset"], "model": judge.MODEL,
                       "retrieval": {"page_size": 6, "top_k": 10, "similarity_threshold": 0.2,
                                     "vector_similarity_weight": 0.3}}

    def test_generation_isolated_clients_failure_and_resume(self):
        barrier, clients = threading.Barrier(2), []
        main_thread = threading.get_ident()
        queries = [{"id": ident, "split": "dev", "question": ident,
                    "answer": "禁止进入生成端的金标", "evidence": ["秘密证据"]} for ident in ("q1", "q2")]

        class FakeClient:
            def __init__(self, *args, **kwargs):
                self.calls = []
                self.thread = threading.get_ident()
                clients.append(self)

            def request(self, method, path, body):
                self.calls.append((path, body))
                if path == "/retrieval":
                    barrier.wait(timeout=5)
                    return {"chunks": []}
                if body["question"] == "q2":
                    raise RuntimeError("模拟生成失败")
                return {"answer": "独立答案", "session_id": "session-q1"}

        output = self.root / "results.jsonl"
        with contextlib.redirect_stdout(io.StringIO()):
            run.run_queries(queries, self.config, "chat", output, concurrency=2, client_factory=FakeClient)
        rows = {row["id"]: row for row in read_jsonl(output)}
        self.assertEqual(set(rows), {"q1", "q2"})
        self.assertEqual(rows["q1"]["status"], "ok")
        self.assertEqual(rows["q2"]["status"], "request_error")
        self.assertTrue(all(row["elapsed_seconds"] > 0 for row in rows.values()))
        self.assertEqual(len(clients), 2)
        self.assertEqual(len({client.thread for client in clients}), 2)
        for client in clients:
            self.assertNotEqual(client.thread, main_thread)
            self.assertEqual(len({body["question"] for _, body in client.calls}), 1)
            for path, body in client.calls:
                self.assertNotIn("session_id", body)
                self.assertNotIn("禁止进入生成端", json.dumps(body, ensure_ascii=False))
                self.assertNotIn("秘密证据", json.dumps(body, ensure_ascii=False))
        before = output.read_bytes()
        forbidden = Mock(side_effect=AssertionError("续跑不应重复已记录请求"))
        run.run_queries(queries, self.config, "chat", output, concurrency=2, client_factory=forbidden)
        self.assertEqual(output.read_bytes(), before)
        forbidden.assert_not_called()

    def response(self):
        judgement = {"answer_correct": True, "numeric_and_scope_correct": None, "citation_support": False,
                     "has_factual_claims": True, "judgement_reason": "答案正确但没有显式引用。", "failure_category": "缺少引用"}
        return {"model": judge.MODEL,
                "bridge_metadata": {"transport": judge.TRANSPORT, "reasoning_effort": judge.EFFORT,
                                    "actual_model": judge.MODEL, "api_equivalent": False},
                "choices": [{"message": {"content": json.dumps(judgement)}}], "usage": {"total_tokens": 3}}

    def test_judge_parallel_requests_failure_skip_and_resume(self):
        ids = ["q1", "q2", "q3"]
        gold = {ident: {"id": ident, "question": ident, "answer": "金标" + ident} for ident in ids}
        generated = {ident: {"id": ident, "status": "ok", "answer": "生成" + ident} for ident in ids[:2]}
        barrier, requests = threading.Barrier(2), []

        def request(base_url, payload, token, timeout):
            requests.append((payload, threading.get_ident()))
            barrier.wait(timeout=5)
            if json.loads(payload["messages"][1]["content"])["question"] == "q2":
                raise RuntimeError("模拟裁判失败")
            return self.response()

        output = self.root / "reviews.jsonl"
        with contextlib.redirect_stdout(io.StringIO()):
            judge.judge_queries(ids, gold, generated, "signature", output, "url", "token", 1,
                                concurrency=2, request_fn=request)
        rows = {row["id"]: row for row in read_jsonl(output)}
        self.assertEqual(rows["q1"]["judge_status"], "ok")
        self.assertEqual(rows["q2"]["judge_status"], "failed")
        if "q3" in rows:
            self.assertEqual(rows["q3"]["judge_status"], "skipped_generation_failure")
        self.assertEqual(len(requests), 2)
        self.assertEqual(len({thread for _, thread in requests}), 2)
        for payload, _ in requests:
            self.assertEqual(len(payload["messages"]), 2)
            self.assertNotIn("session_id", payload)
            material = json.loads(payload["messages"][1]["content"])
            ident = material["question"]
            self.assertEqual(material["candidate_answer"], "生成" + ident)
            self.assertEqual(material["reference_answer"], "金标" + ident)
        before = output.read_bytes()
        forbidden = Mock(side_effect=AssertionError("续跑不能覆盖失败或成功判断"))
        judge.judge_queries(ids, gold, generated, "signature", output, "url", "token", 1,
                            concurrency=2, request_fn=forbidden)
        self.assertEqual(before, output.read_bytes())
        forbidden.assert_not_called()

    def test_generation_manifest_freezes_concurrency_and_plan(self):
        config, ingestion, queries = [self.root / name for name in ("config.json", "ingestion.json", "queries.jsonl")]
        write_json(config, self.config)
        write_json(ingestion, {"dataset_id": "dataset", "corpus_sha256": "hash", "embedding_model": "embedding",
                               "documents": {"doc": {"id": "doc"}}})
        write_jsonl(queries, [{"id": ident, "question": ident, "split": "dev"} for ident in ("q1", "q2")])
        output = self.root / "run"
        argv = ["run", "--config", str(config), "--ingestion", str(ingestion), "--queries", str(queries),
                "--output", str(output), "--retrieval-only", "--concurrency", "2"]
        client = Mock()
        client.request.return_value = [{"id": "doc", "run": "DONE", "chunk_count": 1}]
        def read(path):
            return {"version": "fixed"} if str(path) == "upstream.lock.json" else read_json(path)
        with patch("sys.argv", argv), patch.object(run, "read_json", side_effect=read), \
                patch.object(run, "Client", return_value=client), patch.object(run, "run_queries") as worker:
            run.main()
            self.assertEqual(worker.call_args.args[-1], 2)
            identity = read_json(output / "manifest.json")["identity"]
            self.assertEqual(identity["concurrency"], 2)
            self.assertEqual(identity["query_ids"], ["q1", "q2"])
            self.assertEqual(identity["full_split_count"], 2)
            argv[-1] = "1"
            with self.assertRaises(ValueError):
                run.main()
            self.assertEqual(worker.call_count, 1)

    def test_judge_manifest_freezes_concurrency_and_keeps_full_denominator(self):
        ids = ["q1", "q2", "q3"]
        write_json(self.root / "manifest.json", {"identity": {"query_ids": ids}})
        write_jsonl(self.root / "results.jsonl", [{"id": ident, "status": "ok", "answer": ident} for ident in ids[:2]])
        gold = self.root / "gold.jsonl"
        write_jsonl(gold, [{"id": ident, "question": ident, "answer": "金标"} for ident in ids])
        argv = ["judge", str(self.root), "--gold", str(gold), "--concurrency", "2", "--execute"]
        def request(base_url, payload, token, timeout):
            if json.loads(payload["messages"][1]["content"])["question"] == "q2":
                raise RuntimeError("模拟失败")
            return self.response()
        with patch("sys.argv", argv), patch.object(judge, "post_completion", side_effect=request), \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as exited:
                judge.main()
            self.assertEqual(exited.exception.code, 1)
            manifest = read_json(self.root / "model_reviews.manifest.json")
            self.assertEqual(manifest["identity"]["concurrency"], 2)
            self.assertEqual(manifest["signature"], digest(manifest["identity"]))
            summary = read_json(self.root / "model_reviews.summary.json")
            self.assertEqual(summary["denominator"], 3)
            self.assertEqual(summary["ok"], 1)
            self.assertEqual(summary["failed_or_skipped"] + summary["missing"], 2)
            argv[-2] = "1"
            with self.assertRaises(ValueError):
                judge.main()

    def test_generation_failure_stops_new_requests_and_drains_inflight(self):
        self.check_fail_fast(False)

    def test_judge_failure_stops_new_requests_and_drains_inflight(self):
        self.check_fail_fast(True)

    def check_fail_fast(self, judging):
        output = self.root / "bounded.jsonl"
        queries = [{"id": f"q{i}", "question": f"q{i}", "split": "dev"} for i in range(39)]
        calls = []
        status_key = "judge_status" if judging else "status"

        def worker(query, *args):
            ident = query["id"]
            calls.append(ident)
            if ident != "q0":
                deadline = time.monotonic() + 5
                # 确认失败行已落盘，再让另一个在途请求结束。
                while not (output.exists() and output.stat().st_size):
                    if time.monotonic() >= deadline:
                        raise AssertionError("失败记录未及时落盘")
                    time.sleep(0.01)
            return {"id": ident, status_key: "failed" if ident == "q0" else "ok",
                    "error": "UsageLimitExceeded" if ident == "q0" else None,
                    "judge_type": "model", "judge_run_signature": "signature", "elapsed_seconds": 1}

        def execute():
            if judging:
                by_id = {row["id"]: row for row in queries}
                return judge.judge_queries(list(by_id), by_id, by_id, "signature", output,
                                           "url", "token", 1, concurrency=2)
            return run.run_queries(queries, self.config, "chat", output, concurrency=2)

        target = "timeguard.judge.judge_one" if judging else "timeguard.run.run_one"
        with patch(target, side_effect=worker), contextlib.redirect_stdout(io.StringIO()):
            self.assertFalse(execute())
        self.assertEqual(set(calls), {"q0", "q1"})
        rows = read_jsonl(output)
        self.assertEqual({row["id"] for row in rows}, {"q0", "q1"})
        self.assertEqual(rows[0]["error"], "UsageLimitExceeded")
        before = output.read_bytes()
        with patch(target, side_effect=AssertionError("失败续跑不得继续发题")) as forbidden:
            self.assertFalse(execute())
        forbidden.assert_not_called()
        self.assertEqual(output.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
