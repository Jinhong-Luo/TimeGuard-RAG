from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from timeguard.common import write_jsonl
from timeguard.financebench_paired import COUNTS, PILOT, SPLITS, generation_state, make_plan, summarize
from timeguard.judge import citation_context


class FinanceBenchPairedTest(unittest.TestCase):
    def test_plan_binds_both_arms_each_split_and_hides_gold(self):
        source, lines = [], []
        for split, count in COUNTS.items():
            for offset in range(count):
                number = f"{len(source):05d}"
                ident = f"financebench_id_{number}"
                source.append({"id": ident, "question": "财报问题", "split": split, "answer": "仅评分可见"})
                phase = PILOT if split == "dev" and offset < 6 else "公开正式"
                for order, arm in enumerate(("A0", "A1") if offset % 2 else ("A1", "A0"), 1):
                    lines.append([f"FB-{number}-{arm}", phase, ident, SPLITS[split], arm, order])
        schedule = {"TimeGuard-RAG": {"共享最终证据预算令牌": 8000, "答题及评分位置": lines}}
        plan = make_plan(schedule, source)
        self.assertEqual(len(plan), 300)
        self.assertEqual(sum(p["phase"] == PILOT for p in plan), 12)
        self.assertNotIn("answer", plan[0])
        lines[1][5] = 1
        with self.assertRaisesRegex(ValueError, "顺序"):
            make_plan(schedule, source)

    def test_recovery_counts_inflight_and_detects_duplicate_success(self):
        with TemporaryDirectory() as temp:
            path = Path(temp) / "generation"
            p1, p2 = path / "attempt-0001", path / "attempt-0002"
            start = [{"position": "A0"}, {"position": "A1"}]
            write_jsonl(p1 / "started.jsonl", start)
            write_jsonl(p1 / "results.jsonl", [{"position": "A0", "status": "ok"}])
            write_jsonl(p2 / "started.jsonl", [start[1]])
            write_jsonl(p2 / "results.jsonl", [{"position": "A1", "status": "ok"}])
            spent, rows = generation_state(path, {"A0", "A1"})
            self.assertEqual(dict(spent), {"A0": 1, "A1": 2})
            self.assertEqual(set(rows), {"A0", "A1"})
            write_jsonl(p2 / "started.jsonl", start)
            write_jsonl(p2 / "results.jsonl", [{"position": "A0", "status": "ok"}])
            with self.assertRaisesRegex(ValueError, "重复执行"):
                generation_state(path, {"A0", "A1"})

    def test_incomplete_pair_keeps_the_denominator(self):
        plan = [{"position": a, "id": "q", "split": "dev", "pair_arm": a} for a in ("A0", "A1")]
        summary = summarize(plan, {}, {"A1": {"answer_correct": True, "citation_support": None}})["all"]
        self.assertEqual(summary["arms"]["A0"]["denominator"], 1)
        self.assertEqual(summary["arms"]["A1"]["joint_unknown"], 1)
        self.assertEqual(summary["comparable_pairs"], 0)

    def test_paired_only_uses_index_verified_model_received_citations(self):
        chunk = {"block_sha256": "指纹", "id": "块", "document_id": "报告", "content": "金额"}
        layer = {"paired": "A0", "block_hashes": ["指纹"],
                 "requests": [{"status": 200, "complete_chunks": [chunk]}]}
        row = {"answer": "[ID:1]", "reference": {"chunks": [chunk]},
               "native_trace": {"citation_origin": "model", "evidence_layers": {"answers": [layer]}}}
        self.assertEqual(citation_context(row)["mapping_status"], "incomplete")
        chunk["source_check"] = "intact"
        self.assertEqual(citation_context(row)["mapping_status"], "mapped")
