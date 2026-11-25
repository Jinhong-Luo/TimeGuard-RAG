"""按冻结排期运行 FinanceBench 的原生参照和财报增强组。"""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys

from . import judge, openai_transport
from .common import digest, read_json, read_jsonl, write_json
from .matrix import frozen_index
from .native_baseline import DEFAULT_RERANKER, arm_config, container_worker, private_runtime, worker_sources
from .retrieval_experiment import catalog


PILOT = "首批端到端核验"
SPLITS = {"tune": "历史调优", "dev": "历史开发", "test": "历史跨公司保留"}
COUNTS = {"tune": 88, "dev": 39, "test": 23}
QUERY_PATH = Path("data/financebench/queries.jsonl")
INGESTION_PATH = Path("data/ingestion.json")
UPSTREAM = (
    "api/db/services/dialog_service.py", "rag/prompts/generator.py", "rag/nlp/search.py",
    "common/metadata_utils.py", "rag/advanced_rag/harness/config.py",
    "rag/advanced_rag/agentic_rag_graph.py", "rag/advanced_rag/agentic_rag.py",
    "rag/advanced_rag/harness/action_session.py", "rag/advanced_rag/harness/tools/search.py",
    "rag/advanced_rag/harness/prompts/report_prompt.py",
)


def make_plan(schedule, source):
    rows = judge.unique_rows(source, "查询题单")
    if Counter(q["split"] for q in source) != COUNTS:
        raise ValueError("FinanceBench 分组题数与冻结排期不同")
    section = schedule["TimeGuard-RAG"]
    if section["共享最终证据预算令牌"] != 8000:
        raise ValueError("两组必须共享原生 8,000 token 最终预算")
    plan, answered = [], set()
    positions = section["答题及评分位置"]
    for offset in range(0, len(positions), 2):
        first, second = positions[offset:offset + 2]
        ident = first[2]
        if (first[1:4] != second[1:4] or {first[4], second[4]} != {"A0", "A1"}
                or (first[5], second[5]) != (1, 2) or ident in answered
                or first[3] != SPLITS[rows[ident]["split"]]):
            raise ValueError("同题两组或题目顺序与排期不符")
        answered.add(ident)
        query = {key: rows[ident][key] for key in ("id", "question", "split", "as_of") if key in rows[ident]}
        for position, phase, _, _, arm, order in (first, second):
            if position != f"FB-{ident.rsplit('_', 1)[-1]}-{arm}":
                raise ValueError("位置编号与题号不匹配")
            plan.append({**query, "position": position, "phase": phase, "pair_arm": arm, "order": order})
    pilot = [q for q in plan if q["phase"] == PILOT]
    if answered != rows.keys() or len(pilot) != 12 or any(q["split"] != "dev" for q in pilot):
        raise ValueError("排期须包含完整 150 对和 6 对开发核验题")
    return plan


def save(path, value):
    temporary = path.with_suffix(".tmp")
    write_json(temporary, value)
    temporary.replace(path)


def worker_spec(args):
    setup = read_json(args.setup)
    state = read_json(INGESTION_PATH)
    freeze = frozen_index(INGESTION_PATH, Path("runs/r0-gpu-retrieval-all"), QUERY_PATH)
    if setup["dataset_id"] != state["dataset_id"]:
        raise ValueError("专用对话与冻结索引不一致")
    transport = openai_transport.TRANSPORT if args.model_provider == "openai" else judge.TRANSPORT
    endpoint = setup["bridge_base_url"]
    sources = {name: digest(Path(__file__).with_name(name).read_bytes())
               for name in worker_sources("pair_a1", args.model_provider == "openai")}
    spec = {key: setup[key] for key in ("chat_id", "dataset_id", "model")}
    spec.update(model_provider=args.model_provider, model_endpoint=endpoint, candidate="pair_a1",
                candidate_files=sources, ordered_pairs=True, arm_config=arm_config("combined", DEFAULT_RERANKER),
                concurrency=args.concurrency, timeout_seconds=1200,
                documents=catalog(read_jsonl("data/financebench/documents.jsonl"), state),
                upstream_files={name: digest(Path("vendor/ragflow", name).read_bytes()) for name in UPSTREAM})
    return spec, {"freeze": freeze, "transport": transport, "worker_sha256": digest(spec),
                  "setup_sha256": digest(setup), "evidence_budget": 8000,
                  "judge": {"model": judge.MODEL, "effort": judge.EFFORT, "concurrency": args.concurrency,
                            "timeout_seconds": 360, "schema_sha256": digest(judge.SCHEMA),
                            "instructions_sha256": digest(judge.INSTRUCTIONS)}}


def generation_state(attempts, planned):
    spent, latest = Counter(), {}
    for attempt in sorted(attempts.glob("attempt-*")):
        started_path, rows_path = attempt / "started.jsonl", attempt / "results.jsonl"
        started = {r["position"] for r in read_jsonl(started_path)} if started_path.is_file() else set()
        rows = read_jsonl(rows_path) if rows_path.is_file() else []
        if not started <= planned or len(rows) != len({r["position"] for r in rows}):
            raise ValueError("生成日志含重复位置或排期外位置")
        spent.update(started)
        for row in rows:
            position = row["position"]
            if position not in started or latest.get(position, {}).get("status") == "ok":
                raise ValueError("生成结果缺少开始记录或重复执行成功位置")
            latest[position] = row
    return spent, latest


def generate(out, spec, queries, all_positions):
    attempts = out / "generation"
    attempts.mkdir(exist_ok=True)
    while True:
        spent, latest = generation_state(attempts, all_positions)
        pending = [q for q in queries if latest.get(q["position"], {}).get("status") != "ok"
                   and spent[q["position"]] < 2]
        if not pending:
            return latest
        attempt = attempts / f"attempt-{len(list(attempts.glob('attempt-*'))) + 1:04d}"
        attempt.mkdir()
        print(f"{attempt.name} 开始 {len(pending)} 个答题位置", flush=True)
        code = container_worker({**spec, "queries": pending}, attempt)
        if not (attempt / "started.jsonl").stat().st_size:
            raise RuntimeError(f"运行器未开始题目（退出码 {code}），见 {attempt / 'worker.log'}")
        save(attempt / "exit.json", {"returncode": code, "planned": len(pending)})


def rule_judgement(row, why):
    explicit = row.get("answer_status") in ("clarification", "no_evidence")
    return {"judge_type": "rule", "judge_status": why, "answer_correct": False if explicit else None,
            "citation_support": None, "model_call_attempted": False, "generation_status": row.get("status", "missing")}


def grade(out, queries, generated, signature, spec):
    folder = out / "judgements"
    folder.mkdir(exist_ok=True)
    gold = judge.unique_rows(read_jsonl("data/financebench/gold.jsonl"), "评分金标")
    official = spec["model_provider"] == "openai"
    if official:
        transport, fn, base, key = openai_transport.TRANSPORT, openai_transport.post_completion, None, openai_transport.api_key()
    else:
        _, settings = private_runtime()
        transport, fn, base, key = judge.TRANSPORT, judge.post_completion, settings["base_url"], settings["token"]

    def one(query):
        ident, position = query["id"], query["position"]
        path = folder / f"{position}.json"
        row = generated.get(position, {})
        source_digest = digest(row)
        if path.exists():
            result = read_json(path)
            if result["generation_sha256"] != source_digest or result["signature"] != signature:
                raise ValueError("续跑裁判与本次答题记录不一致")
            return result
        began = folder / f"{position}.started"
        if began.exists():
            result = rule_judgement(row, "评分中断")
        elif row.get("status") == "ok" and row.get("answer_status") == "completed":
            began.touch(exist_ok=False)
            result = judge.judge_one(gold[ident], row, signature, base, key, 360, fn, transport)
        else:
            result = rule_judgement(row, row.get("answer_status", "生成失败或缺失"))
        result.update(position=position, id=ident, signature=signature, generation_sha256=source_digest)
        save(path, result)
        print(f"{position} 评分 {result['judge_status']}", flush=True)
        return result

    with ThreadPoolExecutor(max_workers=spec["concurrency"]) as pool:
        return {r["position"]: r for r in pool.map(one, queries)}


def joint(row):
    fields = (row.get("answer_correct"), row.get("citation_support"))
    if False in fields:
        return False
    return True if fields == (True, True) else None


def summarize(plan, generated, judgements):
    def group(queries):
        pairs = {}
        arms = {"A0": Counter(), "A1": Counter()}
        for query in queries:
            position, arm = query["position"], query["pair_arm"]
            decision = judgements.get(position, {})
            value = joint(decision)
            row = generated.get(position, {})
            events = [e for e in row.get("native_trace", {}).get("events", []) if e["kind"] == "openai_request"]
            pairs.setdefault(query["id"], {})[arm] = value
            arms[arm].update(denominator=1, joint_pass=value is True, joint_unknown=value is None,
                             answer_pass=decision.get("answer_correct") is True,
                             citation_pass=decision.get("citation_support") is True,
                             generation_error=row.get("status") != "ok",
                             generation_api_calls=sum(e["attempted"] for e in events),
                             generation_input_tokens=sum((e.get("raw_usage") or {}).get("input_tokens", 0) for e in events),
                             generation_output_tokens=sum((e.get("raw_usage") or {}).get("output_tokens", 0) for e in events),
                             judge_input_tokens=(decision.get("usage") or {}).get("prompt_tokens", 0),
                             judge_output_tokens=(decision.get("usage") or {}).get("completion_tokens", 0),
                             judge_calls=decision.get("model_call_attempted", False))
        transitions = Counter((p["A0"], p["A1"]) for p in pairs.values())
        return {"arms": arms, "stable_fail": transitions[False, False], "stable_pass": transitions[True, True],
                "gain": transitions[False, True], "loss": transitions[True, False],
                "net_gain": transitions[False, True] - transitions[True, False],
                "comparable_pairs": sum(n for key, n in transitions.items() if None not in key)}

    return {"all": group(plan), **{split: group([q for q in plan if q["split"] == split]) for split in SPLITS}}


def pilot_gate(pilot, generated, reviews, official):
    rows = [generated.get(q["position"], {}) for q in pilot]
    layers = [a for row in rows for a in row.get("native_trace", {}).get("evidence_layers", {}).get("answers", [])]
    enhancement = [a["enhancement"] for a in layers if a.get("enhancement")]
    events = [e for row in rows for e in row.get("native_trace", {}).get("events", []) if e["kind"] == "openai_request"]
    return {
        "all_answer_positions": all(r.get("status") == "ok" for r in rows),
        "all_grading_positions": len(reviews) == len(pilot) and all(
            r["judge_status"] in ("ok", "clarification", "no_evidence") for r in reviews.values()),
        "live_fact_selection": any(e.get("selected_facts") and e["path"] == "事实筛选" for e in enhancement),
        "native_model_tool_round": not official or any(e["success"] and e.get("model_tool_results", 0) for e in events),
        "actual_citations_traced": any(a.get("block_hashes") and any(
            c.get("source_check") == "intact" for r in a["requests"] for c in r.get("complete_chunks") or []) for a in layers),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", required=True, type=Path)
    parser.add_argument("--setup", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stage", choices=("plan", "pilot", "all"), default="pilot")
    parser.add_argument("--model-provider", choices=("codex", "openai"), default="codex")
    parser.add_argument("--concurrency", type=int, choices=range(1, 5), default=2)
    parser.add_argument("--program-report", type=Path)
    args = parser.parse_args()
    plan = make_plan(read_json(args.schedule), read_jsonl(QUERY_PATH))
    if args.stage == "plan":
        print(json.dumps({"题目": len(plan) // 2, "位置": len(plan), "首批": sum(q["phase"] == PILOT for q in plan)}, ensure_ascii=False))
        return
    import fcntl
    checks = read_json(args.program_report) if args.program_report else {}
    if set(checks) != {f"F{i:02d}" for i in range(1, 21)} or not all(v.get("passed") is True for v in checks.values()):
        parser.error("试跑前需要本轮全部二十个程序检查通过")
    os.umask(0o077)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        spec, identity = worker_spec(args)
        identity.update(plan_sha256=digest(plan), program_report_sha256=digest(checks))
        signature = digest(identity)
        manifest = {"identity": identity, "signature": signature}
        path = args.output / "manifest.json"
        if path.exists() and read_json(path) != manifest:
            raise ValueError("实验配置已经变化，请使用新目录")
        save(path, manifest)
        positions = {q["position"] for q in plan}
        pilot = [q for q in plan if q["phase"] == PILOT]
        generated = generate(args.output, spec, pilot, positions)
        decisions = grade(args.output, pilot, generated, signature, spec)
        gate = pilot_gate(pilot, generated, decisions, args.model_provider == "openai")
        save(args.output / "pilot.json", {"gate": gate, "summary": summarize(pilot, generated, decisions)})
        print(json.dumps({"首批验收": gate}, ensure_ascii=False), flush=True)
        if not all(gate.values()):
            sys.exit("首批未达到接线验收条件，结果已保存")
        if args.stage == "all":
            generated = generate(args.output, spec, plan, positions)
            decisions = grade(args.output, plan, generated, signature, spec)
            save(args.output / "summary.json", summarize(plan, generated, decisions))


if __name__ == "__main__":
    main()
