"""离线对齐三层证据与同题标准证据对照；原始页清单仅保存在私有结果中。"""
import argparse
from collections import Counter
from pathlib import Path

from .common import digest, read_json, read_jsonl, write_json
from .judge import build_request
from .score import doc_name


def page_set(chunks):
    pages = set()
    complete = chunks is not None
    for chunk in chunks or []:
        name = doc_name(chunk)
        positions = chunk.get("positions")
        if not name or not chunk.get("document_id") or not isinstance(positions, list) or not positions:
            complete = False
            continue
        for pos in positions:
            if not isinstance(pos, list) or not pos or type(pos[0]) is not int or pos[0] < 1:
                complete = False
            else:
                pages.add((name, pos[0]))
    return pages, complete


def layers(row):
    trace = (row.get("native_trace") or {}).get("evidence_layers") or {}
    retrievals = trace.get("retrievals")
    candidates = [c for call in retrievals or [] for c in call.get("chunks") or []]
    pages, valid = page_set(candidates)
    complete = (row.get("status") == "ok" and trace.get("version") == 2 and isinstance(retrievals, list)
                and all(call.get("chunks") is not None for call in retrievals))
    out = {"candidates": (pages, valid and complete)}
    answers = trace.get("answers") or []
    ids = {(c.get("document_id"), c.get("id")) for c in candidates}
    if any((c.get("document_id"), c.get("id")) not in ids for a in answers for c in a.get("selected") or []):
        out["candidates"] = pages, False
    # 多次合成时只保留事实轨迹，避免将其他答案的证据并入最终回答。
    answer = answers[0] if row.get("status") == "ok" and len(answers) == 1 else {}
    out["selected"] = page_set(answer.get("selected"))
    out["packet"] = page_set(answer.get("packet"))
    requests = [r for r in answer.get("requests", []) if r.get("status") == 200]
    chunks = requests[0].get("complete_chunks") if len(requests) == 1 else None
    if answer.get("empty_response") and not answer["requests"]:
        chunks = []
    out["received"] = page_set(chunks)
    known = [c for c in chunks or [] if c.get("source_check") == "intact"]
    pages, valid = page_set(known)
    out["intact"] = pages, valid and chunks is not None and all(
        c.get("source_check") in ("intact", "edited") for c in chunks or [])
    return out


def assess(row, normal, oracle, gold):
    targets = {(e["doc_name"].casefold(), e["evidence_page_num"] + 1) for e in gold["evidence"]}
    if not targets:
        raise ValueError("诊断题缺少标准证据页")
    stages = layers(row)
    hits = {key: pages & targets for key, (pages, _) in stages.items()}
    flags = {key: complete for key, (_, complete) in stages.items()}
    result = {"id": gold["id"], "gold_pages": sorted(targets), "normal_correct": normal.get("answer_correct"),
              "oracle_correct": oracle.get("answer_correct"), "trace_complete": flags,
              "hit_pages": {k: sorted(v) for k, v in hits.items()}}
    calls = ((row.get("native_trace") or {}).get("evidence_layers") or {}).get("retrievals")
    if normal.get("answer_correct") is True:
        reason = "组合组判对"
    elif normal.get("answer_correct") is not False or oracle.get("answer_correct") is None:
        reason = "裁判未知"
    elif flags["candidates"] and calls == [] and oracle["answer_correct"]:
        reason = "先查未发起检索的原因"
    elif flags["candidates"] and not hits["candidates"] and oracle["answer_correct"]:
        reason = "先查页级召回"
    elif flags["received"] and hits["candidates"] - hits["received"]:
        reason = "先查证据选择或预算"
    elif hits["received"] == targets and oracle["answer_correct"] is False:
        reason = "复查取数、题意或金标"
    elif flags["intact"] and hits["received"] - hits["intact"]:
        reason = "查检索摘录与原块完整性"
    elif not all(flags.values()):
        reason = "观察不足"
    elif hits["received"] == targets and oracle["answer_correct"]:
        reason = "查同页片段与回答用证"
    else:
        reason = "混合原因待复核"
    return {**result, "retrieval_call_count_including_nested": len(calls) if calls is not None else None,
            "next_check": reason}


def documents(chunks):
    if chunks is None:
        return None
    grouped = {}
    for chunk in chunks:
        key = chunk.get("document_id"), doc_name(chunk)
        item = grouped.setdefault(key, {"document_id": key[0], "document_name": key[1], "pages": set()})
        pages, _ = page_set([chunk])
        item["pages"].update(p for _, p in pages)
    return [{**item, "pages": sorted(item["pages"])} for item in grouped.values()]


def page_inventory(row):
    trace = (row.get("native_trace") or {}).get("evidence_layers")
    if not trace:
        return None
    return {"retrievals": [{"call": e["call"], "channel": e.get("channel"), "node": e.get("node"), "search_round": e.get("search_round"),
                            "documents": documents(e["chunks"])} for e in trace["retrievals"]],
            "answers": [{"selected": documents(e["selected"]), "packet": documents(e["packet"]),
                         "requests": [{"status": r["status"], "received": documents(r["complete_chunks"]),
                                       "intact": documents([c for c in r["complete_chunks"] if c.get("source_check") == "intact"])
                                       if r["complete_chunks"] is not None else None}
                                      for r in e["requests"]]} for e in trace["answers"]]}


def indexed(rows):
    out = {r["id"]: r for r in rows}
    if len(out) != len(rows):
        raise ValueError("输入题号重复")
    return out


def reviews(path, ids, gold):
    manifest = read_json(path / "model_reviews.manifest.json")
    ident = manifest["identity"]
    outputs = indexed(read_jsonl(path / "results.jsonl"))
    rows = indexed(read_jsonl(path / "model_reviews.jsonl"))
    if set(ident["expected_ids"]) != set(ids) or outputs.keys() != set(ids) or rows.keys() != set(ids):
        raise ValueError("生成或裁判题号与计划不一致")
    if (digest(ident) != manifest["signature"] or
            digest((path / "results.jsonl").read_bytes()) != ident["generation_results_sha256"]):
        raise ValueError("裁判与生成文件的指纹不匹配")
    for key, review in rows.items():
        if review.get("judge_run_signature") != manifest["signature"]:
            raise ValueError("裁判记录签名不匹配")
        if review["judge_status"] == "ok" and digest(build_request(gold[key], outputs[key])) != review["judge_input_sha256"]:
            raise ValueError("应使用当前评分规则重新评分：" + key)
    return outputs, rows, ident


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normal", required=True, type=Path)
    parser.add_argument("--oracle", required=True, type=Path)
    parser.add_argument("--baseline", type=Path, help="另列历史组合组与同规则标准证据组的配对结果")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--details", required=True, type=Path)
    args = parser.parse_args()
    ids = read_json(args.normal / "manifest.json")["identity"]["query_ids"]
    gold = indexed(read_jsonl("data/financebench/gold.jsonl"))
    generated, nr, nident = reviews(args.normal, ids, gold)
    oracle_rows, gr, gident = reviews(args.oracle, ids, gold)
    for k in ("model", "effort", "version", "schema", "prompt_sha256", "gold_sha256", "transport"):
        if nident[k] != gident[k]:
            raise ValueError("裁判规则不一致：" + k)
    execution = read_json(args.normal / "execution.json")
    if not execution["model_identity_verified"] or execution["actual_calls"] <= execution["failed_calls"]:
        raise ValueError("组合组生成模型未核验")
    ledger = next(r for r in read_jsonl(args.oracle / "audit.jsonl") if r["stage"] == "generation")
    if (not ledger["verified"] or ledger["successful_model_calls"] != len(ids) or not all(
            r["actual_model"] == "gpt-6.1-sol" and r["reasoning_effort"] == "low"
            for r in ledger["records"] if r["success"]) or
            digest((args.oracle / "results.jsonl").read_bytes()) != ledger["results_sha256"]):
        raise ValueError("标准证据回答的模型或数据来源未核验")
    for i in ids:
        chunks = oracle_rows[i]["reference"]["chunks"]
        if ([c["content"] for c in chunks] != [e["evidence_text"] for e in gold[i]["evidence"]]
                or oracle_rows[i].get("retrieved_chunks") or oracle_rows[i]["question"] != gold[i]["question"]
                or generated[i]["question"] != gold[i]["question"]):
            raise ValueError("对照材料与题目不匹配：" + i)
    details = [{**assess(generated[i], nr[i], gr[i], gold[i]), "inventory": page_inventory(generated[i]),
                "normal_reason": nr[i].get("judgement_reason"), "oracle_reason": gr[i].get("judgement_reason")}
               for i in ids]
    stages = {}
    for k in ("candidates", "selected", "packet", "received", "intact"):
        observed = [r for r in details if r["trace_complete"][k]]
        stages[k] = {"observable": len(observed), "unknown": len(ids)-len(observed),
                     "any_hit": sum(bool(r["hit_pages"][k]) for r in observed),
                     "confirmed_any_all_questions": sum(bool(r["hit_pages"][k]) for r in details),
                     "all_hit": sum(len(r["hit_pages"][k]) == len(r["gold_pages"]) for r in observed),
                     "macro_recall_observed": (sum(len(r["hit_pages"][k])/len(r["gold_pages"]) for r in observed)
                                               / len(observed) if observed else None)}
    def counts(key):
        return {label: sum(r[key] is value for r in details)
                for label, value in (("正确", True), ("错误", False), ("未知", None))}
    paired = Counter((r["normal_correct"], r["oracle_correct"]) for r in details)
    transitions = {}
    for src, dst in (("candidates", "selected"), ("selected", "packet"), ("packet", "received"), ("received", "intact")):
        comparable = [r for r in details if r["trace_complete"][src] and r["trace_complete"][dst]]
        n = sum(len(r["hit_pages"][src]) for r in comparable)
        retained = sum(len(set(r["hit_pages"][src]) & set(r["hit_pages"][dst])) for r in comparable)
        transitions[src + "_" + dst] = {"comparable_questions": len(comparable),
                                        "input_gold_pages": n, "retained": retained, "lost": n - retained}
    summary = {"planned": len(ids), "normal": counts("normal_correct"), "oracle": counts("oracle_correct"),
               "paired": [{"normal_correct": n, "oracle_correct": g, "count": v} for (n, g), v in paired.items()],
               "stages": stages, "transitions": transitions, "next_checks": dict(Counter(r["next_check"] for r in details)),
               "normal_wrong_oracle_right": dict(Counter(r["next_check"] for r in details
                     if r["normal_correct"] is False and r["oracle_correct"] is True)),
               "normal_results_sha256": nident["generation_results_sha256"],
               "oracle_results_sha256": gident["generation_results_sha256"]}
    channel_calls = Counter()
    raw_hits = expanded_hits = 0
    for row in details:
        trace = (generated[row["id"]].get("native_trace") or {}).get("evidence_layers") or {}
        raw_pages = set()
        for event in trace.get("retrievals", []):
            channel_calls[event["channel"]] += 1
            if event["channel"] == "retrieval":
                raw_pages.update(page_set(event["chunks"])[0])
        targets = set(row["gold_pages"])
        raw_hits += bool(targets & raw_pages)
        expanded_hits += bool(row["hit_pages"]["candidates"]) and not bool(targets & raw_pages)
    summary["candidate_sources"] = {"call_counts_nested": dict(channel_calls), "native_any_hit_observed": raw_hits,
                                    "additional_any_hit_observed": expanded_hits}
    if args.baseline:
        _, previous, _ = reviews(args.baseline, ids, gold)
        old_spec = read_json(args.baseline / "spec.json")
        new_spec = read_json(args.normal / "spec.json")
        if any(old_spec[k] != new_spec[k] for k in ("arm_config", "dataset_id", "chat_id", "model", "queries")):
            raise ValueError("历史基线与本次观察组的原生配置或题目不同")
        original_errors = [r for r in details if previous[r["id"]].get("answer_correct") is False]
        summary["original_errors"] = {"count": len(original_errors),
            "oracle_correct": sum(r["oracle_correct"] is True for r in original_errors),
            "oracle_wrong": sum(r["oracle_correct"] is False for r in original_errors),
            "oracle_unknown": sum(r["oracle_correct"] is None for r in original_errors),
            "rerun_checks": dict(Counter(r["next_check"] for r in original_errors))}
        for row in details:
            row["baseline_correct"] = previous[row["id"]].get("answer_correct")
    write_json(args.details, details)
    write_json(args.output, summary)


if __name__ == "__main__":
    main()
