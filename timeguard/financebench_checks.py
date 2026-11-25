"""FinanceBench 配对实验的二十个程序位置。使用冻结资料和回放。"""

import argparse
from decimal import Decimal
import io
from pathlib import Path
import re
import subprocess
import unittest

from .common import Client, digest, read_json, read_jsonl, write_json
from .native_baseline import CONTAINER
from .retrieval_experiment import catalog
from .scope import rank, scope


TESTS = {
    "F07": ("test_live_finance.LiveFinanceTest.test_f07_sufficient_originals_stop_before_selection_and_search",),
    "F08": ("test_live_finance.LiveFinanceTest.test_f08_existing_sources_are_read_and_used",),
    "F09": ("test_live_finance.LiveFinanceTest.test_f09_supplement_is_once_and_same_scope",),
    "F10": ("test_live_finance.LiveFinanceTest.test_f10_supplement_failure_falls_back",),
    "F11": ("test_live_finance.LiveFinanceTest.test_f11_trimmed_evidence_falls_back_before_final_answer",
            "test_final_snapshot.SnapshotTest.test_native_numbers_follow_received_blocks"),
    "F13": ("test_r1.AppTests.test_calculation_guards", "test_formula_guard", "test_fact_selection"),
    "F14": ("test_finance_route.RoutingTest.test_no_guess_on_ambiguous_currency",
            "test_agentic_candidate.FinancialTests.test_bad_json_and_retry"),
    "F15": ("test_r1.AppTests.test_real_graph_http_replay_restart", "test_history_read"),
    "F16": ("test_r1.AppTests.test_real_graph_http_replay_restart",),
    "F17": ("test_r1.AppTests.test_capacity_timeout_and_recovery", "test_r1.AppTests.test_circuit_probe"),
    "F19": ("test_financebench_paired", "test_four_feedback.FeedbackTest.test_recovery_only_reuses_identical_model_request"),
    "F20": ("test_native_baseline", "test_judge.JudgeTests.test_only_actual_citations_enter_judge",
            "test_openai_transport"),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pdf_and_sources(state):
    path = Path("data/financebench/pdfs/3M_2018_10K.pdf")
    actual = digest(path.read_bytes())
    require(actual == state["documents"]["3M_2018_10K"]["sha256"], "3M 年报 PDF 指纹变化")
    text = subprocess.check_output(["pdftotext", "-layout", str(path), "-"], encoding="utf-8")
    pages = [i for i, page in enumerate(text.split("\f"), 1)
             if "2018" in page and "2017" in page and re.search(r"Net\s+sales[^\n]+32,765[^\n]+31,657", page, re.I)]
    require(pages, "PDF 中无法定位期列和净销售额")
    return {"pdf_sha256": actual, "table_pages": pages}


def retrieval(state, docs):
    parser = state["dataset_config"]["parser_config"]["layout_recognize"]
    engine = subprocess.check_output(["docker", "inspect", "-f",
        '{{range .Config.Env}}{{if eq . "DOC_ENGINE=elasticsearch"}}{{println .}}{{end}}{{end}}', CONTAINER],
        encoding="utf-8").strip()
    require(parser == "DeepDOC" and engine == "DOC_ENGINE=elasticsearch" and
            state["embedding_model"].startswith("BAAI/bge-small-en-v1.5@"), "后端组件与冻结索引不符")
    config = read_json("configs/r0-low.json")
    client = Client(config["base_url"], timeout=60)
    plan = scope("3M FY2018 consolidated net sales", docs)
    require(plan["companies"] == ["3M"], "公司识别失败")
    require(scope("Advanced Micro Devices FY2022 quarterly results", docs)["companies"] == ["AMD"], "公司别名识别失败")
    base = {**config["retrieval"], "dataset_ids": [state["dataset_id"]], "document_ids": plan["document_ids"],
            "question": "3M FY2018 consolidated net sales", "keyword": False}
    counts = {}
    for name, weight in (("词项权重", 0.0), ("向量权重", 1.0)):
        result = client.request("POST", "/retrieval", {**base, "vector_similarity_weight": weight})
        chunks = result["chunks"]
        require(chunks and all(c["document_id"] in plan["document_ids"] for c in chunks), "检索通路无结果或越界")
        counts[name] = len(chunks)
    by_name = {d["doc_name"].casefold(): d for d in docs}
    ordered = rank([{"content": "FY 2017 fourth quarter"}, {"content": "FY 2018 first quarter"}], [2018], by_name, [1])
    require("first quarter" in ordered[0]["content"], "财年或季度排序错误")
    return {"parser": parser, "engine": engine, "embedding": state["embedding_model"], "counts": counts}


def cutoffs(docs, names, before, after=None):
    question = "3M FY2018 and FY2022 consolidated sales"
    targets = {d["id"] for d in docs if d["doc_name"] in names}
    require(len(targets) == len(names), "日期验收报告未入库")
    left = scope(question, docs, before)["document_ids"]
    if after:
        right = scope(question, docs, after)["document_ids"]
        require(targets.isdisjoint(left) and targets.issubset(right), "同一原报告未按披露日期跨界")
    else:
        require(targets.isdisjoint(left), "早于披露日时已检索到报告")
    return {"before_or_as_of": before, "after": after}


def unknown_scope(docs):
    plan = scope("What was the consolidated net income in FY2018?", docs)
    require(not plan["companies"] and plan["document_ids"] is None, "含糊公司应先追问")
    unknown = {"id": "unknown-date", "company": "3M", "doc_name": "unknown"}
    require("unknown-date" not in scope("3M FY2018 net sales", [*docs, unknown], "2019-12-31")["document_ids"],
            "日期未知报告出现在带截止日的查询")
    return {}


def calculation(parent):
    from .financial_agent import Operand, calculate_financial_metric, evidence_lines, verify_operand

    source = next(c for c in read_jsonl(parent) if c["id"] == "ddf0478121f5a2df")
    source = {**source, "document_id": source.get("document_id") or source.get("doc_id")}
    text = source["content"]
    require(all(v in text for v in ("32,765", "31,657", "2018", "2017")), "冻结表格的期间或操作数变化")
    lines = [{"number": i, "text": value} for i, value in enumerate(evidence_lines(text), 1)]
    line = next(x["number"] for x in lines if "32,765" in x["text"] and "31,657" in x["text"])
    # 原表使用同一张金额表内的美元符号，逐列核验时保留表内币种口径。
    operands = [Operand(chunk_id=source["id"], line_number=line, year=year, period="FY",
                        number=value, currency="XXX", scale="millions", kind="money", row_label="Netsales")
                for year, value in ((2018, "32,765"), (2017, "31,657"))]
    for operand in operands:
        verify_operand(operand, source, lines)
    result = calculate_financial_metric("growth", *operands)
    expected = ((Decimal(32765) / Decimal(31657) - 1) * 100).quantize(Decimal(".0001"))
    require(Decimal(result.split("=")[-1].strip().removesuffix("%")) == expected, "Decimal 结果与原文操作数不符")
    return {"source_sha256": digest(text), "growth_percent": str(expected), "currency_in_table": "$"}


def trace_check(path):
    row, = read_jsonl(path)
    layers = row["native_trace"]["evidence_layers"]
    answer, = layers["answers"]
    request, = [r for r in answer["requests"] if r["status"] == 200]
    intact = sum(c.get("source_check") == "intact" for c in request["complete_chunks"] or [])
    require(layers["retrievals"] and answer["selected"] and answer["block_hashes"] and intact,
            "真实请求缺少检索、候选、最终消息或索引回读")
    return {"result_sha256": digest(row), "retrievals": len(layers["retrievals"]), "intact": intact}


def run_tests(names):
    stream = io.StringIO()
    outcome = unittest.TextTestRunner(stream=stream).run(unittest.defaultTestLoader.loadTestsFromNames(names))
    require(outcome.testsRun and outcome.wasSuccessful(), stream.getvalue()[-3000:])
    return {"tests": list(names), "assertion_cases": outcome.testsRun}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent-table", required=True, type=Path)
    p.add_argument("--trace", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    import sys
    sys.path.insert(0, str(Path("tests").resolve()))
    state = read_json("data/ingestion.json")
    docs = catalog(read_jsonl("data/financebench/documents.jsonl"), state)
    checks = {
        "F01": lambda: pdf_and_sources(state),
        "F02": lambda: retrieval(state, docs),
        "F03": lambda: cutoffs(docs, {"3M_2018_10K"}, "2019-02-06"),
        "F04": lambda: cutoffs(docs, {"3M_2018_10K"}, "2019-02-06", "2019-02-07"),
        "F05": lambda: cutoffs(docs, {"3M_2022_10K"}, "2023-02-07", "2023-02-08"),
        "F06": lambda: unknown_scope(docs),
        "F12": lambda: calculation(args.parent_table),
        "F18": lambda: trace_check(args.trace),
        **{code: lambda names=names: run_tests(names) for code, names in TESTS.items()},
    }
    report = {}
    for code, run in sorted(checks.items()):
        try:
            report[code] = {"passed": True, **run()}
        except Exception as error:
            report[code] = {"passed": False, "type": type(error).__name__, "reason": str(error)[:3000]}
        print(code, "通过" if report[code]["passed"] else f"失败 {report[code]['type']}", flush=True)
    write_json(args.output, report)
    if not all(v["passed"] for v in report.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
