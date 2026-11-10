"""在已准备好公司元数据的独立环境中，逐项验证指定的组合链路。"""
import argparse
from pathlib import Path
from .common import read_json, read_jsonl, write_json
from .native_baseline import run_arm
from .native_matrix import judge


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reranker", required=True)
    args = parser.parse_args()
    cases = [
        ("evidence", "tune", ["03029", "04672"]),
        ("fallback", "dev", ["05915", "00790"]),
        ("concise", "dev", ["00684"]),
        ("temporal", "dev", ["00288"]),
        ("adaptive", "tune", ["03029"]),
        ("finance", "dev", ["00799"]),
    ]
    for candidate, split, numbers in cases:
        ids = ["financebench_id_" + n for n in numbers]
        run_arm(args.output, "combined", split, None, args.reranker, "regression", candidate, ids)
        path = args.output / f"combined-{candidate}-regression-{split}"
        if judge(path):
            raise RuntimeError("指定题号的裁判未完成")
        reviews = read_jsonl(path/"model_reviews.jsonl")
        answers = read_jsonl(path/"results.jsonl")
        write_json(path/"regression.json", {"variant":candidate, "planned":len(ids),
            "correct":sum(r["answer_correct"] is True for r in reviews),
            "unknown":sum(r["answer_correct"] is None for r in reviews),
            "calls":read_json(path/"execution.json")["actual_calls"],
            "paths":[[e["kind"] for e in r["native_trace"]["events"] if e["kind"] in
                ("fallback","financial_route","financial_answer","financial_clarify","evidence_packet")] for r in answers]})


if __name__ == "__main__":
    main()
