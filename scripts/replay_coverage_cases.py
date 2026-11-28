"""在冻结的官方 API 请求上，逐题对照原提示和覆盖检查提示。"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path

from replay_table_notes import completion
from replay_two_table_cases import RUN, api, evidence_tokens, judge, save
from timeguard.common import digest


CODES = ("02024", "00882", "00540")
# 提示对三题相同，只使用题干与当次可见材料，不指定公司、财年或任何答案值。
CHECK = (
    "Before answering, check coverage from the question and the supplied evidence. "
    "Identify the requested entity, time, scope and quantity. Inspect the visible headings, "
    "rows and notes for parallel items covered by the question. Check whether each item "
    "has a dated figure, a unit and a supporting citation. Check effective dates if relevant. "
    "For a derived figure, state the definition; if more than one reasonable definition "
    "is supported and changes the result, report the alternatives briefly. "
    "Give the supported findings and explicitly identify any remaining gap."
)


def load_cases():
    expected = {f"FB-{code}-A1" for code in CODES}
    cases = {}
    for path in sorted(RUN.glob("generation/*/results.jsonl")):
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                position = row["position"]
                if position in expected:
                    if position in cases:
                        raise ValueError("重复的冻结位置")
                    cases[position] = row
    if cases.keys() != expected:
        raise ValueError("冻结位置不全")
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    os.umask(0o077)
    cases = load_cases()
    gold = {}
    if args.live:
        key = api.api_key()
        gold = {g["id"]: g for g in map(json.loads, Path("data/financebench/gold.jsonl").read_text().splitlines())}
    for number, code in enumerate(CODES):
        row = cases[f"FB-{code}-A1"]
        layer, = row["native_trace"]["evidence_layers"]["answers"]
        req, = layer["requests"]
        base = req["body"]
        if req["status"] != 200 or base["model"] != "gpt-6.1-sol@low" or base["messages"][0]["role"] != "system":
            raise ValueError("冻结请求不符")
        _, evidence = base["messages"][-1]["content"].split("\nEvidence:\n", 1)
        count = evidence_tokens(evidence)
        for mode in (("原样", "逐项") if number % 2 == 0 else ("逐项", "原样")):
            body = deepcopy(base)
            if mode == "逐项":
                body["messages"][0]["content"] += "\n" + CHECK
            print(code, mode, count, digest(evidence), flush=True)
            if not args.live:
                continue
            folder = args.output / f"{code}-{mode}"
            folder.mkdir(parents=True, exist_ok=True)
            generated = deepcopy(row)
            generated["answer"] = completion(folder, "answer", body, key)
            # 新答案沿用同一份提示块映射。仅在裁判输入中使用参考答案。
            generated["native_trace"]["citation_origin"] = "model"
            result = folder / "judge.json"
            signature = digest([body, generated["answer"], gold[row["id"]]])
            if not result.exists():
                save(folder / "judge-started.json", signature)
                save(result, judge.judge_one(gold[row["id"]], generated, signature, None,
                                             key, 180, api.post_completion, api.TRANSPORT))
            j = json.loads(result.read_text())
            if j["judge_status"] != "ok" or j["judge_run_signature"] != signature:
                raise RuntimeError("评分失败或签名与当前运行不一致")
            print(code, mode, j["answer_correct"], j["citation_support"], flush=True)


if __name__ == "__main__":
    main()
