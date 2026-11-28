"""两道错题的终答重放。在含正式冻结记录的项目根目录执行。"""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

from timeguard import judge, openai_transport as api

RUN = Path("runs/financebench-openai-20261125-v2")
# 从旧选块定位表格。只登记块号、物理页、表题及边界，不指定任何答案行或数值。
CASES = {
    "01936": (1, 15, "An analysis of the Company's restructuring plan liability is as follows:"),
    "02024": (2, 94, "The benefit payments to retirees are expected to be paid as follows:"),
}


def selected_rows():
    rows = {}
    for path in sorted(RUN.glob("generation/*/results.jsonl")):
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                for code in CASES:
                    if row["position"] == f"FB-{code}-A1":
                        if code in rows:
                            raise ValueError("发现重复位置")
                        rows[code] = row
    if rows.keys() != CASES.keys():
        raise ValueError("原始位置不完整")
    return rows


def table_from_page(chunk, page, caption):
    if page not in [p[0] for p in chunk["positions"]]:
        raise ValueError("表格页超出原命中范围")
    doc = Path("data/financebench/pdfs") / Path(chunk["document_name"]).name
    raw = subprocess.check_output(["pdftotext", "-layout", "-f", str(page),
                                   "-l", str(page), str(doc), "-"]).decode()
    if raw.count(caption) != 1:
        raise ValueError("表题不能唯一定位")
    after = raw.split(caption)[1].lstrip("\n")
    table = caption + "\n" + re.split(r"\n[ \t]*\n", after, maxsplit=1)[0]
    if table not in raw or len(table.splitlines()) < 4:
        raise ValueError("表格回读不完整")
    return table


def evidence_tokens(text):
    cmd = ["docker", "exec", "-i", "timeguard_ljh-ragflow-cpu-1", "python", "-c",
           "import sys; from common.token_utils import num_tokens_from_string as n; print(n(sys.stdin.read()))"]
    count = int(subprocess.check_output(cmd, input=text.encode()).decode().strip())
    if not 0 < count <= 8000:
        raise ValueError(f"证据预算不满足：{count}")
    return count


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    os.umask(0o077)
    args.output.mkdir(exist_ok=True, parents=True)
    rows = selected_rows()
    for code, (number, page, caption) in CASES.items():
        row = rows[code]
        layer = row["native_trace"]["evidence_layers"]["answers"][0]
        if len(layer["requests"]) != 1:
            raise ValueError("终答请求不唯一")
        req = layer["requests"][0]
        anchor = layer["packet"][number - 1]
        table = table_from_page(anchor, page, caption)
        for mode in ("补表", "原样") if code == "01936" else ("原样", "补表"):
            body = deepcopy(req["body"])
            user = body["messages"][-1]
            intro, sep, evid = user["content"].partition("\nEvidence:\n")
            if not sep or req["status"] != 200 or body["model"] != "gpt-6.1-sol@low":
                raise ValueError("输入与冻结实验不符")
            if mode == "补表":
                # 直接接在命中块内，原检索、选块、块编号、研究摘要和系统指令全部复用。
                begin = evid.index(f"\nID: {number}\n")
                end = evid.find(f"\n\nID: {number + 1}\n", begin)
                end = len(evid) if end < 0 else end
                evid = evid[:end] + "\n" + table + evid[end:]
                user["content"] = intro + sep + evid
            tokens = evidence_tokens(evid)
            print(code, mode, tokens, flush=True)
            if not args.live:
                continue
            folder = args.output / (code + "-" + mode)
            folder.mkdir(exist_ok=True)
            key = api.api_key()
            answer_path = folder / "answer.json"
            if not answer_path.exists():
                save(folder / "generation-started.json", {"tokens": tokens, "body_sha256": hashlib.sha256(
                    json.dumps(body, sort_keys=True).encode()).hexdigest()})
                answer = api.post_completion(None, body, key, 180)
                save(answer_path, answer)
            generated = deepcopy(row)
            generated["answer"] = json.loads(answer_path.read_text())["choices"][0]["message"]["content"]
            if mode == "补表":
                # PDF 回读不等于索引整块核验。新组合块在旧的严格引用规则下保守记为无法映射。
                for c in generated["native_trace"]["evidence_layers"]["answers"][0]["requests"][0]["complete_chunks"]:
                    if c["id"] == anchor["id"]:
                        c["source_check"] = "edited"
            judgement = folder / "judge.json"
            if not judgement.exists():
                gold = next(g for g in map(json.loads, Path("data/financebench/gold.jsonl").read_text().splitlines())
                            if g["id"] == row["id"])
                save(folder / "judge-started.json", {"position": row["position"]})
                j = judge.judge_one(gold, generated, "table-replay", None, key, 180,
                                    api.post_completion, api.TRANSPORT)
                save(judgement, j)
                if j["judge_status"] != "ok":
                    raise RuntimeError(j.get("error") or j["judge_status"])
            result = json.loads(judgement.read_text())
            print(code, mode, result["answer_correct"], result["citation_support"], result["judgement_reason"], flush=True)


if __name__ == "__main__":
    main()
