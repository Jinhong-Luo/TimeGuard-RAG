"""用相同原生草稿提示生成两版笔记，再用相同的补表材料作答。"""
import argparse
import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path

from replay_two_table_cases import CASES, api, evidence_tokens, judge, save, selected_rows, table_from_page


# 固定版 _compose_fallback_draft 对英文题使用的系统提示。
PROMPT = (
    "You are a research assistant writing an INTERMEDIATE DRAFT toward answering the user's "
    "question, using ONLY the retrieved evidence snippets below.\n"
    "Requirements:\n"
    "1. First list concrete FACTS FOUND in the snippets (exact numbers, dates, names preserved).\n"
    "2. Then output a line starting with 'MISSING:' naming precisely which part(s) of the "
    "question the snippets do NOT answer yet.\n"
    "3. No conclusions beyond the evidence; no general knowledge.\n"
    "Keep it under 250 words."
)


def completion(folder, label, body, key):
    path = folder / (label + ".json")
    signature = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    marker = folder / (label + "-started.json")
    if not path.exists():
        save(marker, signature)
        save(path, api.post_completion(None, body, key, 180))
    if json.loads(marker.read_text()) != signature:
        raise ValueError("输入与已有记录不一致")
    return json.loads(path.read_text())["choices"][0]["message"]["content"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    key = api.api_key()
    for code, row in selected_rows().items():
        number, page, caption = CASES[code]
        layer = row["native_trace"]["evidence_layers"]["answers"][0]
        req, = layer["requests"]
        base = req["body"]
        if req["status"] != 200 or base["model"] != "gpt-6.1-sol@low":
            raise ValueError("冻结请求不符")
        anchor = layer["packet"][number - 1]
        table = table_from_page(anchor, page, caption)
        prefix, original = base["messages"][-1]["content"].split("\nEvidence:\n")
        prefix, _ = prefix.split("Research Summary (primary evidence):\n")
        question = prefix.split("Question:\n")[1].split("\n\n")[0]
        start = original.index(f"\nID: {number}\n")
        end = original.find(f"\n\nID: {number + 1}\n", start)
        end = len(original) if end < 0 else end
        clear = original[:end] + "\n" + table + original[end:]
        counts = {s: evidence_tokens(s) for s in (original, clear)}
        for mode in ("原片段", "提前补表"):
            folder = args.output / (code + "-" + mode)
            folder.mkdir(parents=True, exist_ok=True)
            snippets = original if mode == "原片段" else clear
            research = deepcopy(base)
            research["messages"] = [{"role": "system", "content": PROMPT}, {"role": "user", "content":
                f"Question: {question}\n\nRetrieved evidence:\n{snippets}"}]
            note = completion(folder, "note", research, key)
            final = deepcopy(base)
            final["messages"][-1]["content"] = prefix + "Research Summary (primary evidence):\n" + note + "\n\nEvidence:\n" + clear
            answer = completion(folder, "answer", final, key)
            result = folder / "judge.json"
            if not result.exists():
                generated = deepcopy(row)
                generated["answer"] = answer
                for c in generated["native_trace"]["evidence_layers"]["answers"][0]["requests"][0]["complete_chunks"]:
                    if c["id"] == anchor["id"]:
                        c["source_check"] = "edited"
                gold = next(g for g in map(json.loads, Path("data/financebench/gold.jsonl").read_text().splitlines()) if g["id"] == row["id"])
                save(folder / "judge-started.json", True)
                save(result, judge.judge_one(gold, generated, "early-table-notes", None, key, 180, api.post_completion, api.TRANSPORT))
            j = json.loads(result.read_text())
            if j["judge_status"] != "ok":
                raise RuntimeError(j.get("error") or j["judge_status"])
            print(code, mode, counts[snippets], counts[clear], hashlib.sha256(clear.encode()).hexdigest()[:16],
                  j["answer_correct"], j["citation_support"], flush=True)
            print(note, j["judgement_reason"], sep="\n", flush=True)


if __name__ == "__main__":
    main()
