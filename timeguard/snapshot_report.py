"""统计最终节点各层的标准页覆盖；只在生成结束后读取金标。"""
import argparse
from pathlib import Path
from collections import Counter
from .common import read_json, read_jsonl, write_json
from .layer_report import layers, page_set


def report(run, gold_path):
    labels = {v["id"]: v for v in read_jsonl(gold_path)}
    reviews = {v["id"]: v for v in read_jsonl(run / "model_reviews.jsonl")}
    counts = Counter()
    coverage = Counter()
    details = []
    for row in read_jsonl(run / "results.jsonl"):
        stages = layers(row)
        answers = row["native_trace"]["evidence_layers"]["answers"]
        answer = answers[0] if len(answers) == 1 else {}
        stages["入口"] = page_set(answer.get("pool_locations"))
        targets = {(e["doc_name"].casefold(), e["evidence_page_num"] + 1) for e in labels[row["id"]]["evidence"]}
        known = {k: (p & targets if valid else None) for k, (p,valid) in stages.items()}
        counts["题数"] += 1
        counts["标准题页组合数"] += len(targets)
        for k, hits in known.items():
            if hits is not None:
                coverage[k] += len(hits)
                counts[k + "可核验题数"] += 1
        losses = {}
        for left, right in (("candidates", "入口"), ("入口", "selected"), ("selected", "packet"), ("packet", "received")):
            a, b = known[left], known[right]
            losses[left + "→" + right] = sorted(a-b) if a is not None and b is not None else None
        checks = answer.get("replay", {})
        for key, value in checks.items():
            counts[key] += int(value is True)
        review = reviews.get(row["id"], {})
        counts[{True: "判对", False: "判错", None: "未知"}[review.get("answer_correct")]] += 1
        details.append({"题号": row["id"], "各层丢失页": losses, "回放": checks,
                        "生成状态": row["status"], "耗时秒": row["elapsed_seconds"],
                        "判对": review.get("answer_correct"), "引用支持": review.get("citation_support")})
    costs = {label: {k:v for k,v in read_json(run / file).items()
                     if k in ("actual_calls", "failed_calls", "tokens", "wall_seconds")}
             for label,file in (("生成", "execution.json"), ("裁判", "judge-audit.json"))}
    return {"类别": "调优小样本真实生成、原版回放与模型裁判", "统计": dict(counts),
            "标准页命中": dict(coverage), "开销": costs}, details


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("run","gold","output","private_details"):
        p.add_argument(key,type=Path)
    args = p.parse_args()
    summary, details = report(args.run,args.gold)
    write_json(args.private_details, details)
    write_json(args.output, summary)
