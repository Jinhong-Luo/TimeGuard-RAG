"""仅在相同开发集和完整复核结果上选择思考强度。"""

import argparse
from pathlib import Path

from .common import read_json, write_json


def main():
    parser = argparse.ArgumentParser(description="质量相差不超过 2 个百分点时优先选择更快的强度")
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--tolerance", type=float, default=0.02)
    parser.add_argument("--output", default="reports/effort_selection.json")
    args = parser.parse_args()
    candidates, shared = [], None
    for run in args.runs:
        root = Path(run)
        manifest = read_json(root / "manifest.json")["identity"]
        summary = read_json(root / "summary.json")
        if manifest["split"] != "dev" or manifest["retrieval_only"]:
            raise ValueError("只能使用开发集完整问答运行选强度，不能使用保留测试集")
        if summary["completed"] != summary["denominator"] or summary["answer_accuracy"] is None:
            raise ValueError("每档运行和人工答案复核必须全部完成")
        current = {k: v for k, v in manifest.items() if k not in ("effort", "transport_proof")}
        if shared is not None and current != shared:
            raise ValueError("候选实验的问题、模型、Embedding、文档或检索配置不一致")
        shared = current
        candidates.append({"run": run, "effort": manifest["effort"], **summary})
    best_quality = max(c["answer_accuracy"] for c in candidates)
    eligible = [c for c in candidates if c["answer_accuracy"] >= best_quality - args.tolerance]
    chosen = min(eligible, key=lambda x: x["mean_seconds"])
    write_json(args.output, {"chosen_effort": chosen["effort"], "rule": "开发集人工正确率距最优不超过设定容差时选择平均耗时最短者",
                             "quality_tolerance": args.tolerance, "candidates": candidates,
                             "说明": "开发集选择为暂定结论；冻结配置后在保留测试集报告一次最终效果，费用未知时不能宣称最便宜。"})
    print(f"开发集暂选强度：{chosen['effort']}，结果已保存至 {args.output}")


if __name__ == "__main__":
    main()
