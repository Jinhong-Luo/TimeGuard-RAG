"""通过回环 HTTP 服务实测追问、历史截止、请求重放及财报计算。"""
import argparse
import json
from decimal import Decimal
from pathlib import Path
import urllib.request
from uuid import uuid4
from .common import digest, read_jsonl, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", default="http://127.0.0.1:19102")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--calculate", action="store_true")
    p.add_argument("--parent-tables", type=Path, default=Path("runs/r1-table-3m/parents.jsonl"))
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def post(payload):
        req = urllib.request.Request(args.base_url+"/turn", json.dumps(payload).encode(), headers={"Content-Type":"application/json"})
        with opener.open(req, timeout=150) as response:
            return json.load(response)
    session = str(uuid4())
    initial = {"session_id": session, "request_id": str(uuid4()), "question": "计算净销售额增长率。"}
    clarify = post(initial)
    if clarify["status"] != "clarify" or post(initial) != clarify:
        raise ValueError("缺参追问或请求重放与预期不同")
    historical = post({"session_id": str(uuid4()), "request_id": str(uuid4()),
        "question": "3M FY2018 的营收增长率是多少？", "as_of": "2019-02-06"})
    if historical["status"] != "clarify" or historical["facts"]:
        raise ValueError("早于首个披露日期时出现未经支持的回答")
    results = [clarify, historical]
    check = None
    if args.calculate:
        next_turn = {"session_id": session, "request_id": str(uuid4()), "revision": clarify["revision"],
            "question": "For 3M compute the growth in consolidated net sales from FY2017 to FY2018. Use the selected financial data table.", "as_of": "2019-02-07"}
        actual = post(next_turn)
        if actual["revision"] != 2 or post(next_turn) != actual:
            raise ValueError("真实计算请求的续接或重放失败")
        results.append(actual)
        # 生成完毕后才读取独立留存的原表核验这一道业务样例。
        source = next(c for c in read_jsonl(args.parent_tables) if c["id"] == "ddf0478121f5a2df")
        if not all(text in source["content"] for text in ("Netsales", "32,765", "31,657", "2018", "2017")):
            raise ValueError("业务样例的核验来源已改变")
        expected = ((Decimal(32765) / Decimal(31657) - 1) * 100).quantize(Decimal(".0001"))
        check = {"reference_chunk": source["id"], "reference_content_sha256": digest(source["content"]),
                 "expected_rounded_percent": str(expected), "answer_correct": False}
        if actual["status"] == "answered" and len(actual["facts"]) == 2:
            pct = Decimal(actual["answer"].split("=")[-1].strip().removesuffix("%"))
            check["answer_correct"] = pct == expected and [
                (f["year"], f["number"].replace(",", "")) for f in actual["facts"]] == [(2018, "32765"), (2017, "31657")]
    write_json(args.output/"responses.json", results)
    write_json(args.output/"summary.json", {"responses_sha256": digest(results), "states": [r["status"] for r in results],
        "answers": [r["answer"] for r in results], "fact_counts": [len(r["facts"]) for r in results],
        "usage": [r.get("usage") for r in results], "endpoint": args.base_url, "same_request_replay": True,
        "business_check": check})


if __name__ == "__main__":
    main()

