"""验证桥接的普通回复、业务工具和裁判输出；不运行真实业务工具。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request

from codex_bridge import EFFORTS, MODEL, TRANSPORT


def cases() -> dict:
    return {
        "text": {
            "messages": [{"role": "user", "content": "计算 17×19，仅回答整数，不使用工具。"}],
        },
        "tool": {
            "messages": [
                {"role": "system", "content": "你是零售客服。这里只验证工具调用格式，不执行任何业务写入。请按要求调用一次 lookup_order。"},
                {"role": "user", "content": "请查询订单 TEST-ORDER-001，订单属于用户 TEST-USER-001。"},
            ],
            "tools": [{"type": "function", "function": {
                "name": "lookup_order", "description": "查询虚拟测试订单信息，不产生实际业务操作。",
                "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "user_id": {"type": "string"}}, "required": ["order_id", "user_id"], "additionalProperties": False},
            }}],
            "tool_choice": {"type": "function", "function": {"name": "lookup_order"}},
            "parallel_tool_calls": False,
        },
        "judge": {
            "messages": [{"role": "user", "content": "判断候选答案是否与参考答案一致。参考答案：323；候选答案：323。仅返回 JSON 对象，包含 correct 布尔字段和 reason 字符串字段。"}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "probe_judgement", "strict": True, "schema": {"type": "object", "properties": {"correct": {"type": "boolean"}, "reason": {"type": "string"}}, "required": ["correct", "reason"], "additionalProperties": False}}},
        },
    }


def check_response(case: str, response: dict, effort: str) -> dict:
    metadata = response.get("bridge_metadata", {})
    if metadata.get("transport") != TRANSPORT:
        raise ValueError("缺少预期桥接传输标识")
    if metadata.get("reasoning_effort") != effort:
        raise ValueError("返回思考强度与请求不一致")
    if metadata.get("api_equivalent") is not False:
        raise ValueError("响应缺少非 API 等价标记")
    message = response["choices"][0]["message"]
    if case == "text" and str(message.get("content", "")).strip() != "323":
        raise ValueError("普通回答算术探测不正确")
    if case == "tool":
        calls = message.get("tool_calls") or []
        if len(calls) != 1 or calls[0]["function"]["name"] != "lookup_order":
            raise ValueError("未返回预期的单个业务工具")
        arguments = json.loads(calls[0]["function"]["arguments"])
        if arguments != {"order_id": "TEST-ORDER-001", "user_id": "TEST-USER-001"}:
            raise ValueError("业务工具参数与输入不一致")
    if case == "judge":
        judgement = json.loads(message.get("content", ""))
        if judgement.get("correct") is not True or not isinstance(judgement.get("reason"), str):
            raise ValueError("裁判结构化输出不正确")
    return message


def classify_failure(error: str) -> str:
    if "401" in error or "认证失败" in error or "unauthorized" in error.lower():
        return "authentication_failure"
    if "workspace routing discovery" in error.lower():
        return "routing_discovery_failure"
    if "timed out" in error.lower() or "超时" in error:
        return "transport_timeout"
    if "429" in error or "rate limit" in error.lower():
        return "rate_limit"
    return "probe_failure"


def main():
    parser = argparse.ArgumentParser(description="检查 Codex 桥接，不执行真实业务工具")
    parser.add_argument("--base-url", default="http://127.0.0.1:18089/v1")
    parser.add_argument("--effort", nargs="+", choices=EFFORTS, default=["low"])
    parser.add_argument("--all-efforts", action="store_true")
    parser.add_argument("--cases", nargs="+", choices=("text", "tool", "judge"), default=["text", "tool", "judge"])
    parser.add_argument("--timeout", type=float, default=360)
    parser.add_argument("--output", type=Path, default=Path("reports/codex_bridge_probe.json"))
    args = parser.parse_args()
    efforts = list(EFFORTS) if args.all_efforts else list(dict.fromkeys(args.effort))
    result = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "purpose": "仅验证传输和输出格式，不能代表业务基线或用于选择思考档位", "model": MODEL, "transport": TRANSPORT, "results": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("CODEX_BRIDGE_TOKEN", "local-bridge")
    headers = {"Content-Type": "application/json", "Authorization": "Bearer " + token}
    # 本机或 Docker 网关请求不应受上游模型出口代理环境影响。
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    all_cases = cases()
    for effort in efforts:
        for case in args.cases:
            payload = {"model": MODEL, "reasoning_effort": effort, **all_cases[case]}
            begin = time.monotonic()
            record = {"effort": effort, "case": case, "success": False}
            try:
                request = urllib.request.Request(args.base_url.rstrip("/") + "/chat/completions", data=json.dumps(payload, ensure_ascii=False).encode("utf-8"), headers=headers)
                with opener.open(request, timeout=args.timeout) as response:
                    body = json.load(response)
                record.update({"message": check_response(case, body, effort), "bridge_metadata": body.get("bridge_metadata"), "usage": body.get("usage"), "response_model": body.get("model"), "success": True})
            except (urllib.error.URLError, ValueError, KeyError, TimeoutError, OSError) as error:
                message = str(error)
                if isinstance(error, urllib.error.HTTPError):
                    message += " " + error.read(8192).decode("utf-8", errors="replace")
                record.update({"error": message, "failure_category": classify_failure(message)})
            record["wall_seconds"] = round(time.monotonic() - begin, 3)
            result["results"].append(record)
            result["complete"] = len(result["results"]) == len(efforts) * len(args.cases)
            result["all_passed"] = result["complete"] and all(item["success"] for item in result["results"])
            args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({key: record[key] for key in ("effort", "case", "success", "wall_seconds")}, ensure_ascii=False), flush=True)
    raise SystemExit(0 if result["all_passed"] else 1)


if __name__ == "__main__":
    main()
