"""探测模型接口接受的思考强度，不把接口接受等同于内部行为保证。"""

import argparse
import json
import os
import time
import urllib.error
import urllib.request

from .common import write_json


def main():
    parser = argparse.ArgumentParser(description="对指定 OpenAI 兼容入口逐一探测强度，每档最多一次请求")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--model", default="gpt-6.1-sol")
    parser.add_argument("--efforts", nargs="+", default=["low", "medium", "high", "xhigh", "max", "ultra"])
    parser.add_argument("--output", default="runs/model_probe.json")
    args = parser.parse_args()
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError("缺少 OPENAI_API_KEY；若使用 Codex 桥接，按桥接服务的认证方式配置，不能假设 ChatGPT 登录就是 API key")
    rows = []
    for effort in args.efforts:
        payload = {"model": args.model, "messages": [{"role": "user", "content": "只回复数字：7加8等于多少？"}],
                   "reasoning_effort": effort, "max_completion_tokens": 256}
        request = urllib.request.Request(args.base_url.rstrip("/") + "/chat/completions", data=json.dumps(payload).encode(),
                                         headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
        row = {"effort": effort, "accepted": False, "effective_behavior_verified": False}
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                value = json.load(response)
            row.update({"accepted": True, "response_model": value.get("model"), "usage": value.get("usage"),
                        "finish_reason": value.get("choices", [{}])[0].get("finish_reason")})
        except urllib.error.HTTPError as exc:
            # 不输出服务返回的任意请求详情或鉴权头。
            row["http_status"] = exc.code
        except Exception as exc:
            row["error_type"] = type(exc).__name__
        row["seconds"] = time.perf_counter() - start
        rows.append(row)
        write_json(args.output, {"model": args.model, "base_url": args.base_url, "results": rows,
                                 "说明": "accepted 仅表示接口接受；还须确认 RAGFlow 到模型的参数传递。探测不用于质量排名。"})
        print(f"{effort}：{'接口接受' if row['accepted'] else '未接受或请求失败'}，{row['seconds']:.1f} 秒", flush=True)


if __name__ == "__main__":
    main()
