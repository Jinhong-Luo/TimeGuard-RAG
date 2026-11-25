"""在保留原生研究工具的前提下，将模型请求送到官方 Responses API。"""

import json
import os
from pathlib import Path
import re
import stat
import urllib.error
import urllib.request


MODEL = "gpt-6.1-sol"
TRANSPORT = "openai-responses-v1"
URL = "https://api.openai.com/v1/responses"
EFFORTS = ("low", "medium", "high", "xhigh", "max")


def api_key():
    value = os.environ.get("OPENAI_API_KEY", "").strip()
    if not value:
        path = Path(os.environ.get("TIMEGUARD_OPENAI_KEY_FILE") or "~/.config/timeguard/openai-api-key").expanduser()
        if not path.is_file():
            raise ValueError("缺少 OPENAI_API_KEY 或 TimeGuard 私有密钥文件")
        if os.name == "posix" and stat.S_IMODE(path.stat().st_mode) & 0o077:
            raise ValueError("TimeGuard 密钥文件必须禁止组及其他用户读取")
        value = path.read_text(encoding="utf-8").strip()
    if not value or any(char.isspace() for char in value):
        raise ValueError("OpenAI 密钥为空或含空白字符")
    return value


def model_matches(value):
    return isinstance(value, str) and bool(re.fullmatch(r"gpt-6\.1-sol(?:-\d{4}-\d{2}-\d{2})?", value))


def native_navigation_id(value):
    # 固定版 RAGFlow 在模型首次研究前执行导航，使用这两个本地编号前缀。
    return isinstance(value, str) and value.startswith(("nav_", "ladder_"))


def text_content(content):
    if isinstance(content, list) and all(isinstance(part, dict) and part.get("type") == "text"
                                          and isinstance(part.get("text"), str) for part in content):
        return "".join(part["text"] for part in content)
    return content


def request_payload(body, tool_turns=None):
    model, separator, alias = body["model"].partition("@")
    effort = body.get("reasoning_effort") or alias or "low"
    if not model_matches(model) or effort not in EFFORTS or (separator and alias != effort):
        raise ValueError("官方模型名或推理档位与预设不符")
    tool_turns = tool_turns if tool_turns is not None else {}
    messages = []
    for message in body["messages"]:
        role, content = message["role"], text_content(message.get("content"))
        if role == "tool":
            if not isinstance(content, str):
                raise ValueError("本地检索工具结果必须是文本")
            ident = message["tool_call_id"]
            messages.append({"type": "function_call_output",
                             "call_id": "call_tg_" + ident if native_navigation_id(ident) else ident,
                             "output": content})
        elif role == "assistant" and message.get("tool_calls"):
            ids = tuple(call["id"] for call in message["tool_calls"])
            if ids in tool_turns:
                messages.extend(tool_turns[ids])
            elif all(native_navigation_id(ident) for ident in ids):
                if content:
                    messages.append({"role": "assistant", "content": content})
                messages.extend({"type": "function_call", "call_id": "call_tg_" + call["id"],
                                 "name": call["function"]["name"], "arguments": call["function"]["arguments"]}
                                for call in message["tool_calls"])
            else:
                raise ValueError("缺少本题工具调用的官方响应上下文")
        elif role in ("system", "developer", "user", "assistant") and isinstance(content, str):
            messages.append({"role": role, "content": content})
        else:
            raise ValueError("模型请求包含无法转换的消息类型")
    payload = {"model": model, "reasoning": {"effort": effort}, "input": messages, "store": False}
    if body.get("tools"):
        if any(tool.get("type") != "function" for tool in body["tools"]):
            raise ValueError("实验只允许原生本地函数")
        payload["tools"] = [{"type": "function", "strict": False, **tool["function"]} for tool in body["tools"]]
        payload["parallel_tool_calls"] = body.get("parallel_tool_calls", False)
    if "tool_choice" in body:
        choice = body["tool_choice"]
        payload["tool_choice"] = {"type": "function", "name": choice["function"]["name"]} if isinstance(choice, dict) else choice
    response_format = body.get("response_format")
    if response_format and response_format["type"] != "text":
        kind = response_format["type"]
        if kind not in ("json_schema", "json_object"):
            raise ValueError("模型响应格式不受支持")
        payload["text"] = {"format": {"type": kind, **response_format.get("json_schema", {})}}
    limit = body.get("max_completion_tokens") or body.get("max_tokens")
    if limit is not None:
        payload["max_output_tokens"] = limit
    return payload


def completion(body, payload, result, request_id, tool_turns=None):
    actual_model = result.get("model")
    effort = (result.get("reasoning") or {}).get("effort")
    if (not model_matches(actual_model) or effort != payload["reasoning"]["effort"]
            or not request_id or not result.get("id")):
        raise ValueError("官方响应缺少模型、推理档位或请求编号凭证")
    if result.get("status") != "completed":
        reason = (result.get("incomplete_details") or {}).get("reason")
        raise APIError(502, reason or result.get("status"))
    output = result["output"]
    texts, calls = [], []
    for item in output:
        if item["type"] == "message":
            for part in item["content"]:
                if part["type"] == "refusal":
                    raise APIError(502, "refusal")
                if part["type"] == "output_text":
                    texts.append(part["text"])
        elif item["type"] == "function_call":
            calls.append({"id": item["call_id"], "type": "function",
                          "function": {"name": item["name"], "arguments": item["arguments"]}})
    if not calls and not texts:
        raise ValueError("官方响应未包含答案或工具调用")
    message = {"role": "assistant", "content": "".join(texts) if texts else None}
    if calls:
        message["tool_calls"] = calls
        if tool_turns is not None:
            tool_turns[tuple(call["id"] for call in calls)] = output
    usage = result.get("usage")
    return {
        "id": result["id"], "object": "chat.completion", "created": int(result["created_at"]), "model": body["model"],
        "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if calls else "stop"}],
        "usage": {
            "prompt_tokens": usage["input_tokens"], "completion_tokens": usage["output_tokens"],
            "total_tokens": usage["total_tokens"], "prompt_tokens_details": usage.get("input_tokens_details", {}),
            "completion_tokens_details": usage.get("output_tokens_details", {}),
        } if usage else None,
        "bridge_metadata": {"transport": TRANSPORT, "actual_model": actual_model, "reasoning_effort": effort,
                            "openai_request_id": request_id, "openai_response_id": result["id"]},
    }


class APIError(RuntimeError):
    def __init__(self, status, code):
        self.status = status
        self.code = code if isinstance(code, str) and re.fullmatch(r"[a-z0-9_]{1,64}", code) else "unknown"
        super().__init__(f"OpenAI HTTP {status}（{self.code}）")


def error_code(value):
    try:
        return (json.loads(value).get("error") or {}).get("code")
    except (ValueError, AttributeError, TypeError):
        return None


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def post_completion(_base_url, body, key, timeout):
    payload = request_payload(body)
    request = urllib.request.Request(URL, data=json.dumps(payload, ensure_ascii=False).encode(),
                                     headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
    try:
        with urllib.request.build_opener(NoRedirect()).open(request, timeout=timeout) as response:
            return completion(body, payload, json.load(response), response.headers.get("x-request-id"))
    except urllib.error.HTTPError as error:
        raise APIError(error.code, error_code(error.read(4096))) from None
