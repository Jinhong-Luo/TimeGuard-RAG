"""仅替换独立原生评测进程内指向专用模型入口的 HTTP 调用。"""

import asyncio
from functools import wraps
import json
import time

from .common import digest
from . import openai_transport as api


def sse(completion):
    choice, = completion["choices"]
    delta = choice["message"].copy()
    if delta.get("tool_calls"):
        delta["tool_calls"] = [{"index": i, **call} for i, call in enumerate(delta["tool_calls"])]
    base = {key: completion[key] for key in ("id", "model", "created")}
    base["object"] = "chat.completion.chunk"
    frames = [
        {**base, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
        {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": choice["finish_reason"]}],
         "usage": completion["usage"]},
    ]
    return "".join("data: " + json.dumps(frame, ensure_ascii=False) + "\n\n" for frame in frames) + "data: [DONE]\n\n"


def install(endpoint, current):
    import httpx

    key = api.api_key()
    target = httpx.URL(endpoint.rstrip("/") + "/chat/completions")
    original = httpx.AsyncClient.send

    @wraps(original)
    async def send(client, request, *args, **kwargs):
        state = current.get()
        if state is None or request.method != "POST" or request.url != target:
            return await original(client, request, *args, **kwargs)

        started = time.monotonic()
        audit = {"kind": "openai_request", "transport": api.TRANSPORT, "success": False,
                 "http_status": None, "attempted": False}
        state["events"].append(audit)
        try:
            body = json.loads(request.content)
            history = state.setdefault("_openai_tool_turns", {})
            payload = api.request_payload(body, history)
            audit.update(model=payload["model"], requested_effort=payload["reasoning"]["effort"],
                         request_sha256=digest(payload), attempted=True,
                         tool_results=sum(item.get("type") == "function_call_output" for item in payload["input"]))
            outbound = httpx.Request("POST", api.URL, json=payload, extensions=dict(request.extensions),
                                     headers={"Authorization": "Bearer " + key})
            response = await original(client, outbound, follow_redirects=False, auth=None)
            request_id = response.headers.get("x-request-id")
            audit.update(http_status=response.status_code, openai_request_id=request_id)
            if response.status_code != 200:
                raise api.APIError(response.status_code, api.error_code(response.content))
            result = response.json()
            audit.update(raw_usage=result.get("usage"), openai_response_id=result.get("id"))
            completion = api.completion(body, payload, result, request_id, history)
            audit.update(success=True, usage=completion["usage"], **completion["bridge_metadata"],
                         tool_calls=len(completion["choices"][0]["message"].get("tool_calls", ())))
            if body.get("stream"):
                return httpx.Response(200, text=sse(completion), headers={"Content-Type": "text/event-stream"},
                                      request=request)
            return httpx.Response(200, json=completion, request=request)
        except (httpx.HTTPError, api.APIError, KeyError, TypeError, ValueError) as error:
            audit["error_type"] = type(error).__name__
            if isinstance(error, api.APIError):
                audit["error_code"] = error.code
                message, status = str(error), error.status
            else:
                message, status = "官方模型请求或响应校验失败", 502
            return httpx.Response(status, json={"error": {"type": type(error).__name__, "message": message}},
                                  request=request)
        except asyncio.CancelledError:
            audit["error_type"] = "CancelledError"
            raise
        finally:
            audit["elapsed_seconds"] = time.monotonic() - started

    httpx.AsyncClient.send = send
