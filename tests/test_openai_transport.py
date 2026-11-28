import json
from contextvars import ContextVar
import os
import unittest
from unittest.mock import patch

from timeguard.evals import judge
from timeguard.llm import openai_transport as api


def response(effort, output, status="completed"):
    return {"id": "resp_test", "created_at": 1, "model": api.MODEL, "status": status,
            "reasoning": {"effort": effort}, "output": output,
            "usage": {"input_tokens": 4, "output_tokens": 3, "total_tokens": 7}}


class OpenAITransportTest(unittest.TestCase):
    def test_ragflow_generated_navigation_does_not_require_api_history(self):
        for ident in ("nav_scoped", "ladder_rank"):
            body = {"model": api.MODEL, "messages": [
                {"role": "assistant", "content": "", "tool_calls": [
                    {"id": ident, "type": "function", "function": {"name": "retrieve", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": ident, "content": [{"type": "text", "text": "来源原文"}]},
            ]}
            self.assertEqual(api.request_payload(body)["input"], [
                {"type": "function_call", "call_id": "call_tg_" + ident, "name": "retrieve", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "call_tg_" + ident, "output": "来源原文"},
            ])

    def test_tool_round_trip_stays_in_the_question(self):
        body = {"model": api.MODEL + "@low", "messages": [{"role": "user", "content": "查询一份财报"}],
                "tools": [{"type": "function", "function": {"name": "search", "parameters": {"type": "object"}}}]}
        payload = api.request_payload(body)
        self.assertEqual(payload["reasoning"], {"effort": "low"})
        self.assertIs(payload["store"], False)
        self.assertEqual(payload["tools"][0]["name"], "search")
        self.assertIs(payload["tools"][0]["strict"], False)
        returned = [
            {"type": "reasoning", "id": "rs_test", "summary": [], "encrypted_content": "opaque"},
            {"type": "function_call", "id": "fc_test", "call_id": "call_test", "name": "search", "arguments": "{}"},
        ]
        history = {}
        result = api.completion(body, payload, response("low", returned), "req_test", history)
        body["messages"].extend([result["choices"][0]["message"],
                                 {"role": "tool", "tool_call_id": "call_test", "content": "财报原文"}])
        with self.assertRaisesRegex(ValueError, "本题工具调用"):
            api.request_payload(body)
        self.assertEqual(api.request_payload(body, history)["input"][1:], [
            *returned, {"type": "function_call_output", "call_id": "call_test", "output": "财报原文"},
        ])

    def test_medium_strict_judge_requires_official_proof(self):
        body = {"model": api.MODEL, "reasoning_effort": "medium", "messages": [],
                "response_format": {"type": "json_schema", "json_schema": {
                    "strict": True, "name": "finance", "schema": judge.SCHEMA}}}
        payload = api.request_payload(body)
        self.assertIs(payload["text"]["format"]["strict"], True)
        decision = {**dict.fromkeys(judge.FIELDS, True), "has_factual_claims": True,
                    "judgement_reason": "实收原文和报告数值一致。", "failure_category": "未发现错误"}
        text = {"type": "output_text", "text": json.dumps(decision)}
        value = api.completion(body, payload, response("medium", [{"type": "message", "content": [text]}]), "req_test")
        self.assertEqual(judge.parse_judgement(value, api.TRANSPORT)[0], decision)
        with self.assertRaisesRegex(ValueError, "固定传输"):
            judge.parse_judgement(value)
        value["bridge_metadata"].pop("openai_request_id")
        with self.assertRaisesRegex(ValueError, "调用凭证"):
            judge.parse_judgement(value, api.TRANSPORT)

    def test_incomplete_and_wrong_effort_fail_closed(self):
        body = {"model": api.MODEL, "messages": []}
        payload = api.request_payload(body)
        with self.assertRaises(api.APIError):
            api.completion(body, payload, response("low", [], status="incomplete"), "req_test")
        with self.assertRaisesRegex(ValueError, "推理档位"):
            api.completion(body, payload, response("medium", []), "req_test")


class NativeTransportTest(unittest.IsolatedAsyncioTestCase):
    async def test_only_the_selected_model_endpoint_uses_official_api(self):
        import httpx
        from timeguard.llm.openai_native import install

        current = ContextVar("题目", default=None)
        endpoint = "http://127.0.0.1:18866/v1"
        observed = []

        async def send(client, request, *args, **kwargs):
            observed.append(request)
            if str(request.url) != api.URL:
                return httpx.Response(202, request=request)
            output = [{"type": "message", "content": [{"type": "output_text", "text": "结果"}]}]
            return httpx.Response(200, json=response("low", output), headers={"x-request-id": "req_test"},
                                  request=request)

        with patch.object(httpx.AsyncClient, "send", send), patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            install(endpoint, current)
            async with httpx.AsyncClient() as client:
                self.assertEqual((await client.get(endpoint + "/models")).status_code, 202)
                state = {"events": []}
                token = current.set(state)
                try:
                    body = {"model": api.MODEL, "stream": True, "messages": [{"role": "user", "content": "问题"}]}
                    result = await client.post(endpoint + "/chat/completions", json=body,
                                               headers={"Authorization": "Bearer codex-placeholder"})
                finally:
                    current.reset(token)
        self.assertIn("data: [DONE]", result.text)
        self.assertIn("结果", result.text)
        self.assertEqual(observed[1].headers["Authorization"], "Bearer test-key")
        event, = state["events"]
        self.assertTrue(event["success"])
        self.assertEqual(event["openai_request_id"], "req_test")
        self.assertNotIn("test-key", json.dumps(event))
