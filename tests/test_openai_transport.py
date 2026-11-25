import json
import unittest

from timeguard import judge, openai_transport as api


def response(effort, output, status="completed"):
    return {"id": "resp_test", "created_at": 1, "model": api.MODEL, "status": status,
            "reasoning": {"effort": effort}, "output": output,
            "usage": {"input_tokens": 4, "output_tokens": 3, "total_tokens": 7}}


class OpenAITransportTest(unittest.TestCase):
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
