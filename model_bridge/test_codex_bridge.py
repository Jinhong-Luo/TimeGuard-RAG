"""验证桥接边界；这些测试不连接 Codex 或执行模型。"""

import json
from pathlib import Path
import tempfile
import subprocess
import time
import unittest
from unittest.mock import patch

from codex_bridge import (
    BridgeError, CodexSession, DISABLED_FEATURES, MODEL,
    completion_message, parse_model_effort, validate_request,
    validate_bind_host,
)
from probe import check_response, classify_failure


class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.request = {
            "model": MODEL,
            "messages": [{"role": "user", "content": "查询订单"}],
            "tools": [{"type": "function", "function": {"name": "get_order", "parameters": {"type": "object"}}}],
        }

    def test_effort_alias_and_conflict(self):
        self.assertEqual(parse_model_effort({"model": MODEL + "@max"}, "low"), (MODEL, "max"))
        with self.assertRaises(BridgeError):
            parse_model_effort({"model": MODEL + "@low", "reasoning_effort": "high"}, "low")
        with self.assertRaises(BridgeError):
            parse_model_effort({"model": "gpt-6-sol"}, "low")
        with self.assertRaises(BridgeError):
            parse_model_effort({"model": MODEL, "reasoning_effort": "unknown"}, "low")

    def test_business_tool_is_only_converted_to_data(self):
        message = completion_message({"content": None, "tool_calls": [{"name": "get_order", "arguments": '{"order_id":"123"}'}]}, self.request)
        self.assertEqual(message["tool_calls"][0]["function"]["name"], "get_order")
        self.assertEqual(json.loads(message["tool_calls"][0]["function"]["arguments"]), {"order_id": "123"})

    def test_unknown_tool_rejected(self):
        with self.assertRaises(BridgeError):
            completion_message({"content": None, "tool_calls": [{"name": "exec_command", "arguments": "{}"}]}, self.request)

    def test_invalid_arguments_and_multiple_actions_rejected(self):
        with self.assertRaises(BridgeError):
            completion_message({"content": None, "tool_calls": [{"name": "get_order", "arguments": "[]"}]}, self.request)
        with self.assertRaises(BridgeError):
            completion_message({"content": None, "tool_calls": [{"name": "get_order", "arguments": "{}"}] * 2}, self.request)

    def test_tool_choice_respected(self):
        self.request["tool_choice"] = "none"
        with self.assertRaises(BridgeError):
            completion_message({"content": None, "tool_calls": [{"name": "get_order", "arguments": "{}"}]}, self.request)

    def test_judge_json_response(self):
        request = {"response_format": {"type": "json_object"}}
        answer = completion_message({"content": '{"correct":true}', "tool_calls": []}, request)
        self.assertTrue(json.loads(answer["content"])["correct"])
        with self.assertRaises(ValueError):
            completion_message({"content": "不是 JSON", "tool_calls": []}, request)

    def test_multimodal_and_parallel_rejected(self):
        self.request["messages"][0]["content"] = [{"type": "image_url", "image_url": {"url": "file:///answers"}}]
        with self.assertRaises(BridgeError):
            validate_request(self.request)
        self.request["messages"][0]["content"] = "普通问题"
        self.request["parallel_tool_calls"] = True
        with self.assertRaises(BridgeError):
            validate_request(self.request)

    def test_native_tool_request_fails_closed(self):
        session = CodexSession("codex", Path("."), 1)
        session.events.put({"method": "item/commandExecution/requestApproval", "id": 1, "params": {}})
        with patch.object(session, "send") as send:
            with self.assertRaises(BridgeError):
                session.receive(time.monotonic() + 1)
        self.assertIn("error", send.call_args.args[0])

    def test_start_isolates_environment_and_checks_effort(self):
        with patch("tempfile.TemporaryDirectory") as unused:
            temp = str(Path.cwd())
            session = CodexSession("codex", Path(temp), 1)
            session.tmp = type("Tmp", (), {"name": temp})()
            response = {"model": MODEL, "reasoningEffort": "high", "instructionSources": [], "thread": {"id": "test"}}
            with patch.object(session, "rpc", return_value=response) as rpc:
                session.start_thread("high")
                params = rpc.call_args.args[1]
                self.assertEqual(params["environments"], [])
                self.assertEqual(params["dynamicTools"], [])
                self.assertEqual(params["sandbox"], "read-only")
                self.assertEqual(params["approvalPolicy"], "never")
            with patch.object(session, "rpc", return_value=response):
                with self.assertRaises(BridgeError):
                    session.start_thread("low")

    def test_external_instruction_source_rejected(self):
        session = CodexSession("codex", Path("."), 1)
        session.tmp = type("Tmp", (), {"name": str(Path.cwd())})()
        response = {"model": MODEL, "reasoningEffort": "low", "instructionSources": [str(Path.cwd() / "answers/AGENTS.md")]}
        with patch.object(session, "rpc", return_value=response):
            with self.assertRaises(BridgeError):
                session.start_thread("low")

    def test_key_tool_surfaces_disabled(self):
        for feature in ("shell_tool", "view_image", "multi_agent", "apps", "plugins", "browser_use", "computer_use", "memories"):
            self.assertIn(feature, DISABLED_FEATURES)

    def test_only_loopback_or_authenticated_docker_gateway(self):
        self.assertEqual(validate_bind_host("127.0.0.1", None), "127.0.0.1")
        for host in ("0.0.0.0", "8.8.8.8", "localhost", "::1"):
            with self.assertRaises(BridgeError):
                validate_bind_host(host, "test-token")
        with self.assertRaises(BridgeError):
            validate_bind_host("172.18.0.1", None)

    def test_docker_gateway_must_be_verified(self):
        listed = subprocess.CompletedProcess([], 0, "network-id\n", "")
        inspected = subprocess.CompletedProcess([], 0, json.dumps([{"Driver": "bridge", "IPAM": {"Config": [{"Gateway": "172.18.0.1"}]}}]), "")
        with patch("subprocess.run", side_effect=[listed, inspected]):
            self.assertEqual(validate_bind_host("172.18.0.1", "token"), "172.18.0.1")
        with patch("subprocess.run", side_effect=[listed, inspected]):
            with self.assertRaises(BridgeError):
                validate_bind_host("172.19.0.1", "token")
        with patch("subprocess.run", side_effect=OSError("没有 Docker 权限")):
            with self.assertRaises(BridgeError):
                validate_bind_host("172.18.0.1", "token")

    def test_probe_rejects_wrong_effort_and_classifies_routing(self):
        response = {"bridge_metadata": {"transport": "codex-app-server-structured-output-v1", "reasoning_effort": "medium", "api_equivalent": False}}
        with self.assertRaises(ValueError):
            check_response("text", response, "low")
        self.assertEqual(classify_failure("workspace routing discovery failed"), "routing_discovery_failure")

    def test_probe_checks_business_arguments(self):
        response = {"bridge_metadata": {"transport": "codex-app-server-structured-output-v1", "reasoning_effort": "low", "api_equivalent": False}, "choices": [{"message": {"tool_calls": [{"function": {"name": "lookup_order", "arguments": '{"order_id":"TEST-ORDER-001","user_id":"TEST-USER-001"}'}}]}}]}
        self.assertIn("tool_calls", check_response("tool", response, "low"))
        response["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = '{}'
        with self.assertRaises(ValueError):
            check_response("tool", response, "low")


if __name__ == "__main__":
    unittest.main()
