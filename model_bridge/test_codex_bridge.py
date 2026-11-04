"""验证桥接边界；这些测试不连接 Codex 或执行模型。"""

import json
import io
import os
from pathlib import Path
import tempfile
import subprocess
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from codex_bridge import (
    AuditLog, BridgeError, BridgeHandler, CodexSession, DISABLED_FEATURES, MODEL,
    completion_message, parse_model_effort, validate_request,
    validate_bind_host,
    audit_usage,
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

    def make_handler(self, audit, body):
        handler = object.__new__(BridgeHandler)
        handler.path = "/v1/chat/completions"
        raw = json.dumps(body).encode()
        handler.headers = {"Content-Length": str(len(raw))}
        handler.rfile = io.BytesIO(raw)
        handler.respond = Mock()
        handler.server = SimpleNamespace(audit=audit, semaphore=__import__("threading").BoundedSemaphore(1),
                                         args=SimpleNamespace(default_effort="low", codex="codex", runtime_dir=".", timeout=1))
        return handler

    def fake_session(self):
        session = Mock()
        session.actual_model = MODEL
        session.actual_effort = "high"
        session.instruction_hashes = {"AGENTS.md": "a" * 64}
        session.last_usage = {"inputTokens": 7, "outputTokens": 3, "totalTokens": 10}
        session.__enter__ = Mock(return_value=session)
        session.__exit__ = Mock(return_value=False)
        session.complete.return_value = ({"role": "assistant", "content": "回复中的私密文本"}, session.last_usage)
        return session

    def test_audit_success_has_actual_identity_but_no_conversation(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            path = Path(tmp) / "audit.jsonl"
            handler = self.make_handler(AuditLog(path), {"model": MODEL + "@high", "messages": [{"role": "user", "content": "提示中的私密文本"}]})
            with patch.dict(os.environ, {"CODEX_BRIDGE_TOKEN": ""}), patch("codex_bridge.CodexSession", return_value=self.fake_session()):
                handler.do_POST()
            raw = path.read_text(encoding="utf-8")
            row = json.loads(raw)
            self.assertTrue(row["success"])
            self.assertEqual(row["requested_model"], MODEL + "@high")
            self.assertEqual((row["actual_model"], row["reasoning_effort"]), (MODEL, "high"))
            self.assertEqual(row["usage"]["totalTokens"], 10)
            self.assertEqual(len(row["bridge_source_sha256"]), 64)
            self.assertEqual(row["request_id"], handler.respond.call_args.args[1]["id"])
            self.assertNotIn("私密文本", raw)

    def test_audit_failure_keeps_usage_not_error_text(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            path = Path(tmp) / "audit.jsonl"
            handler = self.make_handler(AuditLog(path), {"model": MODEL + "@high", "messages": [{"role": "user", "content": "问题"}]})
            session = self.fake_session()
            session.complete.side_effect = BridgeError("错误中包含秘密凭据")
            with patch.dict(os.environ, {"CODEX_BRIDGE_TOKEN": ""}), patch("codex_bridge.CodexSession", return_value=session):
                handler.do_POST()
            raw = path.read_text(encoding="utf-8")
            row = json.loads(raw)
            self.assertFalse(row["success"])
            self.assertEqual(row["error_type"], "BridgeError")
            self.assertEqual(row["usage"]["totalTokens"], 10)
            self.assertNotIn("秘密凭据", raw)

    def test_audit_auth_failure_does_not_read_body_or_header(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            path = Path(tmp) / "audit.jsonl"
            handler = self.make_handler(AuditLog(path), {"model": "不能写入日志的秘密"})
            handler.headers["Authorization"] = "Bearer 错误密钥"
            with patch.dict(os.environ, {"CODEX_BRIDGE_TOKEN": "正确密钥"}), patch("codex_bridge.CodexSession") as session:
                handler.do_POST()
            session.assert_not_called()
            raw = path.read_text(encoding="utf-8")
            self.assertEqual(json.loads(raw)["http_status"], 401)
            self.assertNotIn("密钥", raw)
            self.assertNotIn("秘密", raw)
            self.assertEqual(handler.rfile.tell(), 0)

    def test_audit_write_failure_blocks_success_response(self):
        handler = self.make_handler(Mock(append=Mock(side_effect=OSError("不可写"))), {"model": MODEL + "@high", "messages": [{"role": "user", "content": "问题"}]})
        with patch.dict(os.environ, {"CODEX_BRIDGE_TOKEN": ""}), patch("codex_bridge.CodexSession", return_value=self.fake_session()):
            handler.do_POST()
        self.assertEqual(handler.respond.call_args.args[0], 502)
        self.assertEqual(handler.respond.call_args.args[1]["error"]["type"], "audit_write_error")

    def test_audit_is_optional_and_discards_unknown_fields(self):
        AuditLog().append({"prompt": "不落盘"})
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            path = Path(tmp) / "audit.jsonl"
            AuditLog(path).append({"request_id": "abc", "prompt": "不落盘", "Authorization": "不落盘"})
            row = json.loads(path.read_text(encoding="utf-8"))
            self.assertNotIn("prompt", row)
            self.assertNotIn("Authorization", row)
        self.assertIsNone(audit_usage(None))
        self.assertEqual(audit_usage({"totalTokens": 5, "inputTokens": -1, "secret": "不能保存"}), {"totalTokens": 5})

    def test_invalid_model_value_is_not_copied_to_audit(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            path = Path(tmp) / "audit.jsonl"
            handler = self.make_handler(AuditLog(path), {"model": "夹带私密文本", "messages": []})
            with patch.dict(os.environ, {"CODEX_BRIDGE_TOKEN": ""}), patch("codex_bridge.CodexSession") as session:
                handler.do_POST()
            session.assert_not_called()
            raw = path.read_text(encoding="utf-8")
            self.assertIsNone(json.loads(raw)["requested_model"])
            self.assertNotIn("私密文本", raw)


if __name__ == "__main__":
    unittest.main()
