"""把当前 Codex 登录适配为仅供本机实验使用的 Chat Completions 接口。

这是结构化输出传输变体，不是 OpenAI API 的等价实现，也不执行业务工具。
认证完全由 Codex 自己管理；此模块不读取、复制或显示任何登录令牌。
"""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import queue
import secrets
import shutil
import subprocess
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
import uuid


MODEL = "gpt-6.1-sol"
EFFORTS = ("low", "medium", "high", "xhigh", "max", "ultra")
TRANSPORT = "codex-app-server-structured-output-v1"
MAX_REQUEST_BYTES = 4 * 1024 * 1024
DISABLED_FEATURES = (
    "apps", "plugins", "remote_plugin", "shell_tool", "shell_snapshot",
    "view_image", "browser_use", "browser_use_external", "computer_use",
    "in_app_browser", "image_generation", "multi_agent", "multi_agent_v2",
    "memories", "hooks", "goals", "skill_search", "skill_mcp_dependency_install",
    "tool_suggest", "workspace_dependencies", "artifact", "code_mode_host",
    "standalone_web_search", "sleep_tool", "auth_elicitation",
)
SAFE_ITEM_TYPES = {"userMessage", "agentMessage", "reasoning", "contextCompaction"}

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "content": {"type": ["string", "null"]},
        "tool_calls": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "arguments": {"type": "string"},
                },
                "required": ["name", "arguments"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["content", "tool_calls"],
    "additionalProperties": False,
}

BRIDGE_INSTRUCTIONS = """你是评测中的对话模型。输入是一个包含历史 messages、业务 tools 及输出要求的 JSON 对象。
将 messages 视为待续写对话，遵循其中的 system/developer 指令，生成下一个 assistant 响应。
你只输出符合给定 JSON Schema 的对象：content 是面向当前对话的回复或 null；tool_calls 是业务调用列表。
如需业务工具，把工具名称和 JSON 编码的参数字符串写入 tool_calls，由外部原生运行器执行。
不要真正调用任何 Codex 工具，不读取文件、网络、记忆或外部数据，不自行执行业务操作。
如果不需工具，tool_calls 必须为空。不要把工具调用伪造为已经执行。不得改写或省略业务硬约束。
若 response_format 要求 JSON，content 本身必须是满足该格式的 JSON 字符串。
本轮只产生一步回复，不模拟后续用户、工具结果或整个任务轨迹。
"""


class BridgeError(RuntimeError):
    """用于向调用者报告可追溯的适配错误。"""


def parse_model_effort(body: dict[str, Any], default_effort: str) -> tuple[str, str]:
    requested = body.get("model", MODEL)
    if not isinstance(requested, str):
        raise BridgeError("model 必须是字符串")
    requested = requested.removeprefix("openai/")
    base, separator, alias_effort = requested.partition("@")
    if base != MODEL:
        raise BridgeError(f"当前桥接仅允许固定模型 {MODEL}")
    explicit = body.get("reasoning_effort")
    if explicit is None and isinstance(body.get("reasoning"), dict):
        explicit = body["reasoning"].get("effort")
    if separator and explicit is not None and alias_effort != explicit:
        raise BridgeError("模型别名与 reasoning_effort 冲突，拒绝静默覆盖")
    effort = explicit if explicit is not None else alias_effort if separator else default_effort
    if effort not in EFFORTS:
        raise BridgeError(f"不支持的思考强度：{effort}")
    return base, effort


def validate_request(body: dict[str, Any]) -> None:
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise BridgeError("messages 必须是非空列表")
    if body.get("n", 1) != 1:
        raise BridgeError("桥接仅支持 n=1")
    if body.get("parallel_tool_calls") is True:
        raise BridgeError("桥接不支持 parallel_tool_calls=true")
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in {"system", "developer", "user", "assistant", "tool"}:
            raise BridgeError("消息角色或结构无效")
        content = message.get("content")
        if content is not None and not isinstance(content, str):
            raise BridgeError("当前桥接仅支持文本消息，不接收文件或图片引用")
    for tool in body.get("tools") or []:
        if tool.get("type") != "function" or not isinstance(tool.get("function"), dict):
            raise BridgeError("只支持 function 类型的业务工具描述")


def completion_message(payload: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != {"content", "tool_calls"}:
        raise BridgeError("模型没有返回约定的结构化消息")
    content, calls = payload["content"], payload["tool_calls"]
    if content is not None and not isinstance(content, str):
        raise BridgeError("返回 content 不是文本或 null")
    if not isinstance(calls, list):
        raise BridgeError("返回 tool_calls 不是列表")
    names = {tool["function"]["name"] for tool in request.get("tools") or []}
    converted = []
    for call in calls:
        if not isinstance(call, dict) or call.get("name") not in names:
            raise BridgeError("模型请求了未提供的业务工具")
        arguments = call.get("arguments")
        if not isinstance(arguments, str) or not isinstance(json.loads(arguments), dict):
            raise BridgeError("工具参数必须是编码为字符串的 JSON 对象")
        converted.append({"id": "call_" + uuid.uuid4().hex, "type": "function", "function": {"name": call["name"], "arguments": arguments}})
    choice = request.get("tool_choice", "auto")
    if choice == "none" and converted:
        raise BridgeError("tool_choice=none 时模型仍然返回了工具调用")
    if choice == "required" and not converted:
        raise BridgeError("tool_choice=required 时模型没有返回工具调用")
    if isinstance(choice, dict):
        expected = choice.get("function", {}).get("name")
        if not converted or any(call["function"]["name"] != expected for call in converted):
            raise BridgeError("模型没有遵循指定工具选择")
    if len(converted) > 1:
        raise BridgeError("一步模型调用只允许一个业务动作")
    if not converted and content is None:
        raise BridgeError("模型返回了空响应")
    response_format = request.get("response_format") or {}
    if not converted and response_format.get("type") in {"json_object", "json_schema"}:
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            raise BridgeError("response_format 要求 JSON 对象")
    message = {"role": "assistant", "content": content}
    if converted:
        message["tool_calls"] = converted
    return message


def resolve_codex(value: str) -> str:
    resolved = shutil.which(value) or value
    if os.name == "nt" and Path(resolved).suffix.lower() in {".cmd", ".ps1", ""}:
        candidates = list((Path(os.environ.get("APPDATA", "")) / "npm/node_modules/@openai/codex/node_modules/@openai").glob("codex-win32-*/vendor/*/bin/codex.exe"))
        if candidates:
            return str(candidates[0])
    return resolved


def validate_bind_host(host: str, token: str | None) -> str:
    """只允许本机回环，或 Docker 确认的私有 bridge 网关地址。"""
    try:
        address = ipaddress.ip_address(host)
    except ValueError as error:
        raise BridgeError("--host 必须是明确的 IPv4 地址") from error
    if address.version != 4:
        raise BridgeError("当前桥接监听仅支持 IPv4")
    if address.is_loopback:
        return str(address)
    if address.is_unspecified or not address.is_private:
        raise BridgeError("禁止监听所有网卡或公网地址")
    if not token:
        raise BridgeError("监听 Docker 网关时必须配置 CODEX_BRIDGE_TOKEN")
    try:
        networks = subprocess.run(["docker", "network", "ls", "--filter", "driver=bridge", "--format", "{{.ID}}"], check=True, capture_output=True, text=True, timeout=15)
        ids = networks.stdout.split()
        if not ids:
            raise BridgeError("没有可验证的 Docker bridge 网络")
        inspected = subprocess.run(["docker", "network", "inspect", *ids], check=True, capture_output=True, text=True, timeout=15)
        gateways = {item.get("Gateway") for network in json.loads(inspected.stdout) if network.get("Driver") == "bridge" for item in network.get("IPAM", {}).get("Config", [])}
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        raise BridgeError("无法核验 Docker 网关，拒绝非回环监听") from error
    if str(address) not in gateways:
        raise BridgeError("指定地址不是当前 Docker bridge 网络的网关")
    return str(address)


class CodexSession:
    """每次请求独立 app-server 和临时线程，防止跨题历史泄漏。"""

    def __init__(self, codex: str, runtime_dir: Path, timeout: float):
        self.codex = resolve_codex(codex)
        self.runtime_dir = runtime_dir.resolve()
        self.timeout = timeout
        self.events: queue.Queue = queue.Queue()
        self.pending: list[dict] = []
        self.counter = 0
        self.process = None
        self.tmp = None
        self.diagnostics = []

    def __enter__(self):
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(prefix="request-", dir=self.runtime_dir)
        command = [self.codex, "app-server", "--stdio"]
        for feature in DISABLED_FEATURES:
            command.extend(["-c", f"features.{feature}=false"])
        # 只关闭工具与额外上下文，登录仍由原 Codex 配置管理。
        for override in (
            'web_search="disabled"', 'personality="none"',
            'project_doc_max_bytes=0', 'features.code_mode.enabled=false',
            'plugins._default.enabled=false', 'apps._default.enabled=false',
            'check_for_update_on_startup=false', 'approval_policy="never"',
            'model_provider="openai"',
            'skills.include_instructions=false', 'features.skip_host_skill_discovery=true',
        ):
            command.extend(["-c", override])
        # 禁止本地已配置的 MCP；只读配置名称，不读取登录文件或输出配置值。
        import tomllib
        config_path = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex")) / "config.toml"
        if config_path.exists():
            with config_path.open("rb") as handle:
                config = tomllib.load(handle)
            for namespace in ("mcp_servers", "plugins", "apps"):
                for name in config.get(namespace, {}):
                    if "." in name:
                        raise BridgeError("本地插件或 MCP 名称含点号，需先提供单独的无工具 Codex 配置")
                    command.extend(["-c", f"{namespace}.{name}.enabled=false"])
        self.process = subprocess.Popen(command, cwd=self.tmp.name, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", bufsize=1)
        threading.Thread(target=self._read_diagnostics, daemon=True).start()
        threading.Thread(target=self._read, daemon=True).start()
        try:
            self.rpc("initialize", {"clientInfo": {"name": "llm_projects_model_bridge", "version": "0.1.0"}, "capabilities": {"experimentalApi": True}})
        except Exception:
            self.__exit__()
            raise
        self.send({"method": "initialized", "params": {}})
        return self

    def __exit__(self, *_):
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
            for handle in (self.process.stdin, self.process.stdout, self.process.stderr):
                if handle:
                    handle.close()
        if self.tmp:
            self.tmp.cleanup()

    def _read(self):
        try:
            for line in self.process.stdout:
                try:
                    self.events.put(json.loads(line))
                except json.JSONDecodeError:
                    self.events.put({"_error": "app-server 输出了非 JSON 数据"})
        finally:
            self.events.put({"_error": "app-server 已退出"})

    def _read_diagnostics(self):
        import re
        for line in self.process.stderr:
            # 只保留启动错误；不暴露令牌、环境变量或模型请求日志。
            if line.startswith("Error:") or "invalid type:" in line or "unknown field" in line:
                line = re.sub(r"(?:sk-|eyJ)[A-Za-z0-9_.-]+", "[已隐藏]", line)
                self.diagnostics.append(line.strip()[:600])
                self.diagnostics = self.diagnostics[-3:]

    def send(self, message: dict):
        self.process.stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
        self.process.stdin.flush()

    def receive(self, deadline: float) -> dict:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise BridgeError("Codex 请求超时")
        try:
            message = self.events.get(timeout=remaining)
        except queue.Empty as error:
            raise BridgeError("Codex 请求超时") from error
        if "_error" in message:
            raise BridgeError(message["_error"] + ("；" + "；".join(self.diagnostics) if self.diagnostics else ""))
        if "method" in message and "id" in message:
            # 桥接从不批准任何执行、联网、登录或交互请求。
            self.send({"id": message["id"], "error": {"code": -32601, "message": "模型评测桥接禁止执行工具或请求额外权限"}})
            raise BridgeError("检测到被禁止的服务端请求：" + message["method"])
        return message

    def rpc(self, method: str, params: dict) -> dict:
        self.counter += 1
        request_id = self.counter
        self.send({"id": request_id, "method": method, "params": params})
        deadline = time.monotonic() + self.timeout
        while True:
            message = self.receive(deadline)
            if message.get("id") == request_id:
                if "error" in message:
                    raise BridgeError(f"{method} 失败：{message['error'].get('message', '未知错误')}")
                return message.get("result", {})
            self.pending.append(message)

    def model_info(self) -> dict:
        models = self.rpc("model/list", {"limit": 100, "includeHidden": False})
        for model in models.get("data", []):
            if model.get("id") == MODEL or model.get("model") == MODEL:
                return model
        # 独立集成的目录可能滞后；始终保留指定模型，交由真实调用决定访问权。
        return {"id": MODEL, "model": MODEL, "catalogListed": False, "supportedReasoningEfforts": []}

    def start_thread(self, effort: str) -> dict:
        # environments=[] 在协议中明确表示关闭环境访问；运行目录本身不含题库。
        started = self.rpc("thread/start", {
            "model": MODEL, "modelProvider": "openai", "cwd": self.tmp.name,
            "allowProviderModelFallback": False,
            "approvalPolicy": "never", "sandbox": "read-only", "ephemeral": True,
            "environments": [], "runtimeWorkspaceRoots": [], "dynamicTools": [],
            "baseInstructions": BRIDGE_INSTRUCTIONS,
            "developerInstructions": "本实验不得使用任何运行时工具。业务操作只能作为结构化文本返回。",
            "config": {"model_reasoning_effort": effort, "project_doc_max_bytes": 0},
        })
        if started.get("model") != MODEL:
            raise BridgeError("Codex 实际模型与固定模型不符")
        if started.get("reasoningEffort") != effort:
            raise BridgeError("Codex 实际思考强度与请求不符")
        codex_home = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
        allowed = (codex_home / "AGENTS.md").resolve()
        self.instruction_hashes = {}
        for source in started.get("instructionSources", []):
            path = Path(source).resolve()
            if path != allowed:
                raise BridgeError("Codex 加载了项目外部指令文件，拒绝污染评测")
            self.instruction_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return started

    def complete(self, body: dict[str, Any], effort: str) -> tuple[dict, dict | None]:
        model = self.model_info()
        supported = {item.get("reasoningEffort", item.get("effort")) for item in model.get("supportedReasoningEfforts", [])}
        if supported and effort not in supported:
            raise BridgeError(f"当前登录的模型目录不支持 {effort}")
        started = self.start_thread(effort)
        prompt = {key: body[key] for key in ("messages", "tools", "tool_choice", "response_format") if key in body}
        self.rpc("turn/start", {
            "threadId": started["thread"]["id"], "model": MODEL, "effort": effort,
            "environments": [], "input": [{"type": "text", "text": json.dumps(prompt, ensure_ascii=False)}],
            "outputSchema": OUTPUT_SCHEMA,
        })
        deadline = time.monotonic() + self.timeout
        final_text = None
        usage = None
        while True:
            event = self.pending.pop(0) if self.pending else self.receive(deadline)
            method, params = event.get("method"), event.get("params", {})
            if method in {"item/started", "item/completed"}:
                item = params.get("item", {})
                if item.get("type") not in SAFE_ITEM_TYPES:
                    raise BridgeError("检测到非文本运行时动作，评测无效：" + str(item.get("type")))
                if method == "item/completed" and item.get("type") == "agentMessage":
                    final_text = item.get("text")
            if method == "thread/tokenUsage/updated":
                usage = params.get("tokenUsage", {}).get("total")
            if method == "turn/completed":
                turn = params.get("turn", {})
                if turn.get("status") != "completed":
                    raise BridgeError("Codex 轮次未成功完成：" + json.dumps(turn.get("error"), ensure_ascii=False))
                break
        if not final_text:
            raise BridgeError("Codex 完成但没有返回结构化消息")
        return completion_message(json.loads(final_text), body), usage


class BridgeServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, args):
        self.args = args
        self.semaphore = threading.BoundedSemaphore(args.concurrency)
        host = validate_bind_host(args.host, os.environ.get("CODEX_BRIDGE_TOKEN"))
        super().__init__((host, args.port), BridgeHandler)


class BridgeHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        # 不记录请求正文、Authorization、问题或参考答案。
        return

    def respond(self, status: int, value: dict):
        data = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            return self.respond(200, {"status": "ready", "transport": TRANSPORT, "model": MODEL, "api_equivalent": False})
        if self.path in {"/models", "/v1/models"}:
            ids = [MODEL] + [f"{MODEL}@{effort}" for effort in EFFORTS]
            return self.respond(200, {"object": "list", "data": [{"id": model, "object": "model", "owned_by": "local-codex-bridge"} for model in ids]})
        self.respond(404, {"error": {"message": "接口不存在"}})

    def do_POST(self):
        if self.path not in {"/chat/completions", "/v1/chat/completions"}:
            return self.respond(404, {"error": {"message": "接口不存在"}})
        token = os.environ.get("CODEX_BRIDGE_TOKEN")
        if token and not secrets.compare_digest(self.headers.get("Authorization", ""), "Bearer " + token):
            return self.respond(401, {"error": {"message": "本地桥接认证失败"}})
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if size <= 0 or size > MAX_REQUEST_BYTES:
                raise BridgeError("请求大小不合法")
            body = json.loads(self.rfile.read(size))
            if not isinstance(body, dict):
                raise BridgeError("请求必须是 JSON 对象")
            validate_request(body)
            model, effort = parse_model_effort(body, self.server.args.default_effort)
            begin = time.monotonic()
            with self.server.semaphore:
                with CodexSession(self.server.args.codex, Path(self.server.args.runtime_dir), self.server.args.timeout) as session:
                    message, tokens = session.complete(body, effort)
            response = {
                "id": "chatcmpl-" + uuid.uuid4().hex, "object": "chat.completion", "created": int(time.time()),
                "model": body.get("model", model),
                "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if message.get("tool_calls") else "stop"}],
                "bridge_metadata": {"transport": TRANSPORT, "api_equivalent": False, "actual_model": MODEL, "reasoning_effort": effort, "billed_cost": None, "wall_seconds": time.monotonic() - begin, "sampling_parameters_applied": False, "multi_agent_enabled": False, "environment_access": False, "instruction_source_sha256": session.instruction_hashes},
            }
            if tokens:
                response["usage"] = {"prompt_tokens": tokens["inputTokens"], "completion_tokens": tokens["outputTokens"], "total_tokens": tokens["totalTokens"], "prompt_tokens_details": {"cached_tokens": tokens.get("cachedInputTokens", 0)}, "completion_tokens_details": {"reasoning_tokens": tokens.get("reasoningOutputTokens", 0)}}
            if body.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                delta = dict(message)
                if "tool_calls" in delta:
                    delta["tool_calls"] = [{"index": i, **call} for i, call in enumerate(delta["tool_calls"])]
                chunk = {key: response[key] for key in ("id", "created", "model")}
                chunk.update({"object": "chat.completion.chunk", "choices": [{"index": 0, "delta": delta, "finish_reason": None}]})
                self.wfile.write(("data: " + json.dumps(chunk, ensure_ascii=False) + "\n\n").encode())
                chunk["choices"] = [{"index": 0, "delta": {}, "finish_reason": response["choices"][0]["finish_reason"]}]
                if "usage" in response:
                    chunk["usage"] = response["usage"]
                self.wfile.write(("data: " + json.dumps(chunk) + "\n\ndata: [DONE]\n\n").encode())
            else:
                self.respond(200, response)
        except (BridgeError, ValueError, OSError, KeyError) as error:
            self.respond(502, {"error": {"message": str(error), "type": "codex_bridge_error"}})


def main():
    parser = argparse.ArgumentParser(description="使用现有 Codex 登录的本机结构化输出桥接")
    parser.add_argument("--codex", default="codex")
    parser.add_argument("--port", type=int, default=18089)
    parser.add_argument("--host", default="127.0.0.1", help="仅允许回环或已验证的 Docker bridge 网关；后者必须设置访问令牌")
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--default-effort", choices=EFFORTS, default="low")
    parser.add_argument("--runtime-dir", default=str(Path.cwd() / ".bridge-runtime"))
    parser.add_argument("--check", action="store_true", help="只检查模型目录和线程隔离，不运行模型")
    args = parser.parse_args()
    if args.check:
        with CodexSession(args.codex, Path(args.runtime_dir), args.timeout) as session:
            model = session.model_info()
            thread = session.start_thread(args.default_effort)
            print(json.dumps({"model": model.get("model", model.get("id")), "catalog_listed": model.get("catalogListed", True), "supported_reasoning_efforts": model.get("supportedReasoningEfforts"), "actual_model": thread["model"], "actual_effort": thread["reasoningEffort"], "instruction_source_sha256": session.instruction_hashes, "transport": TRANSPORT, "model_inference_performed": False}, ensure_ascii=False, indent=2))
        return
    server = BridgeServer(args)
    print(json.dumps({"listening": f"http://{args.host}:{args.port}/v1", "transport": TRANSPORT}, ensure_ascii=False), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
