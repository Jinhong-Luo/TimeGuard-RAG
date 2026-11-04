"""在项目自身 RAGFlow 服务创建账户和令牌，不向终端输出凭证。"""

import argparse
import base64
import json
import os
import secrets
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

from .common import read_json, write_json


def encrypted_password(password, public_key):
    value = subprocess.run(["openssl", "pkeyutl", "-encrypt", "-pubin", "-inkey", str(public_key),
                            "-pkeyopt", "rsa_padding_mode:pkcs1"], input=base64.b64encode(password.encode()),
                           check=True, capture_output=True).stdout
    return base64.b64encode(value).decode()


class AccountClient:
    def __init__(self, base_url, timeout=180):
        self.base_url = base_url.rstrip("/") + "/api/v1"
        self.auth = None
        self.timeout = timeout
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request(self, method, path, payload=None):
        headers = {"Content-Type": "application/json"}
        if self.auth:
            headers["Authorization"] = self.auth
        req = urllib.request.Request(self.base_url + path, method=method, headers=headers,
                                     data=json.dumps(payload).encode() if payload is not None else None)
        try:
            with self.opener.open(req, timeout=self.timeout) as response:
                value = json.load(response)
                auth = response.headers.get("Authorization")
                if auth:
                    self.auth = auth
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"服务配置接口 HTTP {exc.code}：{method} {path}") from None
        if value.get("code") != 0:
            raise RuntimeError(f"服务配置错误 {value.get('code')}：{value.get('message', '')}")
        return value.get("data")


def main():
    parser = argparse.ArgumentParser(description="为本项目创建独立 RAGFlow 账户和 API key")
    parser.add_argument("--base-url", default="http://127.0.0.1:19380")
    parser.add_argument("--credentials", default="deploy/account.runtime.json")
    parser.add_argument("--embedding", action="store_true", help="注册项目内 TEI 的原生 HuggingFace 模型提供商")
    parser.add_argument("--chat", action="store_true", help="注册六档模型别名；原生提供商会进行真实连通验证")
    parser.add_argument("--bridge-base-url", default="http://172.18.0.1:18864/v1")
    args = parser.parse_args()
    path = Path(args.credentials)
    if path.exists():
        account = read_json(path)
    else:
        account = {"email": "timeguard@localhost.local", "password": secrets.token_urlsafe(32), "base_url": args.base_url}
        write_json(path, account)
        os.chmod(path, 0o600)
    client = AccountClient(args.base_url, timeout=1200 if args.chat else 180)
    payload = {"email": account["email"], "password": encrypted_password(account["password"], "vendor/ragflow/conf/public.pem")}
    if not account.get("registered"):
        try:
            client.request("POST", "/users", {**payload, "nickname": "TimeGuard"})
        except RuntimeError as exc:
            if "already registered" not in str(exc):
                raise
        account["registered"] = True
        write_json(path, account)
    client.request("POST", "/auth/login", payload)
    account["session_token"] = client.auth
    if not account.get("ragflow_api_key"):
        account["ragflow_api_key"] = client.request("POST", "/system/tokens", {})["token"]
    write_json(path, account)
    os.chmod(path, 0o600)
    print(f"项目账户与 API key 已保存到 {path}，文件权限0600；未向终端输出凭证")
    if args.embedding:
        providers = client.request("GET", "/providers")
        if not any(p["name"] == "HuggingFace" for p in providers):
            client.request("PUT", "/providers", {"provider_name": "HuggingFace"})
        instances = client.request("GET", "/providers/HuggingFace/instances")
        if not any(i["instance_name"] == "local_tei" for i in instances):
            client.request("POST", "/providers/HuggingFace/instances", {
                "instance_name": "local_tei", "api_key": "local-tei", "base_url": "http://tei-cpu:80", "region": "",
                "model_info": [{"model_name": "BAAI/bge-small-en-v1.5", "model_type": ["embedding"], "max_tokens": 512}]})
        models = client.request("GET", "/models?type=embedding")
        entries = models.get("models", []) if isinstance(models, dict) else models
        matching = [m for m in entries if m.get("name") == "BAAI/bge-small-en-v1.5"
                    and m.get("instance_name") == "local_tei" and m.get("provider_name") == "HuggingFace"]
        if not matching:
            raise RuntimeError("模型创建后未出现在租户模型列表")
        account["embedding_model"] = "BAAI/bge-small-en-v1.5@local_tei@HuggingFace"
        account["embedding_model_id"] = matching[0]["model_id"]
        write_json(path, account)
        print("项目内 TEI 模型已通过原生接口注册并验证")
    if args.chat:
        key = os.environ.get("CODEX_BRIDGE_TOKEN")
        if not key:
            raise ValueError("缺少 CODEX_BRIDGE_TOKEN；请只在当前进程环境中设置")
        provider = "OpenAI-API-Compatible"
        instance = "codex_local"
        efforts = ["low", "medium", "high", "xhigh", "max", "ultra"]
        providers = client.request("GET", "/providers")
        if not any(p["name"] == provider for p in providers):
            client.request("PUT", "/providers", {"provider_name": provider})
        instances = client.request("GET", f"/providers/{provider}/instances")
        models = client.request("GET", "/models?type=chat")
        entries = models.get("models", []) if isinstance(models, dict) else models
        present = {m.get("name") for m in entries if m.get("provider_name") == provider
                   and m.get("instance_name") == instance}
        if not {f"gpt-6.1-sol@{e}" for e in efforts} <= present:
            payload = {"instance_name": instance, "api_key": key, "base_url": args.bridge_base_url, "region": "",
                       "model_info": [{"model_name": f"gpt-6.1-sol@{effort}", "model_type": ["chat"],
                                       "max_tokens": 32768} for effort in efforts]}
            # 统一实例和固定上下文预算，只让模型别名携带档位差异。
            if any(i["instance_name"] == instance for i in instances):
                client.request("PUT", f"/providers/{provider}/instances/{instance}", payload)
            else:
                client.request("POST", f"/providers/{provider}/instances", payload)
        models = client.request("GET", "/models?type=chat")
        entries = models.get("models", []) if isinstance(models, dict) else models
        present = {m.get("name") for m in entries if m.get("provider_name") == provider
                   and m.get("instance_name") == instance}
        if not {f"gpt-6.1-sol@{e}" for e in efforts} <= present:
            raise RuntimeError("六档模型别名没有完整出现在租户模型列表")
        account["chat_models"] = {e: f"gpt-6.1-sol@{e}@{instance}@{provider}" for e in efforts}
        account["model_transport"] = "codex-app-server-structured-output-v1"
        write_json(path, account)
        print("六档模型别名已注册；仍须根据网关审计日志核实实际思考强度，不能以注册成功替代质量评测")


if __name__ == "__main__":
    main()
