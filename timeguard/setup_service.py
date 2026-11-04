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
    def __init__(self, base_url):
        self.base_url = base_url.rstrip("/") + "/api/v1"
        self.auth = None
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request(self, method, path, payload=None):
        headers = {"Content-Type": "application/json"}
        if self.auth:
            headers["Authorization"] = self.auth
        req = urllib.request.Request(self.base_url + path, method=method, headers=headers,
                                     data=json.dumps(payload).encode() if payload is not None else None)
        try:
            with self.opener.open(req, timeout=180) as response:
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
    args = parser.parse_args()
    path = Path(args.credentials)
    if path.exists():
        account = read_json(path)
    else:
        account = {"email": "timeguard@localhost.local", "password": secrets.token_urlsafe(32), "base_url": args.base_url}
        write_json(path, account)
        os.chmod(path, 0o600)
    client = AccountClient(args.base_url)
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
        matching = [m for m in entries if m.get("model_name") == "BAAI/bge-small-en-v1.5" and m.get("model_instance") == "local_tei"]
        if not matching:
            raise RuntimeError("模型创建后未出现在租户模型列表")
        account["embedding_model"] = "BAAI/bge-small-en-v1.5@local_tei@HuggingFace"
        account["embedding_model_id"] = matching[0]["model_id"]
        write_json(path, account)
        print("项目内 TEI 模型已通过原生接口注册并验证")


if __name__ == "__main__":
    main()
