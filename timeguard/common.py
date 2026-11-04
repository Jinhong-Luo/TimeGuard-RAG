"""运行记录、文件校验和无密钥泄露的原生接口。"""

import hashlib
import json
import os
import urllib.error
import urllib.request
import uuid
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8", newline="\n")


def digest(value):
    raw = value if isinstance(value, bytes) else json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(raw).hexdigest()


def download(url, path, expected_git_blob=None):
    path = Path(path)
    if not path.exists():
        request = urllib.request.Request(url, headers={"User-Agent": "TimeGuard-RAG/0.1"})
        with urllib.request.urlopen(request, timeout=120) as response:
            raw = response.read()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    raw = path.read_bytes()
    if expected_git_blob:
        actual = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        if actual != expected_git_blob:
            raise ValueError(f"文件校验失败：{path}")
    return digest(raw)


class Client:
    def __init__(self, base_url, key=None, timeout=300):
        self.base_url = base_url.rstrip("/")
        if not self.base_url.endswith("/api/v1"):
            self.base_url += "/api/v1"
        self.key = key or os.environ.get("RAGFLOW_API_KEY")
        if not self.key:
            raise ValueError("缺少 RAGFLOW_API_KEY；请在服务器环境变量中配置，不要写进仓库")
        self.timeout = timeout

    def request(self, method, path, body=None, content_type="application/json"):
        raw = body if isinstance(body, bytes) else json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base_url + path, data=raw, method=method,
                                     headers={"Authorization": "Bearer " + self.key, "Content-Type": content_type})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                result = json.load(response)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"RAGFlow HTTP {exc.code}：{method} {path}") from None
        if result.get("code") != 0:
            raise RuntimeError(f"RAGFlow 业务错误 {result.get('code')}：{str(result.get('message', ''))[:300]}")
        return result.get("data")

    def upload(self, dataset_id, file):
        file = Path(file)
        if any(c in file.name for c in '\r\n"'):
            raise ValueError("文件名不能包含引号或换行")
        boundary = "timeguard" + uuid.uuid4().hex
        body = (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{file.name}"\r\n'
                'Content-Type: application/pdf\r\n\r\n').encode() + file.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
        return self.request("POST", f"/datasets/{dataset_id}/documents", body, "multipart/form-data; boundary=" + boundary)
