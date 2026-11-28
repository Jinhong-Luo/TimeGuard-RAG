"""将原生重排请求交给模型分词器截断，使用的证据正文仍由 RAGFlow 保留。"""
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import urllib.error
import urllib.request

PORT = 18565
UPSTREAM = "http://timeguard_ljh-r1-rerank/rerank"
SOURCE_HASH = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class Handler(BaseHTTPRequestHandler):
    def respond(self, code, data):
        raw = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path == "/health":
            self.respond(200, {"source_sha256": SOURCE_HASH, "truncate": True})
        else:
            self.respond(404, {"error": "路径不存在"})

    def do_POST(self):
        if self.path != "/rerank":
            return self.respond(404, {"error": "路径不存在"})
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 8 * 1024 * 1024:
                return self.respond(413, {"error": "请求过大或为空"})
            data = json.loads(self.rfile.read(size))
            if not isinstance(data, dict) or not isinstance(data.get("query"), str) or not isinstance(data.get("texts"), list):
                return self.respond(400, {"error": "重排请求字段不完整"})
            req = urllib.request.Request(UPSTREAM, json.dumps({**data, "truncate": True}).encode(),
                                         headers={"Content-Type": "application/json"})
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(req, timeout=28) as response:
                scores = json.load(response)
            self.respond(200, scores)
        except urllib.error.HTTPError as exc:
            self.respond(exc.code, {"error": "TEI 拒绝重排输入"})
        except (ValueError, OSError):
            self.respond(502, {"error": "重排服务未返回有效结果"})

    def log_message(self, fmt, *args):
        pass


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()

