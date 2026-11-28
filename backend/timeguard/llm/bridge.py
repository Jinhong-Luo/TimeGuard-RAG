"""通过项目模型桥发送请求，记录结构化证据筛选的响应。"""
import json
import time
import urllib.error
import urllib.parse
import urllib.request


def post_completion(base_url, payload, token, timeout):
    parsed = urllib.parse.urlparse(base_url)
    if parsed.scheme not in ("http", "https") or parsed.username or parsed.password:
        raise ValueError("裁判地址须为无内嵌凭据的 HTTP(S) 地址")
    raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    if len(raw) > 3 * 1024 * 1024:
        raise ValueError("裁判材料超过限制；不静默截断证据")
    request = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=raw,
                                     headers={"Content-Type": "application/json", "Authorization": "Bearer " + token})
    # 此连接用于服务器本机/容器桥接，不继承上游模型出口代理。
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read(4096).decode("utf-8", errors="replace").replace(token, "[已隐藏]")
        raise RuntimeError(f"裁判桥接 HTTP {error.code}：{detail}") from None


def chat(rt, role, material, log, send=post_completion):
    payload = {"model":"gpt-6.1-sol@low", "stream":False, "response_format":{"type":"json_object"},
               "messages":[{"role":"system","content":role},{"role":"user","content":json.dumps(material,ensure_ascii=False)}]}
    record = {"request":payload,"usage":None}
    log.append(record)
    start = time.monotonic()
    try:
        reply = send(rt["base_url"],payload,rt["token"],310)
        record.update(response=reply["choices"][0]["message"]["content"],usage=reply.get("usage"))
        return json.loads(record["response"])
    finally:
        record["seconds"] = time.monotonic()-start
