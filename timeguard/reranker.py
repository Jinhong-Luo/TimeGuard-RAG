"""为调优实验准备固定版本的 CPU 重排服务，只接入项目 Docker 内网。"""
from pathlib import Path
import subprocess
import time
import urllib.request
import json
from .common import Client, digest, read_json, write_json

MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
IMAGE = "sha256:5368d285d6044ce93aa157c6d24877ff2896540ca1fea42cfd7c7c0e619ed823"
CONTAINER = "timeguard_ljh-r1-rerank"
NATIVE_CONTAINER = "timeguard_ljh-ragflow-cpu-1"


def compatible_endpoint():
    adapter = Path(__file__).with_name("tei_adapter.py")
    fingerprint = digest(adapter.read_bytes())
    endpoint = "http://127.0.0.1:18565"
    def health():
        response = subprocess.run(["docker", "exec", NATIVE_CONTAINER, "curl", "-fsS", "--max-time", "2",
                                   endpoint + "/health"], capture_output=True)
        return json.loads(response.stdout) if not response.returncode else None
    status = health()
    if not status:
        path = f"/tmp/timeguard-tei-{fingerprint[:12]}.py"
        subprocess.run(["docker", "cp", str(adapter), NATIVE_CONTAINER + ":" + path], check=True)
        subprocess.run(["docker", "exec", "-d", NATIVE_CONTAINER, "python", path], check=True)
        for _ in range(10):
            time.sleep(.3)
            status = health()
            if status:
                break
    if status != {"source_sha256": fingerprint, "truncate": True}:
        raise ValueError("原生重排适配器版本或状态不一致")
    return endpoint, fingerprint


def main():
    root = Path("data/model-cache/r1-reranker", REVISION).resolve()
    root.mkdir(parents=True, exist_ok=True)
    files = ("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "model.safetensors")
    meta = json.load(urllib.request.urlopen(f"https://huggingface.co/api/models/{MODEL}/revision/{REVISION}?blobs=true", timeout=60))
    if meta["sha"] != REVISION:
        raise ValueError("模型版本不匹配")
    known = {f["rfilename"]: f for f in meta["siblings"]}
    hashes = {}
    for name in files:
        target = root / name
        if not target.exists():
            temp = target.with_suffix(".partial")
            subprocess.run(["curl", "-fLsS", "--retry", "2", "--max-time", "300",
                            f"https://huggingface.co/{MODEL}/resolve/{REVISION}/{name}", "-o", str(temp)], check=True)
            if temp.stat().st_size != known[name]["size"]:
                raise ValueError("模型下载不完整：" + name)
            temp.replace(target)
        hashes[name] = digest(target.read_bytes())
        expected = known[name].get("lfs", {}).get("sha256")
        if target.stat().st_size != known[name]["size"] or expected and expected != hashes[name]:
            raise ValueError("模型文件校验失败：" + name)
    if subprocess.run(["docker", "container", "inspect", CONTAINER], stdout=subprocess.DEVNULL,
                      stderr=subprocess.DEVNULL).returncode:
        subprocess.run(["docker", "run", "-d", "--name", CONTAINER, "--network", "timeguard_ljh_ragflow",
                        "--cpus", "3", "--memory", "3g", "-e", "OMP_NUM_THREADS=3",
                        "-v", f"{root}:/model:ro", IMAGE, "--model-id", "/model", "--dtype", "float32",
                        "--auto-truncate"], check=True)
    info = json.loads(subprocess.check_output(["docker", "inspect", CONTAINER]))[0]
    mounts = [m for m in info["Mounts"] if m["Destination"] == "/model"]
    if (info["Image"] != IMAGE or "timeguard_ljh_ragflow" not in info["NetworkSettings"]["Networks"]
            or len(mounts) != 1 or Path(mounts[0]["Source"]) != root or mounts[0]["RW"]):
        raise ValueError("同名容器与本项目固定镜像、模型或网络不一致")
    if not info["State"]["Running"]:
        subprocess.run(["docker", "start", CONTAINER], check=True)
    for _ in range(45):
        check = subprocess.run(["docker", "exec", CONTAINER, "curl", "-fsS", "http://127.0.0.1/health"],
                               capture_output=True)
        if not check.returncode:
            break
        time.sleep(2)
    else:
        raise RuntimeError("重排容器尚未就绪")
    config = read_json("configs/r0-low.json")
    client = Client(config["base_url"])
    endpoint, adapter_hash = compatible_endpoint()
    instance = "timeguard_r1_ce_tokenizer"
    instances = client.request("GET", "/providers/HuggingFace/instances")
    if not any(i["instance_name"] == instance for i in instances):
        client.request("POST", "/providers/HuggingFace/instances", {"instance_name": instance,
            "api_key": "local-tei", "base_url": endpoint, "region": "",
            "model_info": [{"model_name": MODEL, "model_type": ["rerank"], "max_tokens": 512}]})
    write_json("reports/r1-reranker-compatible.json", {"model": MODEL, "revision": REVISION, "files": hashes,
               "image": IMAGE, "adapter_sha256": adapter_hash, "alias": MODEL + "@" + instance + "@HuggingFace"})


if __name__ == "__main__":
    main()

