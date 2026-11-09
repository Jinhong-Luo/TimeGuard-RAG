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
    instances = client.request("GET", "/providers/HuggingFace/instances")
    if not any(i["instance_name"] == "timeguard_r1_ce" for i in instances):
        client.request("POST", "/providers/HuggingFace/instances", {"instance_name": "timeguard_r1_ce",
            "api_key": "local-tei", "base_url": "http://" + CONTAINER, "region": "",
            "model_info": [{"model_name": MODEL, "model_type": ["rerank"], "max_tokens": 512}]})
    write_json("reports/r1-reranker.json", {"model": MODEL, "revision": REVISION, "files": hashes,
               "image": IMAGE, "alias": MODEL + "@timeguard_r1_ce@HuggingFace"})


if __name__ == "__main__":
    main()

