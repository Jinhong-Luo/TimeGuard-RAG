"""通过进程级代理下载并校验项目镜像，不修改 Docker 守护进程。"""

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

from .common import read_json, write_json


def main():
    parser = argparse.ArgumentParser(description="使用已验证的 crane 下载镜像，然后导入本机 Docker")
    parser.add_argument("--crane", default=".cache/crane")
    parser.add_argument("--proxy", default="http://127.0.0.1:17893")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    env = dict(os.environ)
    env.update({"HTTP_PROXY": args.proxy, "HTTPS_PROXY": args.proxy})
    images = sorted({s["image"] for s in read_json("deploy/compose.generated.json")["services"].values()})
    cache = Path(".cache/images").resolve()
    cache.mkdir(parents=True, exist_ok=True)
    crane = str(Path(args.crane).resolve())

    def invoke(cmd):
        return subprocess.run(cmd, env=env, check=True, capture_output=True, text=True).stdout

    def fetch(image):
        if shutil.disk_usage(cache).free < 18 * 1024**3:
            raise RuntimeError("可用磁盘少于18GiB，停止新的镜像下载；不自动清理其他资源")
        sha = invoke([crane, "digest", "--platform", "linux/amd64", image]).strip()
        manifest = json.loads(invoke([crane, "manifest", "--platform", "linux/amd64", image]))
        if "huggingface/text-embeddings-inference" in image and sha != "sha256:c466c97680cc9c2968108c4b1b44ca7729a091a44b61c840d9487f07d42e9099":
            raise ValueError("TEI镜像摘要与已核查版本不一致")
        expected_config = manifest["config"]["digest"]
        existing = subprocess.run(["docker", "image", "inspect", "--format", "{{.Id}}", image], capture_output=True, text=True)
        if existing.returncode == 0 and existing.stdout.strip() == expected_config:
            print(f"已有匹配镜像：{image}", flush=True)
        else:
            target = cache / (hashlib.sha256(image.encode()).hexdigest()[:16] + ".tar")
            print(f"开始代理下载：{image}", flush=True)
            invoke([crane, "pull", "--platform", "linux/amd64", image, str(target)])
            # 下载前后再次核对不可变摘要，防止同名标签在过程中变更。
            if invoke([crane, "digest", "--platform", "linux/amd64", image]).strip() != sha:
                raise ValueError("镜像标签在下载过程中变化，停止导入")
            invoke(["docker", "load", "--input", str(target)])
            actual = invoke(["docker", "image", "inspect", "--format", "{{.Id}}", image]).strip()
            if actual != expected_config:
                raise ValueError("导入后的镜像配置摘要不匹配")
            if target.parent == cache:
                target.unlink()
            print(f"已导入并验证：{image}", flush=True)
        return {"image": image, "linux_amd64_manifest_digest": sha, "config_digest": expected_config,
                "compressed_bytes": sum(layer["size"] for layer in manifest["layers"])}

    records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(fetch, image): image for image in images}
        for future in concurrent.futures.as_completed(pending):
            records.append(future.result())
            write_json("reports/deployed_images.json", records)


if __name__ == "__main__":
    main()
