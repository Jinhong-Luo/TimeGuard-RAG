"""固定上游源码，并生成仅属于当前项目的 CPU 部署配置。"""

import argparse
import json
import os
import secrets
import subprocess
from pathlib import Path

from .common import read_json, write_json


def command(args, cwd=None, capture=False):
    return subprocess.run(args, cwd=cwd, check=True, text=True, encoding="utf-8", capture_output=capture)


def main():
    parser = argparse.ArgumentParser(description="获取固定源码；不自动启动或停止容器")
    parser.add_argument("--prepare-compose", action="store_true")
    parser.add_argument("--local-embedding", action="store_true", help="启用原生 TEI CPU 英文轻量 Embedding 服务")
    parser.add_argument("--project", default="timeguard_ljh")
    args = parser.parse_args()
    root = Path.cwd()
    lock = read_json(root / "upstream.lock.json")["ragflow"]
    source = root / "vendor/ragflow"
    if not source.exists():
        command(["git", "clone", "--depth", "1", "--branch", lock["tag"], lock["url"], str(source)])
    actual = command(["git", "rev-parse", "HEAD"], source, True).stdout.strip()
    if actual != lock["commit"]:
        raise ValueError(f"上游版本不符：{actual}，预期 {lock['commit']}；不自动覆盖已有修改")
    print(f"已验证 RAGFlow {lock['tag']}：{actual}")
    if not args.prepare_compose:
        return
    import re
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]+", args.project):
        raise ValueError("项目名必须是小写英文、数字、下划线或连字符")
    deploy = root / "deploy"
    deploy.mkdir(exist_ok=True)
    runtime = deploy / "runtime.env"
    if not runtime.exists():
        overrides = {k: secrets.token_hex(24) for k in ["ELASTIC_PASSWORD", "MYSQL_PASSWORD", "MINIO_PASSWORD", "REDIS_PASSWORD"]}
        overrides.update({"DOC_ENGINE": "elasticsearch", "DEVICE": "cpu", "DB_TYPE": "mysql",
                          "METADATA_DB_PROFILE": "mysql", "COMPOSE_PROFILES": "elasticsearch,cpu,metadata-mysql",
                          "RAGFLOW_IMAGE": lock["image"], "MEM_LIMIT": "4294967296"})
        # 从上游默认配置起步，只覆盖当前项目所需的变量；文件含随机密码，不提交。
        text = (source / "docker/.env").read_text(encoding="utf-8")
        for key, value in overrides.items():
            pattern = rf"(?m)^{key}=.*$"
            text = re.sub(pattern, f"{key}={value}", text) if re.search(pattern, text) else text + f"\n{key}={value}\n"
        runtime.write_text(text, encoding="utf-8")
        os.chmod(runtime, 0o600)
    if args.local_embedding:
        text = runtime.read_text(encoding="utf-8")
        for key, value in {"COMPOSE_PROFILES": "elasticsearch,cpu,metadata-mysql,tei-cpu",
                           "TEI_MODEL": "BAAI/bge-small-en-v1.5", "TEI_HOST": "tei-cpu"}.items():
            text = re.sub(rf"(?m)^{key}=.*$", f"{key}={value}", text)
        runtime.write_text(text, encoding="utf-8")
    env = {}
    for line in runtime.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.lstrip().startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            env[key] = value
    result = command(["docker", "compose", "--env-file", str(runtime), "-f", str(source / "docker/docker-compose.yml"),
                      "--project-name", args.project, "config", "--format", "json"], capture=True)
    config = json.loads(result.stdout)
    allowed = {"ragflow-cpu", "es01", "mysql", "minio", "redis"}
    if args.local_embedding:
        allowed.add("tei-cpu")
    if not allowed <= config["services"].keys():
        raise ValueError("上游缺少预期的 CPU/Elasticsearch/MySQL/MinIO/Redis 服务，停止生成")
    config["services"] = {k: v for k, v in config["services"].items() if k in allowed}
    limits = {"ragflow-cpu": ("8g", 4), "es01": ("4g", 2), "mysql": ("2g", 1), "minio": ("1g", 1), "redis": ("512m", 1),
              "tei-cpu": ("2g", 2)}
    for name, service in config["services"].items():
        service.pop("container_name", None)
        service.pop("profiles", None)
        service.pop("deploy", None)
        service["ports"] = []
        if name == "ragflow-cpu":
            service["ports"] = [{"target": 80, "published": "18080", "host_ip": "127.0.0.1", "protocol": "tcp"},
                                {"target": 9380, "published": "19380", "host_ip": "127.0.0.1", "protocol": "tcp"}]
        if "depends_on" in service:
            service["depends_on"] = {k: v for k, v in service["depends_on"].items() if k in allowed}
        service["mem_limit"], service["cpus"] = limits[name]
        service["logging"] = {"driver": "json-file", "options": {"max-size": "10m", "max-file": "3"}}
        service["restart"] = "unless-stopped"
        # 原生 env_file 中的默认密码也同步替换，确保模板和服务使用一致凭证。
        service_env = service.setdefault("environment", {})
        for key in ["ELASTIC_PASSWORD", "MYSQL_PASSWORD", "MINIO_PASSWORD", "REDIS_PASSWORD", "DOC_ENGINE", "DEVICE", "DB_TYPE",
                    "COMPOSE_PROFILES", "TEI_MODEL", "TEI_HOST"]:
            if key in service_env or name == "ragflow-cpu":
                service_env[key] = env[key]
        if name == "mysql":
            service_env["MYSQL_ROOT_PASSWORD"] = env["MYSQL_PASSWORD"]
        if name == "minio":
            service_env["MINIO_ROOT_PASSWORD"] = env["MINIO_PASSWORD"]
        if name == "redis":
            service["command"] = ["redis-server", "--requirepass", env["REDIS_PASSWORD"], "--maxmemory", "256mb",
                                  "--maxmemory-policy", "volatile-lru"]
        if name == "tei-cpu":
            service["image"] = "ghcr.io/huggingface/text-embeddings-inference:cpu-1.8"
            service["command"] = ["--model-id", "BAAI/bge-small-en-v1.5", "--revision",
                                  "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a", "--auto-truncate"]
            model_cache = root / "data/model-cache"
            model_cache.mkdir(parents=True, exist_ok=True)
            service["volumes"] = [{"type": "bind", "source": str(model_cache), "target": "/data"}]
        for volume in service.get("volumes", []):
            if volume.get("type") == "volume":
                directory = root / "data/services" / volume["source"]
                directory.mkdir(parents=True, exist_ok=True)
                volume.update({"type": "bind", "source": str(directory)})
                volume.pop("volume", None)
        if name == "es01":
            # Elasticsearch 以当前目录所有者运行，避免对共享主机使用全局宽权限。
            service["user"] = f"{os.getuid() if hasattr(os, 'getuid') else 1000}:0"
            service_env["ES_JAVA_OPTS"] = "-Xms1g -Xmx1g"
    config["name"] = args.project
    config["volumes"] = {}
    config["networks"] = {k: {**v, "name": args.project + "_" + k} for k, v in config.get("networks", {}).items()}
    target = deploy / "compose.generated.json"
    write_json(target, config)
    write_json(deploy / "image-lock.json", {"tei_cpu": {"image": "ghcr.io/huggingface/text-embeddings-inference:cpu-1.8",
                "linux_amd64_manifest_digest": "sha256:c466c97680cc9c2968108c4b1b44ca7729a091a44b61c840d9487f07d42e9099",
                "说明": "同版本官方轻量TEI镜像，保留原生HuggingFaceEmbed协议；镜像拉取或归档导入时验证摘要。"}})
    os.chmod(target, 0o600)
    print(f"已生成 {target}，仅开放回环端口 18080 / 19380；未启动容器")
    print(f"启动：docker compose -p {args.project} -f deploy/compose.generated.json up -d")


if __name__ == "__main__":
    main()
