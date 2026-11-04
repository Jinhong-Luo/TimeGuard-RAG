"""固定上游源码，并生成仅属于当前项目的受限部署配置。"""

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
    parser.add_argument("--gpu-uuid", help="启用已验证派生镜像，并只暴露指定的一张GPU")
    parser.add_argument("--gpu-image-lock", default="deploy/gpu-image-lock.json")
    parser.add_argument("--task-workers", type=int, choices=[1, 2, 4], default=1,
                        help="GPU解析的原生执行器进程数；需先验证资源余量")
    parser.add_argument("--tasks-per-worker", type=int, choices=[1, 2, 4], default=4,
                        help="每个GPU执行器的任务与分块并发上限")
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
    gpu = None
    if not args.gpu_uuid and (args.task_workers != 1 or args.tasks_per_worker != 4):
        raise ValueError("自定义进程数仅用于已验证GPU资源阶段；CPU后备保持固定配置")
    if args.gpu_uuid:
        if not re.fullmatch(r"GPU-[0-9a-fA-F-]{36}", args.gpu_uuid):
            raise ValueError("必须传入明确的单GPU UUID")
        gpu = read_json(args.gpu_image_lock)
        expected_base = read_json(deploy / "image-lock.json")["ragflow_cpu"]["config_digest"]
        if gpu["base_image"]["config_digest"] != expected_base or gpu["gpu_uuid"] != args.gpu_uuid:
            raise ValueError("GPU镜像的基础版本或已验证设备不一致")
        actual_gpu = command(["docker", "image", "inspect", "--format", "{{.Id}}", gpu["image"]], capture=True).stdout.strip()
        if actual_gpu != gpu["config_digest"]:
            raise ValueError("GPU派生镜像内容与锁文件不符")
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
    limits = {"ragflow-cpu": ("24g", 16), "es01": ("4g", 2), "mysql": ("2g", 1), "minio": ("1g", 1), "redis": ("512m", 1),
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
        if name == "ragflow-cpu":
            service_env["LLM_TIMEOUT_SECONDS"] = "330"
            service_env["LLM_MAX_RETRIES"] = "0"
            service_env.update({"MAX_CONCURRENT_TASKS": "4", "MAX_CONCURRENT_CHUNK_BUILDERS": "4",
                                "OCR_INTRA_OP_NUM_THREADS": "2", "OCR_INTER_OP_NUM_THREADS": "2"})
            if gpu:
                service["image"] = gpu["image"]
                import shlex
                original_command = service.get("command") or []
                if isinstance(original_command, str):
                    original_command = shlex.split(original_command)
                service["command"] = [part for part in original_command if not part.startswith("--workers=")]
                service["command"].append(f"--workers={args.task_workers}")
                service_env.update({"DEVICE": "gpu", "TORCH_CUDA_INDEX_URL": "https://download.pytorch.org/whl/cu128",
                                    "PIP_NO_INDEX": "1", "PIP_CONSTRAINT": "/opt/timeguard/gpu-constraints.txt",
                                    "MAX_CONCURRENT_TASKS": str(args.tasks_per_worker),
                                    "MAX_CONCURRENT_CHUNK_BUILDERS": str(args.tasks_per_worker),
                                    "OCR_GPU_MEM_LIMIT_MB": "8192",
                                    "OCR_ARENA_EXTEND_STRATEGY": "kSameAsRequested",
                                    "OCR_GPUMEM_ARENA_SHRINKAGE": "1"})
                service["deploy"] = {"resources": {"reservations": {"devices": [
                    {"driver": "nvidia", "device_ids": [args.gpu_uuid], "capabilities": ["gpu"]}]}}}
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
            # 共享大分区按绝对剩余空间保护本项目，避免95%使用率直接阻止小型基线索引。
            service_env.update({"cluster.routing.allocation.disk.watermark.low": "20gb",
                                "cluster.routing.allocation.disk.watermark.high": "15gb",
                                "cluster.routing.allocation.disk.watermark.flood_stage": "10gb"})
    config["name"] = args.project
    config["volumes"] = {}
    config["networks"] = {k: {**v, "name": args.project + "_" + k} for k, v in config.get("networks", {}).items()}
    target = deploy / "compose.generated.json"
    write_json(target, config)
    # 镜像锁是已核验的静态来源记录，不在每次生成配置时覆盖。
    if gpu:
        profile = read_json(deploy / "parse-profile.cpu.json")
        profile.update({"profile": f"single-gpu-cpu16-workers{args.task_workers}-tasks{args.tasks_per_worker}", "device": "gpu",
                        "task_executors": args.task_workers, "max_concurrent_tasks": args.tasks_per_worker,
                        "max_concurrent_chunk_builders": args.tasks_per_worker,
                        "total_task_limit": args.task_workers * args.tasks_per_worker,
                        "onnx_provider": "CUDAExecutionProvider", "image_config_digest": gpu["config_digest"],
                        "gpu_uuid": args.gpu_uuid, "visible_gpu_count": 1, "ocr_gpu_mem_limit_mb": 8192,
                        "ocr_arena_extend_strategy": "kSameAsRequested", "ocr_gpumem_arena_shrinkage": True,
                        "torch_version": gpu["native_cuda_probe"]["torch"],
                        "torch_cuda": gpu["native_cuda_probe"]["torch_cuda"],
                        "gpu_model": gpu["native_cuda_probe"]["device_name"],
                        "wheel_lock_sha256": gpu["wheel_lock_sha256"], "constraints_sha256": gpu["constraints_sha256"]})
        write_json(deploy / "parse-profile.gpu.json", profile)
    os.chmod(target, 0o600)
    print(f"已生成 {target}，仅开放回环端口 18080 / 19380；未启动容器")
    print(f"启动：docker compose -p {args.project} -f deploy/compose.generated.json up -d")


if __name__ == "__main__":
    main()
