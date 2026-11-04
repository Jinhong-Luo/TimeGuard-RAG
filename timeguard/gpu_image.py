"""仅追加固定CUDA依赖，构建本项目专用镜像；不修改上游代码或系统驱动。"""

import argparse
import datetime
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlparse

from .common import read_json, write_json


def invoke(arguments, capture=False):
    return subprocess.run(arguments, check=True, text=True, capture_output=capture).stdout


def hash_file(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="构建本项目固定单GPU依赖镜像并验证原生DeepDOC CUDA执行")
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--wheelhouse", default=".cache/gpu-wheelhouse")
    parser.add_argument("--image", default="timeguard-ragflow:v0.27.3-torch2.8.0-cu128")
    args = parser.parse_args()
    root = Path.cwd().resolve()
    wheelhouse = Path(args.wheelhouse).resolve()
    if not wheelhouse.is_relative_to(root):
        raise ValueError("wheelhouse必须属于当前项目")
    if shutil.disk_usage(root).free < 22 * 1024**3:
        raise RuntimeError("构建前可用空间不足22GiB，停止；不清理共享资源")
    base_lock = read_json("deploy/image-lock.json")["ragflow_cpu"]
    base = json.loads(invoke(["docker", "image", "inspect", base_lock["image"]], True))[0]
    if base["Id"] != base_lock["config_digest"]:
        raise ValueError("基础镜像不符合固定摘要")
    wheel_lock = read_json("reports/gpu-wheel-lock.json")
    for item in wheel_lock:
        path = wheelhouse / unquote(urlparse(item["url"]).path.rsplit("/", 1)[1])
        if not path.is_file() or path.stat().st_size != item["size"] or hash_file(path) != item["sha256"]:
            raise ValueError("固定wheel缺失或校验失败：" + item["name"])
    constraints = root / "reports/gpu-install-constraints.txt"
    if not constraints.is_file():
        raise ValueError("缺少固定依赖约束")
    gpu = invoke(["nvidia-smi", "--id=" + args.gpu_uuid,
                  "--query-gpu=uuid,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], True).strip().split(",")
    if gpu[0].strip() != args.gpu_uuid or int(gpu[1]) > 1024 or int(gpu[2]) > 10:
        raise RuntimeError("指定GPU当前不满足空闲条件，不能抢占其他任务")
    builder = "timeguard_ljh_gpu_builder"
    existing = subprocess.run(["docker", "inspect", builder], capture_output=True)
    if existing.returncode == 0:
        raise RuntimeError("本项目GPU构建容器已存在，请先核查上次构建状态，禁止覆盖")
    started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    invoke(["docker", "create", "--name", builder, "--label", "timeguard.project=TimeGuard-RAG",
            "--cpus", "8", "--memory", "16g", "--network", "none", "--gpus", "device=" + args.gpu_uuid,
            "--mount", f"type=bind,source={wheelhouse},target=/wheelhouse,readonly",
            "--env", "DEVICE=gpu", "--env", "PIP_NO_INDEX=1",
            "--env", "PIP_CONSTRAINT=/opt/timeguard/gpu-constraints.txt",
            "--env", "TORCH_CUDA_INDEX_URL=https://download.pytorch.org/whl/cu128",
            "--entrypoint", "/bin/sh", base_lock["image"], "-c", "sleep 7200"])
    invoke(["docker", "start", builder])
    invoke(["docker", "exec", builder, "mkdir", "-p", "/opt/timeguard"])
    invoke(["docker", "cp", str(constraints), builder + ":/opt/timeguard/gpu-constraints.txt"])
    invoke(["docker", "exec", builder, "/ragflow/.venv/bin/python", "-m", "pip", "install",
            "--no-index", "--find-links=/wheelhouse", "--no-cache-dir",
            "-c", "/opt/timeguard/gpu-constraints.txt", "torch==2.8.0+cu128"])
    probe_code = '''import json,numpy as np,torch,onnxruntime as ort
from deepdoc.vision.ocr import load_model
if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
 raise RuntimeError("指定单GPU不可用")
session,options=load_model("/ragflow/rag/res/deepdoc","det")
providers=session.get_providers()
if not providers or providers[0]!="CUDAExecutionProvider":
 raise RuntimeError("原生DeepDOC静默回退到CPU")
output=session.run(None,{session.get_inputs()[0].name:np.zeros((1,3,64,64),dtype=np.float32)},options)
if not all(np.isfinite(value).all() for value in output):
 raise RuntimeError("原生CUDA推理输出存在非有限数")
print("GPU_PROBE="+json.dumps({"torch":torch.__version__,"torch_cuda":torch.version.cuda,
 "onnxruntime":ort.__version__,"providers":providers,"visible_device_count":torch.cuda.device_count(),
 "device_name":torch.cuda.get_device_name(0),"output_shapes":[list(value.shape) for value in output]}))
'''
    output = invoke(["docker", "exec", builder, "/ragflow/.venv/bin/python", "-c", probe_code], True)
    probe = json.loads(next(line.split("=", 1)[1] for line in output.splitlines() if line.startswith("GPU_PROBE=")))
    print(json.dumps(probe, ensure_ascii=False), flush=True)
    # 还原基础镜像入口，避免把构建用sleep入口带入正式服务。
    invoke(["docker", "commit", "--change", "ENTRYPOINT " + json.dumps(base["Config"].get("Entrypoint") or []),
            "--change", "CMD " + json.dumps(base["Config"].get("Cmd") or []), builder, args.image])
    derived = json.loads(invoke(["docker", "image", "inspect", args.image], True))[0]
    if derived["RootFS"]["Layers"][:len(base["RootFS"]["Layers"])] != base["RootFS"]["Layers"]:
        raise RuntimeError("派生镜像没有完整保留固定基础层")
    report = {"image": args.image, "config_digest": derived["Id"], "base_image": base_lock,
              "wheel_lock_sha256": hash_file(root / "reports/gpu-wheel-lock.json"),
              "constraints_sha256": hash_file(constraints), "gpu_uuid": args.gpu_uuid,
              "started_at_utc": started_at, "completed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "native_cuda_probe": probe, "uncompressed_bytes": derived["Size"],
              "说明": "只安装固定CUDA依赖，保留上游源代码和模型；wheelhouse只读挂载，不复制进最终镜像。"}
    write_json("deploy/gpu-image-lock.json", report)
    invoke(["docker", "rm", "--force", builder])
    print("项目GPU派生镜像已验证；当前CPU服务仍保持运行，尚未切换解析条件", flush=True)


if __name__ == "__main__":
    main()
