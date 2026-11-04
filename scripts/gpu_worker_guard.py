#!/usr/bin/env python3
"""在原生解析进程领取任务前验证 GPU，并复用原生模型缓存进入原入口。"""

from __future__ import annotations

import contextlib
import datetime as dt
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import runpy
import secrets
import sys
import time


class GPUVerificationError(RuntimeError):
    """设备、原生会话或探针结果未达到启动条件。"""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@contextlib.contextmanager
def quiet_preflight():
    """屏蔽第三方预检输出，失败时只由守卫输出异常类别。"""
    old_disabled = logging.root.manager.disable
    saved = []
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        logging.disable(logging.CRITICAL)
        with open(os.devnull, "w") as sink:
            for descriptor in (1, 2):
                saved.append((descriptor, os.dup(descriptor)))
                os.dup2(sink.fileno(), descriptor)
            yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        for descriptor, original in reversed(saved):
            os.dup2(original, descriptor)
            os.close(original)
        logging.disable(old_disabled)


def atomic_private_json(path: Path, value: dict) -> None:
    """同目录原子替换证据，临时文件和最终文件权限均为 0600。"""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(6)}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        temporary.unlink(missing_ok=True)


def native_identity(arguments: list[str]) -> tuple[str, str]:
    """直接复用原生参数解析器，保留旧位置参数的优先级。"""
    from rag.svr.task_executor_cli import parse_task_executor_args, resolve_task_executor_index

    parsed = parse_task_executor_args(arguments)
    index = resolve_task_executor_index(parsed)
    consumer = f"task_executor_{parsed.type}_{index}"
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,180}", consumer):
        raise GPUVerificationError("消费者标识不适合证据文件名")
    return consumer, str(index)


@contextlib.contextmanager
def serialized_preflight(root: Path):
    """只串行设备预检和模型加载；门闩等待及原生执行不持有锁。"""
    import fcntl

    path = root / "logs" / ".timeguard_gpu_preflight.lock"
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    started = time.monotonic()
    try:
        os.fchmod(descriptor, 0o600)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield time.monotonic() - started
    finally:
        os.close(descriptor)


def verify_rotation() -> dict:
    """严格按固定原生源码解释默认值，只验证用户要求，不改环境。"""
    raw = os.environ.get("TABLE_AUTO_ROTATE")
    effective = os.getenv("TABLE_AUTO_ROTATE", "true").lower() in ("true", "1", "yes")
    if not effective:
        raise GPUVerificationError("表格自动旋转必须保持原生启用状态")
    return {"raw_environment_value": raw, "native_default": "true", "effective": effective}


def verify_gpu(root: Path) -> dict:
    """验证真实分配及原生会话；不修补上游线程参数或更改解析语义。"""
    rotation = verify_rotation()
    import numpy as np
    import onnxruntime as ort
    import torch
    from deepdoc.vision import ocr

    if os.environ.get("DEVICE") != "gpu":
        raise GPUVerificationError("设备配置不是 GPU")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise GPUVerificationError("必须且只能看到项目分配的一张 GPU")
    # 实际分配并执行，不把库导入成功或驱动版本当作设备可用证据。
    probe = torch.ones(16, device="cuda:0", dtype=torch.float32)
    result = probe.sum()
    torch.cuda.synchronize(0)
    if result.item() != 16.0:
        raise GPUVerificationError("CUDA 分配与运算验证失败")
    del result, probe
    torch.cuda.empty_cache()

    rows = []
    model_directory = root / "rag" / "res" / "deepdoc"
    for name, device in (("det", 0), ("rec", 0), ("layout", None), ("tsr", None)):
        # 与 OCR 和 Recognizer 的原生缓存键一致，不创建第二套验证会话。
        session, run_options = ocr.load_model(str(model_directory), name, device_id=device)
        providers = session.get_providers()
        if not providers or providers[0] != "CUDAExecutionProvider":
            raise GPUVerificationError("原生模型未优先使用 CUDA 执行器")
        configuration = session.get_provider_options().get("CUDAExecutionProvider", {})
        if str(configuration.get("device_id")) != "0":
            raise GPUVerificationError("原生模型 CUDA 设备编号不匹配")
        options = session.get_session_options()
        row = {
            "name": name,
            "native_device_argument": device,
            "providers": providers,
            "cuda_provider_options": {
                key: configuration.get(key)
                for key in ("device_id", "gpu_mem_limit", "arena_extend_strategy")
            },
            "requested_threads": {
                "intra_op": int(os.environ.get("OCR_INTRA_OP_NUM_THREADS", "2")),
                "inter_op": int(os.environ.get("OCR_INTER_OP_NUM_THREADS", "2")),
            },
            "actual_session_options": {
                "intra_op_num_threads": options.intra_op_num_threads,
                "inter_op_num_threads": options.inter_op_num_threads,
                "execution_mode": str(options.execution_mode),
                "enable_cpu_mem_arena": options.enable_cpu_mem_arena,
            },
        }
        if name == "det":
            inputs = session.get_inputs()
            if len(inputs) != 1 or inputs[0].type != "tensor(float)":
                raise GPUVerificationError("检测模型输入与固定探针不兼容")
            output = session.run(None, {inputs[0].name: np.zeros((1, 3, 64, 64), dtype=np.float32)}, run_options)
            if not output or any(not isinstance(value, np.ndarray) or not value.size or not np.isfinite(value).all() for value in output):
                raise GPUVerificationError("检测模型实际推理未产生有限结果")
            torch.cuda.synchronize(0)
            if session.get_providers()[0] != "CUDAExecutionProvider":
                raise GPUVerificationError("检测探针后发生执行器回退")
            row["synthetic_inference"] = {
                "input_shape": [1, 3, 64, 64],
                "output_shapes": [list(value.shape) for value in output],
                "finite": True,
                "用途": "仅检查执行链路，不作为解析质量或吞吐成绩",
            }
            del output
        rows.append(row)
        # 会话仅由上游缓存持有；守卫不额外保留模型或推理输出。
        del session, run_options, options

    return {
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda),
        "onnxruntime_version": ort.__version__,
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_name": torch.cuda.get_device_name(0),
        "cuda_device_uuid": str(getattr(torch.cuda.get_device_properties(0), "uuid", "未提供")),
        "cuda_allocation_and_operation_passed": True,
        "table_auto_rotate": rotation,
        "sessions": rows,
        "说明": "四个原生缓存会话在同一进程中复用；CPU 执行器可作为原生后备，但 CUDA 必须始终是首选。",
    }


def wait_for_release(path: Path, generation: str) -> None:
    """只有明确匹配本代次的门闩才允许进入原生入口。"""
    if not path.is_absolute():
        raise GPUVerificationError("门闩必须使用绝对路径")
    while True:
        try:
            with path.open("r", encoding="utf-8") as stream:
                value = stream.read(257)
            if len(value) > 256:
                raise GPUVerificationError("门闩内容超出长度限制")
            if value.strip() == generation:
                return
        except FileNotFoundError:
            pass
        time.sleep(1)


def main() -> int:
    ready_path = None
    native_argv = sys.argv[1:]
    try:
        root = Path("/ragflow").resolve()
        if not native_argv or native_argv[0] != "rag/svr/task_executor.py" or Path.cwd().resolve() != root:
            raise GPUVerificationError("仅允许固定原生执行器入口")
        entrypoint = root / native_argv[0]
        generation = os.environ.get("TIMEGUARD_RUNTIME_GENERATION", "")
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", generation):
            raise GPUVerificationError("缺少有效运行代次")
        # 替换守卫目录为原生脚本目录，并保持项目根目录可导入。
        sys.path[0] = str(entrypoint.parent)
        if str(root) not in sys.path:
            sys.path.insert(1, str(root))
        sys.argv = native_argv
        os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
        with quiet_preflight():
            consumer, index = native_identity(native_argv[1:])
            ready_path = root / "logs" / f"timeguard_guard_{consumer}.json"
            ready_path.unlink(missing_ok=True)
            with serialized_preflight(root) as lock_wait:
                evidence = verify_gpu(root)
                evidence["preflight_lock_wait_seconds"] = lock_wait
        release = os.environ.get("TIMEGUARD_WORKER_RELEASE")
        report = {
            "schema_version": 1,
            "generation": generation,
            "consumer": consumer,
            "native_index": index,
            "pid": os.getpid(),
            "preflight_ready": True,
            "preflight_completed_at_utc": utc_now(),
            "phase": "waiting_release" if release else "entering_native",
            "release_required": bool(release),
            "environment_whitelist": {
                key: os.environ.get(key)
                for key in (
                    "DEVICE", "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES",
                    "MAX_CONCURRENT_TASKS", "MAX_CONCURRENT_CHUNK_BUILDERS",
                    "OCR_GPU_MEM_LIMIT_MB", "OCR_ARENA_EXTEND_STRATEGY",
                    "OCR_GPUMEM_ARENA_SHRINKAGE", "OCR_INTRA_OP_NUM_THREADS",
                    "OCR_INTER_OP_NUM_THREADS",
                    "TABLE_AUTO_ROTATE",
                )
            },
            "gpu": evidence,
            "source_sha256": {
                "guard": sha256(Path(__file__).resolve()),
                "native_entrypoint": sha256(entrypoint),
                "native_ocr": sha256(root / "deepdoc/vision/ocr.py"),
                "native_recognizer": sha256(root / "deepdoc/vision/recognizer.py"),
                "native_pdf_parser": sha256(root / "deepdoc/parser/pdf_parser.py"),
            },
            "注意": "ready 仅表示本进程通过设备预检；恢复脚本必须同时核对 generation、pid 存活及 phase。",
        }
        atomic_private_json(ready_path, report)
        if release:
            wait_for_release(Path(release), generation)
            report["released_at_utc"] = utc_now()
            report["phase"] = "entering_native"
            atomic_private_json(ready_path, report)
    except (Exception, SystemExit) as error:
        if ready_path is not None:
            try:
                ready_path.unlink(missing_ok=True)
            except OSError:
                pass
        print(f"TIMEGUARD_GPU_GUARD_FAILED {type(error).__name__}", flush=True)
        time.sleep(60)
        return 1

    # 仅守卫失败采用上述限速；原生运行、退出码与异常交由原生监督程序处理。
    runpy.run_path(str(entrypoint), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
