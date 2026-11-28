"""为独立容器工作进程收集源码、包初始化文件并冻结它们的实际字节。"""
from pathlib import Path

from timeguard.common import digest

PACKAGE_ROOT = Path(__file__).resolve().parent


def package_files(*sources):
    files = {"__init__.py", "common.py", "worker_bundle.py", *sources}
    for source in sources:
        for parent in Path(source).parents:
            files.add((parent / "__init__.py").as_posix())
    return sorted(files)


def source_hashes(files):
    return {name: digest((PACKAGE_ROOT / name).read_bytes()) for name in files}


def stage_package(destination, files):
    """保持子包层级，并对实际复制的内容计算清单。"""
    hashes = {}
    for name in files:
        data = (PACKAGE_ROOT / name).read_bytes()
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        hashes[name] = digest(data)
    return hashes
