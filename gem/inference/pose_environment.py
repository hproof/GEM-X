"""采集性能报告所需的本机软硬件信息，不改变运行配置。"""

import importlib.metadata
import os
import platform
import subprocess
import sys
from pathlib import Path


def _read_text(path):
    try:
        return Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return ""


def _version(package):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "未安装"


def collect_environment(device, detector_device=None):
    """采集一次环境快照；可选硬件查询失败时保留其他信息。"""
    import torch

    kernel = platform.release()
    is_wsl = "microsoft" in kernel.lower()
    try:
        system = platform.freedesktop_os_release().get("PRETTY_NAME", platform.system())
    except OSError:
        system = platform.system()
    cpu = platform.processor() or None
    for line in _read_text("/proc/cpuinfo").splitlines():
        if line.startswith("model name"):
            cpu = line.split(":", 1)[1].strip()
            break
    memory = None
    for line in _read_text("/proc/meminfo").splitlines():
        if line.startswith("MemTotal:"):
            memory = f"{int(line.split()[1]) / 1024**2:,.2f} GiB"
            break
    rows = [
        ["运行环境", "WSL2" if is_wsl else platform.system()],
        ["WSL 发行版", os.environ.get("WSL_DISTRO_NAME") if is_wsl else "不适用"],
        ["系统", system],
        ["系统内核", kernel],
        ["处理器", cpu],
        ["逻辑处理器数", os.cpu_count()],
        ["WSL 内存总量" if is_wsl else "系统内存总量", memory],
        ["Python", platform.python_version()],
        ["Python 环境", Path(sys.prefix).name],
        ["PyTorch", str(torch.__version__)],
        ["PyTorch CUDA 运行时", torch.version.cuda or "未启用"],
        [
            "ONNX Runtime",
            (
                _version("onnxruntime")
                if _version("onnxruntime") != "未安装"
                else _version("onnxruntime-gpu")
            ),
        ],
        ["Warp", _version("warp-lang")],
        ["SOMA-X", _version("py-soma-x")],
        ["姿态计算设备", str(device)],
        ["人体检测器设备", str(detector_device) if detector_device is not None else "不适用"],
        ["PyTorch CPU 线程数", torch.get_num_threads()],
        ["PyTorch CPU 互操作线程数", torch.get_num_interop_threads()],
    ]
    for name in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
        rows.append([name, os.environ.get(name, "未设置（由运行库决定）")])
    try:
        if torch.cuda.is_available():
            gpu_device = (
                torch.device(device) if str(device).startswith("cuda") else torch.device("cuda:0")
            )
            properties = torch.cuda.get_device_properties(gpu_device)
            rows.extend(
                [
                    ["GPU", properties.name],
                    ["GPU 显存总量", f"{properties.total_memory // 1024**2:,} MiB"],
                ]
            )
        else:
            rows.append(["CUDA 状态", "不可用"])
    except (RuntimeError, AssertionError, OSError):
        rows.append(["GPU", "无法读取"])
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=3,
            check=True,
        )
        drivers = ", ".join(dict.fromkeys(result.stdout.strip().splitlines()))
    except (OSError, subprocess.SubprocessError):
        drivers = None
    rows.append(["NVIDIA 驱动", drivers or "无法读取或未安装"])
    return rows
