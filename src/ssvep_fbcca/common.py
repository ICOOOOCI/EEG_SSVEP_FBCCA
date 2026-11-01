from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib
import importlib.metadata as metadata
import io
import json
import math
import os
import pickle
import platform
import queue
import struct
import sys
import tempfile
import threading
import time
import traceback
import uuid
import xml.etree.ElementTree as ET
import zipfile
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass, field, replace
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np



# 包模块位于 ``src/ssvep_fbcca``；ROOT 保持指向项目根目录，兼容原有相对
# 记录目录和诊断文件配置。
ROOT = Path(__file__).resolve().parents[2]


def resolve_record_root(directory: str | Path) -> Path:
    """将会话保存目录解析为稳定的绝对路径。

    配置和命令行都允许使用 ``~`` 或相对路径。两者必须在目录探测和
    ``SessionRecorder`` 中使用同一套规则，否则探测通过的目录可能与实际
    写入目录不一致。
    """
    path = Path(directory).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()

PYTHON_VERSION = (3, 10, 11)

FAULTS = {
    "ENVIRONMENT": (10, "运行环境不匹配", "请安装 64 位 CPython 3.10.11 及程序依赖。"),
    "LSL_LIBRARY": (20, "无法加载 MNE-LSL", "在项目虚拟环境安装 mne-lsl==1.12.0 并检查 liblsl。"),
    "LSL_NOT_FOUND": (21, "找不到 EEG LSL 流", "开启发布端的 LSL 输出并检查网络。"),
    "LSL_CONNECTION": (22, "LSL 连接或事件流异常", "检查重复流、通道配置、发布端和网络。"),
    "ACQUISITION": (30, "采样中断或 EEG 数据异常", "检查设备连接、采样率和时间戳。"),
    "DISPLAY": (40, "显示异常", "检查显示器、刷新率、显卡驱动和系统负载。"),
    "AUDIO": (41, "提示音异常", "检查扬声器、系统音量和 PsychoPy 音频设备。"),
    "SAVE": (50, "保存失败", "检查目标目录权限和磁盘空间。"),
    "INTERNAL": (70, "程序异常", "查看终端错误信息和系统临时目录中的诊断报告。"),
}

def configure_console() -> None:
    """将标准输出和标准错误流配置为容错的 UTF-8 编码。"""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass

class RuntimeFault(RuntimeError):
    """携带标准故障代码的运行时异常。"""

    def __init__(self, code: str, detail: str):
        """使用故障代码和可读详情初始化异常。"""
        self.code = code
        super().__init__(detail)

def fault_record(exc: BaseException, default: str = "INTERNAL") -> dict:
    """将异常转换为可序列化的标准故障记录。"""
    code = getattr(exc, "code", default)
    exit_code, title, action = FAULTS[code]
    return {"code": code, "exit_code": exit_code, "title": title, "detail": str(exc),
            "action": action,
            "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))}

def print_fault(fault: dict) -> None:
    """将标准故障记录及处理建议输出到标准错误流。"""
    print(f"[{fault['code']}] {fault['title']}：{fault['detail']}\n处理建议：{fault['action']}",
          file=sys.stderr, flush=True)

def environment_snapshot() -> dict:
    """采集 Python、平台、依赖版本和 LSL 库覆盖配置。"""
    return {"python": sys.version, "executable": sys.executable, "prefix": sys.prefix,
            "platform": platform.platform(), "machine": platform.machine(),
            "bits": struct.calcsize("P") * 8,
            "packages": {d.metadata["Name"]: d.version for d in metadata.distributions()
                         if d.metadata.get("Name")},
            "mne_lsl_lib_override": os.environ.get("MNE_LSL_LIB") or os.environ.get("PYLSL_LIB")}

def check_environment(*, static_ui: bool = False) -> dict:
    """检查解释器和运行依赖，静态 UI 模式可跳过 LSL 检查。"""
    issues = []
    if (sys.version_info[:3] != PYTHON_VERSION or struct.calcsize("P") != 8
            or platform.python_implementation() != "CPython"):
        issues.append(f"需要 64 位 CPython {'.'.join(map(str, PYTHON_VERSION))}；当前 {platform.python_version()}")
    for name in ("numpy", "scipy", "threadpoolctl", "psychopy.visual", "psychopy.event"):
        try:
            importlib.import_module(name)
        except Exception as exc:
            issues.append(f"{name} 无法导入：{exc}")
    result = {"python_version": platform.python_version(), "lsl_checked": not static_ui}
    if not static_ui:
        try:
            mne_lsl = importlib.import_module("mne_lsl")
            lsl = importlib.import_module("mne_lsl.lsl")
            result["mne_lsl_version"] = mne_lsl.__version__
            result["liblsl_version"] = lsl.library_version()
        except Exception as exc:
            raise RuntimeFault("LSL_LIBRARY", str(exc)) from exc
    if issues:
        raise RuntimeFault("ENVIRONMENT", "\n".join(issues) + f"\nPython：{sys.executable}")
    return result

def probe_save_directory(directory: str | Path) -> str:
    """通过写入、同步、原子替换和回读验证保存目录。"""
    directory = resolve_record_root(directory)
    try:
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".write-check-", dir=directory) as tmp:
            source, target = Path(tmp) / "probe.tmp", Path(tmp) / "probe.ok"
            with source.open("wb") as handle:
                handle.write(b"BCI write check\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(source, target)
            if target.read_bytes() != b"BCI write check\n":
                raise OSError("保存后内容校验失败")
    except Exception as exc:
        raise RuntimeFault("SAVE", f"目录 {directory} 不可可靠写入：{exc}") from exc
    return str(directory)

def write_diagnostic(report: dict) -> str | None:
    """诊断记录写入系统临时目录，避免在主程序目录留下辅助文件。"""
    name = time.strftime("run_%Y%m%dT%H%M%SZ_", time.gmtime()) + uuid.uuid4().hex[:8] + ".json"
    try:
        report["environment"] = environment_snapshot()
    except Exception as exc:
        report["environment_snapshot_error"] = str(exc)
    folder = Path(tempfile.gettempdir()) / "bci-eeg-diagnostics"
    try:
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / name
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(f"诊断报告：{path}", flush=True)
        return str(path)
    except OSError as exc:
        print(f"[SAVE] 诊断报告写入失败：{folder}：{exc}", file=sys.stderr, flush=True)
        return None

class InvalidTrial(RuntimeError):
    """数据/时序未通过质量检查；必须计入无效试次，而不是静默跳过。"""

    def __init__(self, message: str, *, epoch: Optional["Epoch"] = None,
                 code: str = "ACQUISITION"):
        """保存失败原因、可选的部分事件窗及标准故障代码。"""
        super().__init__(message)
        self.epoch = epoch
        self.code = code

class DegenerateSignalError(ValueError):
    """没有足够的变化信号，不能产生有意义的 CCA 结果。"""

class AbortSession(RuntimeError):
    """用户取消或急停。"""




__all__ = [name for name in globals() if not name.startswith('__')]

