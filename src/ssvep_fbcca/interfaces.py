"""可替换组件的最小端口定义。

实时 LSL、离线回放和测试替身只需实现这些协议即可接入应用；UI 不需要知道
具体网络库或文件格式。
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol, runtime_checkable
import threading


@runtime_checkable
class Decoder(Protocol):
    """EEG 窗口解码器端口。"""

    def decode(self, data: Any, fs: float) -> dict[str, Any]:
        ...


@runtime_checkable
class EegSource(Protocol):
    """向试次编排提供已对齐 EEG 事件窗的端口。"""

    metadata: Mapping[str, Any]
    clock: Any
    error: Any

    def wait_epoch(self, onset: float, cancel: threading.Event, window_s: float) -> Any:
        ...

    def check_health(self) -> None:
        ...


@runtime_checkable
class EventSink(Protocol):
    """事件发布端口，可由 LSL Marker 或测试记录器实现。"""

    error: Any

    def mark(self, event: Any) -> None:
        ...


@runtime_checkable
class TrialSink(Protocol):
    """试次持久化端口。"""

    session_dir: Any

    def write_trial(self, record: Any) -> Mapping[str, str]:
        ...


__all__ = ["Decoder", "EegSource", "EventSink", "TrialSink"]

