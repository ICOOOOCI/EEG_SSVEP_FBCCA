"""轮椅输出端口：模式1的识别结果都从这里发出。

目前没有连接真实轮椅，只打印命令。接入时只需修改 ``send``，不必改动界面和算法。
"""

from __future__ import annotations

WHEELCHAIR_COMMANDS = ("UP", "DOWN", "LEFT", "RIGHT", "STOP")


class WheelchairPort:
    """把命令发送给轮椅。发送失败时应抛出异常，程序会尝试停止并结束会话。"""

    def send(self, command: str) -> None:
        """发送 ``WHEELCHAIR_COMMANDS`` 之一；方向命令每轮重发一次，可作为心跳。"""
        print(f"[轮椅] {command}（尚未连接真实轮椅）", flush=True)


__all__ = ["WHEELCHAIR_COMMANDS", "WheelchairPort"]
