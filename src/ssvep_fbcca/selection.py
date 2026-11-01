"""滑动窗口上的决策逻辑：分数平滑、连续有效窗口计数和轮椅状态机。

这里不涉及显示、LSL 或线程，因此可以脱离硬件单独测试。
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np


class SelectionVote:
    """平滑各窗口分数；连续若干个有效窗口指向同一候选时才算选中。"""

    def __init__(self, hold: int, ema: float):
        self.hold = int(hold)
        self.ema = float(ema)
        self.reset()

    def reset(self) -> None:
        """清空平滑状态和历史；之后只依据新的窗口判断。"""
        self.history: deque[Optional[int]] = deque(maxlen=self.hold)
        self.smoothed: Optional[np.ndarray] = None

    def smooth(self, scores: np.ndarray) -> np.ndarray:
        """返回平滑后的分数；相邻窗口高度重叠，这一步不等于增加独立证据。"""
        raw = np.asarray(scores, dtype=float)
        if self.smoothed is None or self.smoothed.shape != raw.shape:
            self.smoothed = raw.copy()
        else:
            self.smoothed = self.ema * raw + (1.0 - self.ema) * self.smoothed
        return self.smoothed

    def record(self, candidate: Optional[int]) -> Optional[int]:
        """记录本窗口候选；无效窗口会打断连续计数。稳定时返回该候选。"""
        self.history.append(candidate)
        if (candidate is not None and len(self.history) == self.hold
                and len(set(self.history)) == 1):
            return candidate
        return None


class ChairStateMachine:
    """模式1状态：选择 → 方向键二次确认 → 行驶；任何不确定都以停止结束。

    ``step`` 和 ``tick`` 返回 ``(command, outcome)``：command 是要发送的轮椅
    命令，outcome 不为空时表示本次闪烁循环结束。
    """

    SELECT, CONFIRM, MOVING = "SELECT", "CONFIRM", "MOVING"
    DIRECTIONS = ("UP", "DOWN", "LEFT", "RIGHT")

    def __init__(self, vote: SelectionVote, *, settle_s: float, confirm_timeout_s: float,
                 watchdog_s: float):
        self.vote = vote
        self.settle_s = settle_s
        self.confirm_timeout_s = confirm_timeout_s
        self.watchdog_s = watchdog_s
        self.phase = self.SELECT
        self.target: Optional[str] = None
        self.window_floor = 0.0         # 早于此 LSL 时间的 EEG 不得用于判断
        self.confirm_started = 0.0
        self.last_ok = 0.0

    def active_keys(self) -> Optional[tuple[str, ...]]:
        """当前闪烁的键；None 表示全部。行驶时保留 STOP，不产生无解码的空档。"""
        if self.phase == self.CONFIRM:
            return (self.target,)
        if self.phase == self.MOVING:
            return (self.target, "STOP")
        return None

    def step(self, candidate: Optional[str], now: float) -> tuple[Optional[str], Optional[str]]:
        """处理一个窗口的候选键；无效窗口传入 None。"""
        if self.phase == self.MOVING:
            if candidate == self.target:
                self.last_ok = now
                return self.target, None
            return "STOP", "STOPPED"     # 行驶中不等连续窗口，一次偏离就停
        if self.phase == self.CONFIRM:
            if candidate is not None and candidate != self.target:
                return None, "CANCELLED"
            if self.vote.record(candidate) is None:
                return None, None
            self.phase = self.MOVING
            self.last_ok = now
            return self.target, None
        stable = self.vote.record(candidate)
        if stable is None:
            return None, None
        if stable == "STOP":
            return "STOP", "STOP"
        if stable == "MENU":
            return "STOP", "MENU"
        self.phase = self.CONFIRM
        self.target = stable
        self.confirm_started = now
        self.window_floor = now + self.settle_s
        self.vote.reset()
        return None, None

    def tick(self, now: float) -> tuple[Optional[str], Optional[str]]:
        """每帧检查超时：确认超时取消，行驶中解码中断则停止。"""
        if self.phase == self.CONFIRM and now - self.confirm_started > self.confirm_timeout_s:
            return None, "CANCELLED"
        if self.phase == self.MOVING and now - self.last_ok > self.watchdog_s:
            return "STOP", "STOPPED"
        return None, None

    def stop_pressed(self) -> tuple[Optional[str], Optional[str]]:
        """键盘急停已发送 STOP；作废正在进行的判断。"""
        self.vote.reset()
        if self.phase == self.SELECT:
            return None, None
        return None, "STOPPED"


__all__ = ["SelectionVote", "ChairStateMachine"]
