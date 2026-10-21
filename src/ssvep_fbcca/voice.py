"""模式三语音播报：把 LLM 回复和选择确认读出来，供无法用手操作的用户（如 ALS）使用。

后端按平台选择：Windows 使用系统 SAPI（pywin32），可读取当前朗读到的字符位置，
用于让聊天区滚动与语音同步；macOS 使用 ``say``；其它情况静默降级为无语音。
所有方法都在主线程调用且立即返回，不会阻塞刺激呈现。
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from typing import Optional

_CJK = re.compile(r"[⺀-鿿豈-﫿＀-￯]")


def has_cjk(text: str) -> bool:
    """文本含中日韩字符时使用中文语音。"""
    return bool(_CJK.search(text))


class Speaker:
    """无语音后端：接口与真实后端一致，所有调用均为空操作。"""

    available = False

    def speak(self, text: str) -> None:
        """打断当前朗读并开始朗读新文本。"""

    def stop(self) -> None:
        """立即停止朗读。"""

    def is_speaking(self) -> bool:
        """是否仍在朗读。"""
        return False

    def progress(self) -> Optional[int]:
        """当前朗读到的字符下标；后端不支持时返回 None，由调用方按时间估计。"""
        return None

    def close(self) -> None:
        """释放后端资源。"""
        self.stop()


class SapiSpeaker(Speaker):
    """Windows SAPI 5 后端，异步朗读并可查询字符进度。"""

    available = True
    _ASYNC, _PURGE = 1, 2
    _DONE = 1

    def __init__(self, rate: int = 0):
        import win32com.client  # noqa: PLC0415 - Windows-only optional dependency

        self._voice = win32com.client.Dispatch("SAPI.SpVoice")
        self._voice.Rate = max(-10, min(10, int(rate)))
        voices = self._voice.GetVoices()
        descriptions = [voices.Item(i).GetDescription() for i in range(voices.Count)]
        self._voices = {"default": self._voice.Voice}
        for index, description in enumerate(descriptions):
            lowered = description.lower()
            if "english" in lowered and "en" not in self._voices:
                self._voices["en"] = voices.Item(index)
            if "chinese" in lowered and "zh" not in self._voices:
                self._voices["zh"] = voices.Item(index)

    def speak(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        self._voice.Voice = self._voices.get("zh" if has_cjk(text) else "en", self._voices["default"])
        self._voice.Speak(text, self._ASYNC | self._PURGE)

    def stop(self) -> None:
        self._voice.Speak("", self._ASYNC | self._PURGE)

    def is_speaking(self) -> bool:
        return self._voice.Status.RunningState != self._DONE

    def progress(self) -> Optional[int]:
        return int(self._voice.Status.InputWordPosition)


class SaySpeaker(Speaker):
    """macOS ``say`` 后端；无进度信息，调用方按估计语速滚动。"""

    available = True

    def __init__(self, rate: int = 0):
        self._words_per_minute = str(int(175 * (1 + max(-10, min(10, rate)) / 20)))
        self._process: Optional[subprocess.Popen] = None

    def speak(self, text: str) -> None:
        self.stop()
        text = text.strip()
        if text:
            self._process = subprocess.Popen(["say", "-r", self._words_per_minute, text],
                                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def stop(self) -> None:
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
        self._process = None

    def is_speaking(self) -> bool:
        return self._process is not None and self._process.poll() is None


def make_speaker(enabled: bool, rate: int = 0) -> Speaker:
    """按平台创建语音后端；任何初始化失败都降级为静默，不影响实验。"""
    if not enabled:
        return Speaker()
    try:
        if sys.platform == "win32":
            return SapiSpeaker(rate)
        if sys.platform == "darwin" and shutil.which("say"):
            return SaySpeaker(rate)
    except Exception as exc:  # noqa: BLE001 - voice is optional; never break the session
        print(f"[语音] 无法初始化语音播报，继续无语音运行：{exc}", file=sys.stderr, flush=True)
    return Speaker()


__all__ = ["Speaker", "SapiSpeaker", "SaySpeaker", "make_speaker", "has_cjk"]
