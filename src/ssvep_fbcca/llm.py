"""ChatGPT API 适配器。

这个模块只负责把已经确认的文本草稿发送给大语言模型。它不读取 EEG、
不修改 TrialLedger，也不负责决定何时发送；发送时机仍由上层输入状态机控制。
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Iterable, Mapping, Protocol


class ChatBackendError(RuntimeError):
    """ChatGPT 请求失败或响应格式无效。"""


class ResponsesClient(Protocol):
    """OpenAI SDK client.responses 的最小端口，便于离线测试。"""

    def create(self, **kwargs: Any) -> Any:
        ...


@dataclass(frozen=True)
class ChatMessage:
    """一条对话消息。"""

    role: str
    content: str


def _normalize_messages(messages: Iterable[ChatMessage | Mapping[str, str]]) -> list[dict[str, str]]:
    """校验并转换对话历史，避免把任意对象直接送到远端。"""
    normalized: list[dict[str, str]] = []
    for message in messages:
        if isinstance(message, ChatMessage):
            role, content = message.role, message.content
        else:
            role, content = message.get("role"), message.get("content")
        if role not in {"user", "assistant", "system", "developer"}:
            raise ValueError(f"不支持的消息角色：{role!r}")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("消息内容必须是非空字符串")
        normalized.append({"role": role, "content": content})
    return normalized


class OpenAIChatBackend:
    """使用 OpenAI Responses API 发送文本对话。"""

    def __init__(
        self,
        *,
        model: str | None = None,
        instructions: str | None = None,
        client: Any | None = None,
        timeout: float = 60.0,
    ) -> None:
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError("timeout 必须为正数")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-5-mini")
        if not self.model.strip():
            raise ValueError("model 不能为空")
        self.instructions = instructions
        self.timeout = float(timeout)
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ChatBackendError(
                    "未安装 OpenAI SDK；请执行 .venv/bin/python -m pip install -e '.[llm]'"
                ) from exc
            try:
                client = OpenAI(timeout=self.timeout)
            except Exception as exc:
                raise ChatBackendError(
                    "无法创建 OpenAI 客户端；请确认 OPENAI_API_KEY 已设置"
                ) from exc
        self._responses: ResponsesClient = client.responses

    def send(
        self,
        text: str,
        *,
        history: Iterable[ChatMessage | Mapping[str, str]] = (),
    ) -> str:
        """发送一条已确认的用户消息并返回纯文本回复。"""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("发送内容不能为空")
        messages = _normalize_messages(history)
        messages.append({"role": "user", "content": text})
        request: dict[str, Any] = {"model": self.model, "input": messages}
        if self.instructions:
            request["instructions"] = self.instructions
        try:
            response = self._responses.create(**request)
            answer = getattr(response, "output_text", None)
        except Exception as exc:
            raise ChatBackendError(f"ChatGPT API 请求失败：{exc}") from exc
        if not isinstance(answer, str) or not answer.strip():
            raise ChatBackendError("ChatGPT API 返回了空文本响应")
        return answer.strip()


__all__ = ["ChatBackendError", "ChatMessage", "OpenAIChatBackend"]
