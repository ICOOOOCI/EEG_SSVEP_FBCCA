"""大语言模型 API 适配器。

这个模块只负责把已经确认的文本草稿发送给大语言模型。它不读取 EEG、
不修改 TrialLedger，也不负责决定何时发送；发送时机仍由上层输入状态机控制。
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import ssl
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import json
from typing import Any, Iterable, Mapping


class ChatBackendError(RuntimeError):
    """聊天请求失败或响应格式无效。"""


def _verified_urlopen(request: Request, *, timeout: float) -> Any:
    """使用 certifi 根证书发起 HTTPS 请求，仍然执行完整证书校验。

    部分 macOS Python/虚拟环境没有把系统根证书接入 OpenSSL 默认路径，导致
    标准库 ``urlopen`` 报 ``CERTIFICATE_VERIFY_FAILED``。没有 certifi 时保留
    Python 默认上下文，让错误信息明确暴露环境缺失，而不是降低安全级别。
    """
    context = ssl.create_default_context()
    try:
        import certifi
    except ImportError:
        return urlopen(request, timeout=timeout, context=context)
    context.load_verify_locations(cafile=certifi.where())
    return urlopen(request, timeout=timeout, context=context)


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


class DeepSeekChatBackend:
    """DeepSeek Chat Completions API 适配器。

    DeepSeek 使用标准的聊天消息格式，这里直接使用 Python 标准库发起
    请求，避免为键盘运行环境额外安装 SDK。API key 只从环境变量读取，不写入
    会话记录；默认变量名为 ``DEEPSEEK_API_KEY``。
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        endpoint: str | None = None,
        timeout: float = 60.0,
        api_key_env: str = "DEEPSEEK_API_KEY",
        opener: Any | None = None,
    ) -> None:
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError("timeout 必须为正数")
        self.api_key_env = api_key_env.strip() or "DEEPSEEK_API_KEY"
        self.api_key = api_key or os.getenv(self.api_key_env)
        if not isinstance(self.api_key, str) or not self.api_key.strip():
            raise ChatBackendError(
                f"未找到 DeepSeek API key；请设置环境变量 {self.api_key_env}"
            )
        self.model = (model or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")).strip()
        self.endpoint = (endpoint or os.getenv(
            "DEEPSEEK_API_URL", "https://api.deepseek.com/chat/completions"
        )).strip()
        if not self.model:
            raise ValueError("DeepSeek model 不能为空")
        if not self.endpoint.startswith(("https://", "http://")):
            raise ValueError("DeepSeek endpoint 必须是 HTTP(S) URL")
        self.timeout = float(timeout)
        self._opener = opener or _verified_urlopen

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
        messages.append({"role": "user", "content": text.strip()})
        payload = json.dumps({"model": self.model, "messages": messages,
                              "stream": False}, ensure_ascii=False).encode("utf-8")
        request = Request(self.endpoint, data=payload, method="POST", headers={
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        })
        try:
            with self._opener(request, timeout=self.timeout) as response:
                body = response.read()
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise ChatBackendError(f"DeepSeek API HTTP {exc.code}: {detail}") from exc
        except ssl.SSLCertVerificationError as exc:
            raise ChatBackendError(
                "DeepSeek API TLS证书校验失败；请更新 certifi 或检查系统证书，"
                "不要关闭 HTTPS 证书校验"
            ) from exc
        except (OSError, URLError) as exc:
            raise ChatBackendError(f"DeepSeek API 网络请求失败：{exc}") from exc
        except Exception as exc:
            raise ChatBackendError(f"DeepSeek API 请求失败：{exc}") from exc
        try:
            data = json.loads(body.decode("utf-8") if isinstance(body, bytes) else body)
            answer = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ChatBackendError("DeepSeek API 返回格式无效或没有回复文本") from exc
        if not isinstance(answer, str) or not answer.strip():
            raise ChatBackendError("DeepSeek API 返回了空文本响应")
        return answer.strip()


__all__ = ["ChatBackendError", "ChatMessage", "DeepSeekChatBackend"]
