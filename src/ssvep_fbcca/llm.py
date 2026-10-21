"""大语言模型 API 适配器。

这个模块只负责把已经确认的文本草稿发送给大语言模型。它不读取 EEG、
不修改 TrialLedger，也不负责决定何时发送；发送时机仍由上层输入状态机控制。
"""

from __future__ import annotations

from dataclasses import dataclass
import getpass
import os
from pathlib import Path
import ssl
import stat
import tempfile
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import json
from typing import Any, Iterable, Mapping


class ChatBackendError(RuntimeError):
    """聊天请求失败或响应格式无效。"""


DEFAULT_API_KEY_FILE = Path("~/.config/ssvep_fbcca/deepseek.json").expanduser()


def _api_key_path(path: str | Path | None = None) -> Path:
    """解析本地 API 配置文件路径。"""
    configured = path or os.getenv("DEEPSEEK_API_CONFIG")
    return Path(configured).expanduser() if configured else DEFAULT_API_KEY_FILE


def load_local_api_key(path: str | Path | None = None) -> str | None:
    """从权限受限的 JSON 文件读取 API key；文件不存在时返回 ``None``。"""
    source = _api_key_path(path)
    if not source.exists():
        return None
    if not source.is_file():
        raise ChatBackendError(f"DeepSeek API 配置路径不是文件：{source}")
    if os.name == "posix":
        mode = stat.S_IMODE(source.stat().st_mode)
        if mode & 0o077:
            raise ChatBackendError(
                f"DeepSeek API 配置文件权限过宽：{source}；请执行 chmod 600 {source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ChatBackendError(f"无法读取 DeepSeek API 配置文件：{source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ChatBackendError(f"DeepSeek API 配置必须是 JSON 对象：{source}")
    value = payload.get("api_key", payload.get("deepseek_api_key"))
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ChatBackendError(f"DeepSeek API 配置中的 api_key 不是非空字符串：{source}")
    return value.strip()


def save_local_api_key(api_key: str, path: str | Path | None = None) -> Path:
    """以 600 权限原子写入本地 JSON API 配置。"""
    if not isinstance(api_key, str) or not api_key.strip():
        raise ValueError("api_key 必须是非空字符串")
    destination = _api_key_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        os.chmod(destination.parent, 0o700)
    except OSError:
        pass
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent))
    temporary = Path(temporary_name)
    try:
        if os.name == "posix":
            os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump({"api_key": api_key.strip()}, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        if os.name == "posix":
            os.chmod(destination, 0o600)
    except Exception:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise
    return destination


def configure_local_api_key(path: str | Path | None = None) -> Path:
    """交互式保存 API key；输入使用隐藏回显并要求二次确认。"""
    first = getpass.getpass("DeepSeek API key: ").strip()
    second = getpass.getpass("Confirm DeepSeek API key: ").strip()
    if not first:
        raise ValueError("API key 不能为空")
    if first != second:
        raise ValueError("两次输入的 API key 不一致")
    return save_local_api_key(first, path)


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
    请求，避免为键盘运行环境额外安装 SDK。API key 优先从显式参数或环境变量
    读取；没有环境变量时从用户目录的 JSON 配置文件读取，不写入会话记录。
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        endpoint: str | None = None,
        timeout: float = 60.0,
        api_key_env: str = "DEEPSEEK_API_KEY",
        api_key_file: str | Path | None = None,
        opener: Any | None = None,
    ) -> None:
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError("timeout 必须为正数")
        self.api_key_env = api_key_env.strip() or "DEEPSEEK_API_KEY"
        self.api_key_file = _api_key_path(api_key_file)
        self.api_key = (api_key or os.getenv(self.api_key_env)
                        or load_local_api_key(self.api_key_file))
        if not isinstance(self.api_key, str) or not self.api_key.strip():
            raise ChatBackendError(
                f"未找到 DeepSeek API key；请设置环境变量 {self.api_key_env}，"
                f"或先运行 --configure-api-key（配置文件：{self.api_key_file}）"
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


__all__ = ["ChatBackendError", "ChatMessage", "DeepSeekChatBackend",
           "DEFAULT_API_KEY_FILE", "load_local_api_key", "save_local_api_key",
           "configure_local_api_key"]
