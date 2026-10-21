"""HTTP client for the Ubuntu wheelchair bridge.

The client is deliberately synchronous because it is called after a completed
SSVEP trial on the PsychoPy thread.  Every movement gets a monotonically
increasing sequence number; ``close()`` always attempts a STOP.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
import uuid
from typing import Any, Optional


DEFAULT_DIRECTION_BY_CLASS = {1: "forward", 2: "backward", 3: "left", 4: "right"}


class WheelchairBridgeError(RuntimeError):
    """The HTTP bridge rejected a request or could not be reached."""


class WheelchairBridgeClient:
    """Small dependency-free client for ``ubuntu_chair_server.py``."""

    def __init__(self, base_url: str, *, token: str = "", client_id: Optional[str] = None,
                 strength: int = 25, duration_s: float = 0.6, timeout_s: float = 1.5):
        base_url = str(base_url).strip().rstrip("/")
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("轮椅地址必须以 http:// 或 https:// 开头")
        if not 1 <= int(strength) <= 100:
            raise ValueError("轮椅力度必须为1..100")
        if not 0.1 <= float(duration_s) <= 3.0:
            raise ValueError("轮椅动作时长必须为0.1..3.0秒")
        self.base_url = base_url
        self.token = str(token)
        self.client_id = client_id or f"ssvep-{uuid.uuid4().hex[:12]}"
        self.strength = int(strength)
        self.duration_s = float(duration_s)
        self.timeout_s = float(timeout_s)
        self._seq = 0
        self._lock = threading.Lock()
        self._closed = False

    @property
    def next_sequence(self) -> int:
        return self._seq + 1

    def _request(self, method: str, path: str, payload: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {"Accept": "application/json"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        if self.token:
            headers["X-Chair-Token"] = self.token
        request = urllib.request.Request(self.base_url + path, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                raw = response.read().decode("utf-8", errors="replace")
                return json.loads(raw) if raw else {}
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            detail = str(exc)
            if isinstance(exc, urllib.error.HTTPError):
                try:
                    detail = exc.read().decode("utf-8", errors="replace")[:500]
                except Exception:
                    pass
            raise WheelchairBridgeError(f"轮椅HTTP请求失败 {method} {path}: {detail}") from exc

    def status(self) -> dict[str, Any]:
        return self._request("GET", "/v1/status")

    def move(self, direction: str, *, strength: Optional[int] = None,
             duration_s: Optional[float] = None) -> dict[str, Any]:
        direction = str(direction).strip().lower()
        if direction not in ("forward", "backward", "left", "right"):
            raise ValueError(f"不支持的轮椅方向: {direction}")
        with self._lock:
            if self._closed:
                raise WheelchairBridgeError("轮椅桥接客户端已关闭")
            self._seq += 1
            payload = {"client_id": self.client_id, "seq": self._seq,
                       "direction": direction,
                       "strength": self.strength if strength is None else int(strength),
                       "duration_s": self.duration_s if duration_s is None else float(duration_s)}
            return self._request("POST", "/v1/move", payload)

    def stop(self, *, disarm: bool = False) -> dict[str, Any]:
        with self._lock:
            self._seq += 1
            payload = {"client_id": self.client_id, "seq": self._seq, "disarm": bool(disarm)}
            return self._request("POST", "/v1/stop", payload)

    def close(self) -> None:
        if self._closed:
            return
        try:
            self.stop(disarm=True)
        except Exception as exc:
            print(f"[轮椅] STOP发送失败：{exc}", flush=True)
        finally:
            self._closed = True


__all__ = ["DEFAULT_DIRECTION_BY_CLASS", "WheelchairBridgeClient", "WheelchairBridgeError"]
