"""离线分析适配层。

历史入口支持 ``--offline-session``，但原工程没有提供可导入的离线模块。
这里先提供稳定的模块边界和结果目录；具体离线分析器可以在不影响实时采集
和 UI 的情况下独立替换。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import zipfile


def run(session_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """读取会话 ZIP 的元数据并写出可扩展的离线分析索引。

    该适配器不伪造识别准确率；当会话包含已有摘要时只原样索引，后续算法
    可通过同一个 ``run`` 接口插入，而不需要修改 CLI 或实时路径。
    """
    session = Path(session_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if not session.is_file():
        raise FileNotFoundError(f"找不到离线会话：{session}")
    output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(session) as archive:
        members = sorted(archive.namelist())
        metadata = None
        for name in ("info.json", "metadata.json", "session.json", "progress.json"):
            if name in members:
                try:
                    metadata = json.loads(archive.read(name).decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    metadata = None
                break
    report = {
        "format": "fbcca-offline-index-v1",
        "source_session": str(session),
        "members": members,
        "metadata": metadata,
        "analysis_status": "indexed_only",
        "analysis_note": "未伪造离线识别指标；请接入独立分析器后再计算结果。",
    }
    report_path = output / "offline_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report

