"""离线分析：菜单会话评估 FBCCA；其他历史会话保留元数据索引。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import zipfile


def run(session_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """读取会话 ZIP，菜单采集按完整轮次分开调参和验证。"""
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
        if "menu_labels.json" in members and "eeg_raw.npz" in members:
            from .fbcca_tuning import analyze_menu
            report = analyze_menu(archive, metadata or {}, session, output)
            report_path = output / "offline_report.json"
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            return report
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

