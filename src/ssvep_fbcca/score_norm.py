"""模式3按频率的 FBCCA 分数归一化：用提示式采集估计每个频率“未被注视”时的分数分布。

同一用户在部分频率（例如 13–14 Hz）上的 FBCCA 分数天然偏高，未注视时也常胜出。
本模块从模式3采集 ZIP 中按实时规则截取滑动窗，对 8.0–15.8 Hz 网格的每个频率，
只用目标不是该频率的窗口（含 IDLE）估计均值和标准差；实时选键时改用
z = (分数 - 均值) / 标准差。标签只用于决定哪些窗口算“未注视”，不进入解码器。

    python -m ssvep_fbcca.score_norm chatbox_collection/session_XXXX.zip \
        --output config/chat_score_norm_P01.json
"""

from __future__ import annotations

import argparse
import io
import json
import time
import zipfile
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from .common import ROOT
from .config import BENCHMARK_FREQUENCIES_HZ, Config
from .signal import classify_eeg_window

FORMAT = "chat-score-norm-v1"
MIN_WINDOWS = 30


def _decoder_settings(profile: Config) -> dict[str, Any]:
    """归一化文件与实时解码必须一致的参数。"""
    return {"window_s": float(profile.window_s), "n_harmonics": int(profile.n_harmonics),
            "filter_bands_hz": [list(map(float, band)) for band in profile.filter_bands],
            "weight_a": float(profile.weight_a), "weight_b": float(profile.weight_b),
            "regularization": float(profile.cca_regularization),
            "notch_hz": profile.notch_hz, "target_fs": float(profile.target_fs),
            "line_regression_hz": profile.line_regression_hz}


def build(session_zip: str | Path, cfg: Config) -> dict[str, Any]:
    """从模式3采集 ZIP 计算 40 个网格频率的未注视分数均值/标准差。"""
    profile = cfg.chat_page_profile(1)
    path = Path(session_zip).expanduser().resolve()
    with zipfile.ZipFile(path) as archive:
        labels = json.loads(archive.read("chatbox_labels.json"))
        with np.load(io.BytesIO(archive.read("eeg_raw.npz")), allow_pickle=False) as raw:
            data = np.asarray(raw["data"], dtype=float)[raw["eeg_channel_indices"]]
            stamps = np.asarray(raw["timestamps"], dtype=float)
            fs = float(raw["fs"])
    frequencies = np.round(np.asarray(BENCHMARK_FREQUENCIES_HZ, dtype=float), 1)
    window_n = int(np.ceil(profile.window_s * fs - 1e-9))
    first = cfg.response_delay_s + cfg.chat_gaze_shift_s
    scores: list[list[float]] = [[] for _ in frequencies]
    used_trials = 0
    for trial in labels["trials"]:
        if trial.get("status") != "completed" or not trial.get("usable"):
            continue
        onset, offset = trial["stimulus_onset"], trial["stimulus_offset"]
        target = trial.get("frequency_hz")
        start = onset + first
        windows = 0
        while start + profile.window_s <= offset + 1e-9:
            i0 = int(np.searchsorted(stamps, start))
            if i0 + window_n > data.shape[1]:
                break
            result = classify_eeg_window(
                data[:, i0:i0 + window_n], fs, window_s=profile.window_s,
                target_frequencies_hz=frequencies, n_harmonics=profile.n_harmonics,
                target_fs=profile.target_fs, notch_hz=profile.notch_hz,
                a=profile.weight_a, b=profile.weight_b, regularization=profile.cca_regularization,
                min_valid_channels=profile.min_valid_channels, filter_bands=profile.filter_bands,
                line_regression_hz=profile.line_regression_hz)
            for index, value in enumerate(result["scores"]):
                if target is None or abs(frequencies[index] - target) > 1e-6:
                    scores[index].append(float(value))
            windows += 1
            start += cfg.decode_step_s
        used_trials += windows > 0
    counts = [len(values) for values in scores]
    if min(counts) < MIN_WINDOWS:
        raise ValueError(f"每个频率至少需要 {MIN_WINDOWS} 个未注视窗口，实际最少 {min(counts)} 个；"
                         "请确认刺激时长覆盖 response_delay_s + chat_gaze_shift_s + chat_window_s")
    return {"format": FORMAT, "source_session": path.name,
            "participant": labels.get("participant"),
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "decoder": _decoder_settings(profile),
            "window_first_offset_s": first, "window_step_s": cfg.decode_step_s,
            "trials_used": used_trials, "idle_collected": bool(labels["protocol"].get("idle_collected")),
            "frequencies_hz": frequencies.tolist(),
            "mean": [float(np.mean(v)) for v in scores],
            "std": [float(np.std(v)) for v in scores],
            "n_windows": counts}


def load(path: str | Path) -> dict[str, Any]:
    """读取并检查归一化文件；相对路径以项目根目录为基准。"""
    source = Path(path).expanduser()
    if not source.is_absolute():
        source = ROOT / source
    norm = json.loads(source.read_text(encoding="utf-8"))
    if norm.get("format") != FORMAT:
        raise ValueError(f"{source.name} 不是 {FORMAT} 格式")
    std = np.asarray(norm["std"], dtype=float)
    if (len(norm["frequencies_hz"]) != len(norm["mean"]) or len(std) != len(norm["mean"])
            or not np.all(np.isfinite(std)) or np.any(std <= 0)):
        raise ValueError(f"{source.name} 的均值/标准差不完整")
    norm["path"] = str(source)
    return norm


def vectors(norm: Optional[dict[str, Any]], profile: Config,
            frequencies: Sequence[float]) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """按解码顺序返回 (均值, 标准差)；参数不一致或缺频率时返回 None（回退原始分数）。"""
    if norm is None:
        return None
    if norm["decoder"] != json.loads(json.dumps(_decoder_settings(profile))):
        return None
    table = {round(float(f), 1): (m, s) for f, m, s in
             zip(norm["frequencies_hz"], norm["mean"], norm["std"])}
    try:
        pairs = [table[round(float(f), 1)] for f in frequencies]
    except KeyError:
        return None
    mean, std = (np.asarray(values, dtype=float) for values in zip(*pairs))
    return mean, std


def cli(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="由模式3采集 ZIP 生成按频率的分数归一化文件")
    parser.add_argument("session", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "config/default.toml")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    norm = build(args.session, Config.from_toml(args.config))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(norm, ensure_ascii=False, indent=2), encoding="utf-8")
    order = np.argsort(norm["mean"])[::-1][:5]
    print(f"已写入 {args.output}：{norm['trials_used']} 个试次，每频率 {min(norm['n_windows'])}–"
          f"{max(norm['n_windows'])} 个未注视窗口。未注视时分数最高的频率："
          + "，".join(f"{norm['frequencies_hz'][i]:g} Hz" for i in order))
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
