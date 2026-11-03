"""菜单 EEG 的 FBCCA 参数评估；标签只参与评分，不进入解码器。"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, fields
import hashlib
import io
import json
from pathlib import Path
from typing import Any
import zipfile

import numpy as np
from threadpoolctl import threadpool_limits

from .acquisition import MNEStreamWindow
from .common import InvalidTrial
from .config import Config, _normalise_channel_indices
from .signal import assess_signal_quality, classify_eeg_window


# 与实时主菜单保持相同顺序；后三个是隐藏对照，不是 IDLE 的已验证类别。
MENU_FREQUENCIES_HZ = (8.0, 15.0, 12.0)
MENU_DECOYS_HZ = (9.6, 10.6, 13.6)


@dataclass(frozen=True)
class FbccaParameters:
    bands_hz: tuple[tuple[float, float], ...]
    n_harmonics: int
    weight_a: float = 1.25
    weight_b: float = 0.25
    regularization: float = 1e-10


def parameter_grid(baseline: FbccaParameters) -> list[FbccaParameters]:
    """固定小型探索网格；选择前确定候选，不根据验证轮次扩展搜索。"""
    band_sets = (
        baseline.bands_hz,
        ((6., 90.), (14., 90.), (22., 90.)),
        ((6., 48.), (14., 48.), (22., 48.)),
        ((6., 60.), (14., 60.), (22., 60.), (30., 60.), (38., 60.)),
    )
    candidates = [baseline]
    for bands in band_sets:
        for harmonics in (2, 3, 5):
            for a in (0., 1.25, 2.):
                for b in (0., .25):
                    candidate = FbccaParameters(bands, harmonics, a, b, baseline.regularization)
                    if candidate not in candidates:
                        candidates.append(candidate)
    return candidates


class _RecordedStream:
    def __init__(self, data: np.ndarray, timestamps: np.ndarray):
        self.data, self.timestamps = data, timestamps

    def get_data(self, *, picks: list[int]) -> tuple[np.ndarray, np.ndarray]:
        return self.data[picks], self.timestamps


def _load(archive: zipfile.ZipFile, metadata: dict[str, Any]) -> tuple:
    allowed = {item.name for item in fields(Config)}
    cfg = Config(**{k: v for k, v in metadata.get("config", {}).items() if k in allowed})
    recorded_bands = metadata.get("decoder_settings", {}).get("filter_bands_hz")
    if recorded_bands is not None:
        cfg.fbcca_filter_bands_hz = tuple(tuple(band) for band in recorded_bands)
    cfg.validate()
    with np.load(io.BytesIO(archive.read("eeg_raw.npz")), allow_pickle=False) as eeg:
        data = np.asarray(eeg["data"], dtype=float)
        timestamps = np.asarray(eeg["timestamps"], dtype=float)
        fs = float(eeg["fs"])
        indices = _normalise_channel_indices(
            eeg["eeg_channel_indices"] if "eeg_channel_indices" in eeg else cfg.channel_indices)
    if (data.ndim != 2 or timestamps.ndim != 1 or data.shape[1] != len(timestamps)
            or len(timestamps) < 3 or not np.isfinite(timestamps).all()
            or np.any(np.diff(timestamps) <= 0) or not np.isfinite(fs) or fs <= 0
            or max(indices) >= data.shape[0]):
        raise ValueError("会话 EEG/时间轴/通道索引不合法，不排序、补零或外推")
    labels = json.loads(archive.read("menu_labels.json"))
    for label, frequency in enumerate(MENU_FREQUENCIES_HZ, 1):
        if labels.get("label_map", {}).get(str(label), {}).get("frequency_hz") != frequency:
            raise ValueError("采集标签的菜单频率与实时主菜单不一致")
    trials = [trial for trial in labels["trials"] if trial.get("status") == "completed"]
    if len({trial["trial_id"] for trial in trials}) != len(trials):
        raise ValueError("菜单 trial_id 必须唯一")
    for trial in trials:
        if trial["label"] not in (0, 1, 2, 3):
            raise ValueError("菜单标签必须为 0、1、2、3")
        expected = None if trial["label"] == 0 else MENU_FREQUENCIES_HZ[trial["label"] - 1]
        if trial.get("frequency_hz") != expected:
            raise ValueError("试次频率与 label_map 不一致")
        times = (trial["stimulus_onset"], trial["stimulus_offset"])
        if not np.isfinite(times).all() or times[1] <= times[0]:
            raise ValueError("试次刺激起止时间不合法")
    return cfg, data, timestamps, fs, indices, trials


def _windows(cfg: Config, data: np.ndarray, timestamps: np.ndarray, fs: float,
             indices: np.ndarray, trials: list[dict]) -> tuple[list[dict], list[dict]]:
    source = MNEStreamWindow(_RecordedStream(data, timestamps), indices, 0.)
    windows, invalid = [], []
    for trial in trials:
        start = float(trial["stimulus_onset"]) + cfg.response_delay_s
        number = 0
        while start + cfg.window_s <= trial["stimulus_offset"] + 1e-9:
            record = {"trial_id": trial["trial_id"], "round": trial["round"],
                      "label": trial["label"], "window_index": number, "start_lsl": start}
            try:
                epoch = source.epoch(
                    start, cfg.window_s, fs, max_gap_factor=cfg.max_gap_factor,
                    hard_gap_factor=cfg.hard_gap_factor, rate_tolerance=cfg.rate_tolerance,
                    hard_rate_tolerance=cfg.hard_rate_tolerance,
                    max_warning_gaps=cfg.max_warning_gaps)
                quality = assess_signal_quality(
                    epoch.source_data, unit=cfg.input_unit, max_abs=cfg.quality_max_abs,
                    rail_min=cfg.quality_rail_min, rail_max=cfg.quality_rail_max,
                    rail_margin=cfg.quality_rail_margin,
                    max_clip_fraction=cfg.quality_max_clip_fraction, jump_z=cfg.quality_jump_z,
                    min_valid_channels=cfg.min_valid_channels,
                    max_warning_bad_channels=cfg.max_warning_bad_channels)
                if quality["hard_fail"]:
                    raise InvalidTrial("; ".join(quality["hard_reasons"]))
            except (InvalidTrial, ValueError) as exc:
                invalid.append(dict(record, reason=str(exc)))
            else:
                windows.append(dict(record, eeg=epoch.data,
                                    timestamp_estimated_fs=epoch.diagnostics["timestamp_estimated_fs"]))
            number += 1
            start = float(trial["stimulus_onset"]) + cfg.response_delay_s + number * cfg.decode_step_s
    return windows, invalid


def _metrics(scores: np.ndarray, windows: list[dict], trials: list[dict],
             selected_rounds: list[int]) -> dict[str, Any]:
    predictions = scores.argmax(axis=1) + 1
    active_trials, idle_trials = [], []
    for trial in trials:
        if trial["round"] not in selected_rounds:
            continue
        mask = np.array([window["trial_id"] == trial["trial_id"] for window in windows])
        values = predictions[mask]
        item = {"trial_id": trial["trial_id"], "label": trial["label"],
                "valid_windows": len(values)}
        if trial["label"]:
            # 一个试次一票权重；没有有效窗的试次计 0，不能通过丢弃失败抬高准确率。
            item["correct_fraction"] = float(np.mean(values == trial["label"])) if len(values) else 0.
            active_trials.append(item)
        else:
            item["visible_top_fraction"] = float(np.mean(values <= 3)) if len(values) else None
            idle_trials.append(item)
    usable_idle = [item["visible_top_fraction"] for item in idle_trials
                   if item["visible_top_fraction"] is not None]
    return {
        "rounds": selected_rounds,
        "active_trials": len(active_trials), "idle_trials": len(idle_trials),
        "active_accuracy_trial_macro": (float(np.mean([item["correct_fraction"]
                                                        for item in active_trials]))
                                         if active_trials else None),
        "idle_visible_top_fraction": float(np.mean(usable_idle)) if usable_idle else None,
        "trials_without_valid_windows": sum(item["valid_windows"] == 0
                                           for item in active_trials + idle_trials),
        "active_trial_details": active_trials, "idle_trial_details": idle_trials,
    }


def choose_on_training(summaries: list[dict]) -> int:
    """只接收训练指标；相同时先比空闲可见胜出比例，再保留更早的候选。"""
    def key(index: int) -> tuple:
        metrics = summaries[index]
        idle = metrics["idle_visible_top_fraction"]
        return (metrics["active_accuracy_trial_macro"], -(1. if idle is None else idle), -index)
    return max(range(len(summaries)), key=key)


def analyze_menu(archive: zipfile.ZipFile, metadata: dict[str, Any],
                 source_path: Path, output: Path) -> dict[str, Any]:
    cfg, data, timestamps, fs, indices, trials = _load(archive, metadata)
    rounds = sorted({trial["round"] for trial in trials})
    if len(rounds) < 2:
        raise ValueError("至少需要两轮完整采集，才能按轮次分开调参与验证")
    training, validation = rounds[:-1], rounds[-1:]
    for split in (training, validation):
        if {trial["label"] for trial in trials if trial["round"] in split} != {0, 1, 2, 3}:
            raise ValueError("调参与验证两部分都必须包含三个目标和 IDLE")
    windows, invalid = _windows(cfg, data, timestamps, fs, indices, trials)
    if not windows:
        raise ValueError("没有通过时间轴和信号质量检查的窗口")
    baseline = FbccaParameters(cfg.filter_bands, cfg.n_harmonics, cfg.weight_a,
                               cfg.weight_b, cfg.cca_regularization)
    candidates = parameter_grid(baseline)
    frequencies = (*MENU_FREQUENCIES_HZ, *MENU_DECOYS_HZ)
    cached_correlations: dict[tuple, np.ndarray] = {}
    all_scores, summaries = [], []
    with threadpool_limits(limits=1):
        for index, params in enumerate(candidates):
            cache_key = (params.bands_hz, params.n_harmonics, params.regularization)
            if cache_key not in cached_correlations:
                correlations = []
                for window in windows:
                    result = classify_eeg_window(
                        window["eeg"], fs, window_s=cfg.window_s,
                        target_frequencies_hz=frequencies, target_fs=cfg.target_fs,
                        notch_hz=cfg.notch_hz, n_harmonics=params.n_harmonics,
                        filter_bands=params.bands_hz, a=params.weight_a, b=params.weight_b,
                        regularization=params.regularization, min_valid_channels=cfg.min_valid_channels)
                    correlations.append(result["correlations"])
                cached_correlations[cache_key] = np.stack(correlations)
            weights = np.arange(1, len(params.bands_hz) + 1, dtype=float) ** (-params.weight_a) + params.weight_b
            scores = (cached_correlations[cache_key] ** 2 * weights[None, :, None]).sum(axis=1)
            all_scores.append(scores)
            summaries.append(_metrics(scores, windows, trials, training))
            if index == 0 or (index + 1) % 12 == 0:
                print(f"[FBCCA] 已评估 {index + 1}/{len(candidates)} 组参数", flush=True)
    chosen = choose_on_training(summaries)
    baseline_test = _metrics(all_scores[0], windows, trials, validation)
    candidate_test = _metrics(all_scores[chosen], windows, trials, validation)
    estimated = np.array([window["timestamp_estimated_fs"] for window in windows])
    timing = {
        "declared_fs_hz": fs,
        "timestamp_estimated_fs_min_median_max": [float(estimated.min()), float(np.median(estimated)),
                                                  float(estimated.max())],
        "windows_exceeding_recorded_rate_tolerance": int(np.sum(np.abs(estimated / fs - 1) > cfg.hard_rate_tolerance)),
        "overall_timestamp_rate_hz": float((len(timestamps) - 1) / (timestamps[-1] - timestamps[0])),
        "interpretation": "时间戳推算的到样速率；不是设备 ADC 采样率或丢样证明。保持实时路径的警告并继续策略。",
    }
    report = {
        "format": "fbcca-menu-tuning-v1", "analysis_status": "evaluated_not_applied",
        "source_session": str(source_path),
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "frequency_order_hz": frequencies, "channel_indices": indices.tolist(),
        "window_s": cfg.window_s, "response_delay_s": cfg.response_delay_s,
        "step_s": cfg.decode_step_s, "target_fs": cfg.target_fs, "notch_hz": cfg.notch_hz,
        "completed_trials": len(trials), "valid_windows": len(windows), "invalid_windows": invalid,
        "timestamp_diagnostics": timing,
        "selection_protocol": "前面完整轮次调参，最后一轮验证；先比较目标正确率，再比较 IDLE 可见目标胜出比例；重叠窗不跨轮次。",
        "candidate_count": len(candidates), "chosen_index": chosen,
        "baseline": {"parameters": asdict(baseline), "training": summaries[0], "validation": baseline_test},
        "candidate": {"parameters": asdict(candidates[chosen]), "training": summaries[chosen],
                      "validation": candidate_test},
        "training_search": [{"parameters": asdict(params), "metrics": metrics}
                            for params, metrics in zip(candidates, summaries)],
        "default_parameters_changed": False,
        "limitations": [
            "仅一人一次采集，搜索参数较多；留出轮次仅是本会话诊断，需要新会话再次验证。",
            "目标正确率为含三个隐藏对照的六候选最高分命中率，按试次平均窗口正确比例。",
            "IDLE 指标只是可见目标赢得最高分的窗口比例，不是执行门控和界面选择后的误触发率。",
            "本评估未校准拒识阈值，也未模拟 EMA、UI 切换或硬件执行。",
            "所有候选使用相同原始通道、时间窗、重采样及陷波；不根据标签筛通道或改频率。",
            "标签是要求注视的目标，软件翻帧时间未校准到实际出光时间。",
        ],
        "algorithm_source": "https://doi.org/10.1088/1741-2560/12/4/046008",
    }
    output.mkdir(parents=True, exist_ok=True)
    with (output / "window_predictions.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        score_columns = [f"score_{frequency:g}hz" for frequency in frequencies]
        writer = csv.DictWriter(handle, fieldnames=["profile", "trial_id", "round", "label",
                                                     "window_index", "start_lsl", "prediction",
                                                     "timestamp_estimated_fs", *score_columns])
        writer.writeheader()
        for name, scores in (("baseline", all_scores[0]), ("candidate", all_scores[chosen])):
            for window, values in zip(windows, scores):
                writer.writerow({"profile": name, **{key: value for key, value in window.items() if key != "eeg"},
                                 "prediction": int(values.argmax()) + 1, **dict(zip(score_columns, values))})
    b, c = baseline_test["active_accuracy_trial_macro"], candidate_test["active_accuracy_trial_macro"]
    summary = (
        f"FBCCA 菜单参数评估\n\n来源：{source_path.name}\n"
        f"完整试次：{len(trials)}；有效滑窗：{len(windows)}；无效滑窗：{len(invalid)}\n"
        f"窗长 {cfg.window_s:g}s，步长 {cfg.decode_step_s:g}s，刺激后偏移 {cfg.response_delay_s:g}s。\n"
        f"前 {training} 轮调参，第 {validation} 轮验证；共 {len(candidates)} 组。\n\n"
        f"原参数验证目标命中率：{b:.2%}\n候选参数验证目标命中率：{c:.2%}\n"
        f"候选参数：{json.dumps(asdict(candidates[chosen]), ensure_ascii=False)}\n\n"
        f"声明采样率：{fs:g} Hz；窗内时间戳速率 min/median/max："
        f"{timing['timestamp_estimated_fs_min_median_max']}\n"
        "时间戳到样速率与 ADC 采样率并不等价；不能据此直接更改设备采样率。\n\n"
        "默认参数未替换；候选参数未自动启用，原有拒识阈值不能视为已对新参数标定。\n\n"
        + "\n".join(report["limitations"])
        + "\n\n复现：PYTHONPATH=src python3 -m ssvep_fbcca --offline-session "
        + str(source_path) + " --offline-output " + str(output) + "\n"
        + "算法来源：https://doi.org/10.1088/1741-2560/12/4/046008\n"
    )
    (output / "fbcca_summary.txt").write_text(summary, encoding="utf-8")
    return report
