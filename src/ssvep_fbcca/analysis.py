from __future__ import annotations

from .common import *
from .config import *
from .signal import *
from .acquisition import *
from .events import *



def frame_diagnostics(flip_times: Sequence[float], refresh_hz: float,
                      long_factor: float = 1.5, short_factor: float = .5,
                      max_warning_anomalies: int = 1,
                      rate_tolerance: float = .005) -> dict:
    """根据逐帧刺激使用的刷新率检查软件翻帧时间。"""
    ts = np.asarray(flip_times, float)
    if (ts.ndim != 1 or len(ts) < 2 or not np.isfinite(ts).all()
            or not np.isfinite(refresh_hz) or refresh_hz <= 0):
        raise InvalidTrial("刺激帧时间记录不足或非法")
    if not np.isfinite(rate_tolerance) or not 0 < rate_tolerance < 1:
        raise ValueError("rate_tolerance 必须是(0, 1)内的有限比例")
    intervals = np.diff(ts)
    period = 1 / refresh_hz
    elapsed = float(ts[-1] - ts[0])
    observed_rate = float(len(intervals) / elapsed) if elapsed > 0 else None
    rate_error = ((observed_rate / refresh_hz - 1) if observed_rate is not None else None)
    rate_mismatch = rate_error is None or abs(rate_error) > rate_tolerance
    long = intervals > long_factor * period
    short = intervals < short_factor * period
    nonmonotonic = intervals <= 0
    severe = (intervals >= 2 * period) | nonmonotonic
    missed = np.maximum(1, np.rint(intervals[long] / period).astype(int) - 1).sum() if np.any(long) else 0
    anomaly_count = int(long.sum() + short.sum())
    severe_count = int(severe.sum())
    hard_invalid = (severe_count > 0 or anomaly_count > max_warning_anomalies
                    or rate_mismatch)
    return {"intervals_s": intervals, "long_intervals": int(long.sum()), "short_intervals": int(short.sum()),
            "estimated_missed_frames": int(missed), "min_interval_ms": float(intervals.min() * 1000),
            "median_interval_ms": float(np.median(intervals) * 1000),
            "max_interval_ms": float(intervals.max() * 1000),
            "anomaly_count": anomaly_count,
            "severe_interval_count": severe_count,
            "nonmonotonic_interval_count": int(nonmonotonic.sum()),
            "severe_interval_threshold_ms": float(2 * period * 1000),
            "observed_flip_rate_hz": observed_rate,
            "flip_rate_error_percent": None if rate_error is None else float(rate_error * 100),
            "flip_rate_tolerance_percent": float(rate_tolerance * 100),
            "flip_rate_mismatch": rate_mismatch,
            "valid": not hard_invalid,
            "strict_valid": anomaly_count == 0 and severe_count == 0 and not rate_mismatch,
            "warning": anomaly_count > 0 and not hard_invalid,
            "hard_invalid": hard_invalid,
            "refresh_hz": refresh_hz}

def make_schedule(blocks: int, seed: int) -> list[tuple[int, int]]:
    """按随机种子为每个区组生成一次覆盖全部目标的顺序。"""
    if not isinstance(blocks, int) or isinstance(blocks, bool) or blocks < 1:
        raise ValueError("blocks必须是正整数")
    rng = np.random.default_rng(seed)
    return [(b + 1, int(class_id)) for b in range(blocks)
            for class_id in rng.permutation(np.arange(1, len(TARGETS) + 1))]

def make_luminance_table(refresh_hz: float, duration_s: float) -> np.ndarray:
    """预计算所有目标在每个显示帧上的正弦亮度值。"""
    if not np.isfinite(refresh_hz) or refresh_hz <= 2 * BENCHMARK_FREQUENCIES_HZ.max():
        raise ValueError("刷新率无法支持当前最高刺激频率")
    if not np.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("刺激时长必须为正")
    frames = max(1, math.ceil(duration_s * refresh_hz) + 1)
    t = np.arange(frames)[:, None] / refresh_hz
    phases = np.array([target.phase_rad for target in TARGETS])[None, :]
    return .5 * (1 + np.sin(2 * np.pi * t * BENCHMARK_FREQUENCIES_HZ[None, :] + phases))

def itr_bits_per_minute(p: float, seconds_per_selection: float, n: int = 40) -> float:
    """均匀先验/对称错误模型的ITR估计；低于机会水平记0，不能当成实测互信息。"""
    if not 0 <= p <= 1 or n < 2 or seconds_per_selection <= 0:
        raise ValueError("ITR参数不合法")
    if p <= 1 / n:
        return 0.0
    bits = math.log2(n)
    if p < 1:
        bits += p * math.log2(p) + (1 - p) * math.log2((1 - p) / (n - 1))
    return max(0.0, bits * 60 / seconds_per_selection)

def summarize_trials(records: Sequence[TrialRecord], planned: int) -> dict:
    """保留有效试次准确率；另计每个块/目标首尝试，失败与拒识不能靠重试抹去。"""
    if any(r.mode != "cued" or r.true_class is None for r in records):
        raise ValueError("自由输入没有真实标签，不可计算提示测试准确率")
    attempted = len(records)
    valid = [r for r in records if r.status == "valid"]
    n = len(TARGETS)
    counts, valid_counts, invalid_counts = np.zeros(n, int), np.zeros(n, int), np.zeros(n, int)
    fb_cm = np.zeros((n, n), int)
    cca_cm = np.zeros((n, n), int)
    online_cm = np.zeros((n, n), int)
    calibration_fallback_count = 0
    calibration_fallback_valid_count = 0
    online_decoder_counts: dict[str, int] = {}
    for record in records:
        t = record.true_class - 1
        counts[t] += 1
        if record.result is not None and record.result.get("calibration_fallback"):
            calibration_fallback_count += 1
            if record.status == "valid":
                calibration_fallback_valid_count += 1
        if record.status == "valid":
            if record.result is None:
                raise ValueError("有效试次缺失结果")
            valid_counts[t] += 1
            fb_cm[t, record.result["fbcca_prediction"] - 1] += 1
            cca_cm[t, record.result["cca_prediction"] - 1] += 1
            online_cm[t, record.result["prediction"] - 1] += 1
            decoder = record.result["decoder"]
            online_decoder_counts[decoder] = online_decoder_counts.get(decoder, 0) + 1
        else:
            invalid_counts[t] += 1
    active_s = sum(max(0.0, r.end - r.start) for r in records)
    wall_s = max((r.end for r in records), default=0) - min((r.start for r in records), default=0)
    started_targets = {(r.block_id, r.true_class) for r in records}
    completed_targets = {(r.block_id, r.true_class) for r in valid}
    first_records: dict[tuple[int, int], TrialRecord] = {}
    for record in records:
        key = (record.block_id, record.true_class)
        previous = first_records.get(key)
        if previous is None or record.trial_id < previous.trial_id:
            first_records[key] = record
    first_started = len(first_records)
    first_attempt = {
        "selection_rule": "lowest actual trial_id per (block_id, true_class)",
        "started_targets": first_started,
        "planned_targets": planned,
        "coverage_rate": first_started / planned if planned > 0 else None,
        "all_planned_targets_started": planned > 0 and first_started == planned,
        "retry_attempts": attempted - first_started,
        "trial_ids": sorted(r.trial_id for r in first_records.values()),
        "nonvalid_first_attempts": sum(r.status != "valid" for r in first_records.values()),
    }
    for name, prediction_key in (("online", "prediction"),
                                 ("fbcca", "fbcca_prediction"), ("cca", "cca_prediction")):
        first_correct = sum(
            r.status == "valid" and r.result is not None
            and r.result[prediction_key] == r.true_class
            for r in first_records.values())
        first_attempt[name] = {
            "correct": int(first_correct),
            "accuracy_started": first_correct / first_started if first_started else None,
            "success_planned": first_correct / planned if planned > 0 else None,
        }
    report = {"planned": planned, "attempted": attempted,
        "not_started": max(0, planned - len(started_targets)),
        "completed_targets": len(completed_targets),
        "valid": len(valid), "rejected": sum(r.status == "rejected" for r in records),
        "invalid_or_aborted": sum(r.status in ("invalid", "aborted") for r in records),
        "active_selection_s": active_s, "wall_s_including_breaks": max(0, wall_s),
        "mean_actual_selection_s": active_s / attempted if attempted else None,
        "target_attempts": counts, "target_valid": valid_counts, "target_invalid": invalid_counts,
        "fbcca_confusion": fb_cm, "cca_confusion": cca_cm,
        "online_confusion": online_cm,
        "online_decoder_counts": online_decoder_counts,
        "calibration_fallback_count": calibration_fallback_count,
        "calibration_fallback_valid_count": calibration_fallback_valid_count,
        "first_attempt": first_attempt}
    for name, cm in (("fbcca", fb_cm), ("cca", cca_cm), ("online", online_cm)):
        correct = int(np.trace(cm))
        success_all = correct / attempted if attempted else None
        report[name] = {"correct": correct, "accuracy_valid": correct / len(valid) if valid else None,
            "success_all_attempted": success_all,
            "itr_conservative_actual_bpm": itr_bits_per_minute(success_all, active_s / attempted, n)
                if attempted and active_s > 0 else None,
            "itr_conservative_wall_bpm": itr_bits_per_minute(success_all, wall_s / attempted, n)
                if attempted and wall_s > 0 else None}
    return report

def _pct(value: Optional[float]) -> str:
    """将可选比例格式化为百分比或 ``N/A``。"""
    return "N/A" if value is None else f"{100 * value:.2f}%"

def _brief_reason(reason: str, limit: int = 160) -> str:
    """压缩空白并截短过长的单行原因文本。"""
    message = " ".join(reason.split())
    return message if len(message) <= limit else message[:limit - 1] + "…"

def print_summary(report: dict, ledger: TrialLedger) -> None:
    """输出提示测试的首尝试覆盖率和各解码器准确率摘要。"""
    first = report["first_attempt"]
    coverage_note = ("已覆盖全部计划目标" if first["all_planned_targets_started"]
                     else "尚未覆盖全部计划目标；不能作为完整一场验收")
    print(f"首尝试覆盖 {first['started_targets']}/{first['planned_targets']} "
          f"({_pct(first['coverage_rate'])}) | 重试 {first['retry_attempts']} 次 | {coverage_note}")
    for name, label in (("online", "实际在线"), ("fbcca", "基础FBCCA"), ("cca", "标准CCA")):
        metric = first[name]
        print(f"首尝试{label}：已开始目标 {metric['correct']}/{first['started_targets']} "
              f"({_pct(metric['accuracy_started'])})；按全部计划 "
              f"{metric['correct']}/{first['planned_targets']} "
              f"({_pct(metric['success_planned'])})；无效/中止/拒识均计失败")
    print(f"取得有效结果的目标{report['completed_targets']} | 有效试次{report['valid']} | "
          f"基础FBCCA有效试次准确率 {_pct(report['fbcca']['accuracy_valid'])} | "
          f"标准CCA有效试次准确率 {_pct(report['cca']['accuracy_valid'])} | "
          f"实际在线算法有效试次准确率 {_pct(report['online']['accuracy_valid'])} | "
          f"个人模型回退{report['calibration_fallback_count']}次"
          f"（其中有效试次{report['calibration_fallback_valid_count']}次）")

def summarize_free_trials(records: Sequence[TrialRecord], ledger: TrialLedger) -> dict:
    """自由输入只报告操作记录；不虚构正确率、混淆矩阵或ITR。"""
    if any(r.mode != "free" or r.true_class is not None for r in records):
        raise ValueError("自由输入记录不应含提示目标")
    return {
        "mode": "free", "attempted": len(records),
        "completed": sum(r.status != "aborted" for r in records),
        "valid": sum(r.status == "valid" for r in records),
        "input_actions": sum(r.text_applied for r in records),
        "rejected": sum(r.status == "rejected" for r in records),
        "invalid_or_aborted": sum(r.status in ("invalid", "aborted") for r in records),
        "typed_text": ledger.typed_text,
        "ground_truth_available": False, "accuracy": None, "itr_bits_per_minute": None,
    }

def print_free_summary(report: dict) -> None:
    """输出不含虚构准确率的自由输入会话摘要。"""
    print(f"已完成{report['completed']}轮 | 有效{report['valid']}轮 | "
          "准确率 N/A（自由输入无真实目标）")

def output_display_text(text: str, max_chars: int) -> str:
    """仅裁剪显示尾部，不删减真正的输入字符串；静态|标记插入位置。"""
    if max_chars < 12:
        raise ValueError("输出显示区域过短")
    tail_capacity = max_chars - len("OUTPUT: ") - 1
    visible = text if len(text) <= tail_capacity else "..." + text[-(tail_capacity - 3):]
    return "OUTPUT: " + visible + "|"

def _canonical_unit(value: str) -> str:
    """将常见电压单位写法规范为 ``uV``、``mV`` 或 ``V``。"""
    unit = str(value).strip().replace("μ", "u").replace("µ", "u").lower()
    return {"uv": "uV", "microvolt": "uV", "microvolts": "uV", "v": "V", "volt": "V", "volts": "V",
            "mv": "mV", "millivolt": "mV", "millivolts": "mV"}.get(unit, "UNKNOWN")

def preflight_transport(cfg: Config, collector: ContinuousLSL) -> dict[str, Any]:
    """自动确认最近的LSL EEG可用于试次；电极质量由OpenBCI GUI检查。"""
    collector.check_health()
    if collector.buffer is None:
        raise RuntimeFault("ACQUISITION", "LSL EEG缓冲区尚未建立")
    fs = float(collector.fs)
    try:
        _, _, local, _, _ = collector.buffer.snapshot()
    except InvalidTrial as exc:
        raise RuntimeFault("ACQUISITION", f"启动LSL缓冲读取失败：{exc}") from exc
    if len(local) < 2 or not np.isfinite(local).all():
        raise RuntimeFault("ACQUISITION", "启动LSL尚未收到足够的有效EEG时间戳")
    if np.any(np.diff(local) <= 0):
        raise RuntimeFault("ACQUISITION", "启动LSL时间戳回退或重复")
    available_s = float(local[-1] - local[0])
    if cfg.notch_mode == "history":
        available_s -= cfg.notch_history_s + 0.1
    duration = math.floor(min(5.0, available_s) * fs) / fs
    end = float(local[-1])
    start = end - duration
    if start < local[0]:
        duration -= 1 / fs
        start = end - duration
    if duration < 1.0:
        raise RuntimeFault("ACQUISITION", "最近LSL EEG未覆盖至少1秒连续数据")
    try:
        epoch = collector.buffer.epoch(
            start, duration, fs, cfg.max_gap_factor, cfg.rate_tolerance,
            cfg.hard_gap_factor, cfg.hard_rate_tolerance, cfg.max_warning_gaps,
            **({"notch_history_s": cfg.notch_history_s} if cfg.notch_mode == "history" else {}))
    except (InvalidTrial, ValueError) as exc:
        raise RuntimeFault("ACQUISITION", f"启动LSL数据不连续或包含无效样本：{exc}") from exc
    collector.check_health()
    age = float(collector.clock() - local[-1])
    if not math.isfinite(age) or abs(age) > cfg.max_receive_age_s:
        raise RuntimeFault("ACQUISITION", f"最新LSL EEG时间戳不新鲜或时钟异常：age={age:.3f}秒")

    meta = collector.metadata
    units = ([_canonical_unit(item["unit"]) for item in meta["selected_channels"]]
             if cfg.input_unit == "AUTO" else
             [_canonical_unit(cfg.input_unit)] * len(cfg.channel_indices))
    unverified = ["电极质量及物理接线由OpenBCI GUI/操作者检查，本程序未自动验证",
                  "TimeSeriesRaw及GUI滤波设置来自用户配置，LSL元数据未独立核实"]
    if "UNKNOWN" in units:
        unverified.append("LSL通道单位未全部声明；不推断物理幅度")
    startup_mains = mains_noise_diagnostics(epoch.data, fs, cfg.notch_hz)
    configuration_notes = []
    if meta["channel_count"] == 16 and max(cfg.channel_indices) < 8:
        configuration_notes.append(
            "当前LSL声明16路、程序只取前8路。若使用Cyton+Daisy，GUI关闭9–16路不等于"
            "切换到8通道采集模式；请核对GUI设备模式、原始记录采样率与LSL声明。"
            "仅凭LSL元数据不能判断实际板型或采样率是否错误。")
    review = {
        "review_method": "automatic_lsl_transport",
        "passed": True,
        "expected_openbci_output": "TimeSeriesRaw",
        "source_output_verified_from_lsl": False,
        "stream_name": meta["name"],
        "stream_type": meta["type"],
        "channel_count": meta["channel_count"],
        "selected_channel_indices": list(cfg.channel_indices),
        "selected_channel_labels": list(cfg.channel_labels),
        "electrode_positions_assumed": False,
        "input_units": units,
        "unit_override": cfg.input_unit,
        "upstream_filter_description": cfg.upstream_filter_description,
        "upstream_lowpass_hz": cfg.upstream_lowpass_hz,
        "receiver_notch_hz": cfg.notch_hz,
        "receiver_notch_mode": cfg.notch_mode,
        "receiver_notch_history_s": cfg.notch_history_s if cfg.notch_mode == "history" else 0.0,
        "wiring_user_reviewed": False,
        "electrode_quality_checked_by_app": False,
        "startup_transport": {
            "window_s": duration,
            "uniform_samples": int(epoch.data.shape[1]),
            "nominal_fs_hz": fs,
            "observed_fs_hz": float(collector.estimated_fs),
            "recent_fs_hz": epoch.diagnostics["timestamp_estimated_fs"],
            "latest_sample_age_s": age,
            "timestamp_warnings": list(epoch.diagnostics.get("warnings", [])),
            "warning_gap_count": epoch.diagnostics["warning_gap_count"],
        },
        "startup_mains_noise": startup_mains,
        "configuration_notes": configuration_notes,
        "unverified": unverified,
        "device_timestamp_lag_applied_s": cfg.device_timestamp_lag_s,
    }
    print(f"LSL传输检查通过：{meta['name']} / {meta['type']}；"
          f"使用{len(cfg.channel_indices)}路EEG（{'、'.join(cfg.channel_labels)}）；"
          f"声明{fs:g}Hz，观测{collector.estimated_fs:.3f}Hz；"
          f"最近{duration:.2f}秒通过连续性检查，最新样本{age:.3f}秒前。", flush=True)
    for warning in epoch.diagnostics.get("warnings", []):
        if not warning.startswith("LSL时间戳已同步/去抖"):
            print("LSL时间戳提示：" + warning, flush=True)
    if startup_mains["available"]:
        low, high = startup_mains["line_band_hz"]
        ref_low, ref_high = startup_mains["reference_band_hz"]
        print(f"[采集质量] 滤波前{low:g}–{high:g}Hz / {ref_low:g}–{ref_high:g}Hz能量占比，"
              f"通道中位数={startup_mains['median_line_fraction']:.1%}。", flush=True)
        values = startup_mains["line_fraction_per_channel"]
        print("[各通道] " + "；".join(
            f"{label}=" + (f"{value:.1%}" if value is not None else "不可用")
            for label, value in zip(cfg.channel_labels, values)),
            flush=True)
        if startup_mains["line_dominated"]:
            print("[采集质量提示] 工频附近能量占比高，建议先检查参考/BIAS、电极接触与周围电源。"
                  "此占比不是识别准确率，也不能单独确定干扰来源。", flush=True)
    for note in configuration_notes:
        print("[设备配置提示] " + note, flush=True)
    print("电极质量与实际接线沿用OpenBCI GUI检查；程序不会据此声称自动验证。", flush=True)
    return review



__all__ = [name for name in globals() if not name.startswith('__')]

