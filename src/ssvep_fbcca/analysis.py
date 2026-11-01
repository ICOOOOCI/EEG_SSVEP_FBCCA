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
    tail_capacity = max_chars - 1
    visible = text if len(text) <= tail_capacity else "..." + text[-(tail_capacity - 3):]
    return visible + "|"

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
            cfg.hard_gap_factor, cfg.hard_rate_tolerance, cfg.max_warning_gaps)
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

