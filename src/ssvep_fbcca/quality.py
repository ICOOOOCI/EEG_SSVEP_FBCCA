from __future__ import annotations

from .common import *
from .config import *
from .signal import *
from .events import _json_safe



def spectral_quality(data: np.ndarray, fs: float, target_id: Optional[int],
                     n_harmonics: int) -> dict[str, Any]:
    """退出时计算频谱诊断，不参与在线门控；频率分辨率由真实窗长决定。"""
    x = np.asarray(data, dtype=float)
    if (x.ndim != 2 or x.shape[0] == 0 or x.shape[1] < 8
            or not np.isfinite(x).all() or not np.isfinite(fs) or fs <= 0):
        return {"available": False, "reason": "No finite uniform EEG window"}
    from scipy.signal import periodogram
    frequencies, psd = periodogram(x, fs=fs, window="hann", detrend="constant",
                                  scaling="density", axis=-1)
    harmonics = []
    if target_id is not None:
        fundamental = TARGETS[target_id - 1].frequency_hz
        for order in range(1, n_harmonics + 1):
            frequency = fundamental * order
            if frequency >= fs / 2:
                break
            index = int(np.argmin(np.abs(frequencies - frequency)))
            neighbors = np.asarray([index + offset for offset in (-4, -3, -2, 2, 3, 4)])
            neighbors = neighbors[(neighbors > 0) & (neighbors < len(frequencies) - 1)]
            noise = (psd[:, neighbors].mean(axis=1) if neighbors.size
                     else np.full(x.shape[0], np.nan))
            signal = psd[:, index]
            snr_db = np.full(x.shape[0], np.nan)
            valid = (signal > 0) & (noise > 0) & np.isfinite(noise)
            snr_db[valid] = 10 * (np.log10(signal[valid]) - np.log10(noise[valid]))
            harmonics.append({"order": order, "frequency_hz": frequency,
                              "bin_frequency_hz": float(frequencies[index]),
                              "psd_per_channel": signal, "snr_db_per_channel": snr_db})
    return _json_safe({
        "available": True, "sampling_rate_hz": fs,
        "method": "Hann periodogram; constant detrend; no zero padding or receiver filtering",
        "psd_units": "input_unit_squared_per_hz; physical unit requires acquisition metadata",
        "snr_method": "10*log10(nearest-bin PSD / mean PSD at bin offsets -4,-3,-2,2,3,4)",
        "interpretation": "Descriptive local spectral contrast, not calibrated detection confidence",
        "frequency_resolution_hz": float(frequencies[1] - frequencies[0]),
        "frequencies_hz": frequencies, "psd_per_channel": psd,
        "reference_class": target_id, "harmonics": harmonics,
    })

def _finite_median(values: Sequence[Any]) -> Optional[float]:
    """返回有限数值的中位数；无可用值时返回 ``None``。"""
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.median(finite)) if finite else None

def _cued_quality_report(session: dict[str, Any], npz_sha256: str) -> dict[str, Any]:
    """汇总提示测试的逐目标频谱、呈现和信号质量指标。"""
    trials = session["trials"]
    summaries = []
    for target in TARGETS:
        attempts = [t for t in trials if t["true_class"] == target.class_id]
        valid = [t for t in attempts if t["status"] == "valid"]
        harmonics = []
        for order in range(1, session["decoder_settings"]["n_harmonics"] + 1):
            trial_medians = []
            for trial in valid:
                spectral = trial.get("spectral_quality") or {}
                for harmonic in spectral.get("harmonics", []):
                    if harmonic["order"] == order:
                        trial_medians.append(_finite_median(harmonic["snr_db_per_channel"]))
                        break
            harmonics.append({"order": order,
                              "frequency_hz": target.frequency_hz * order,
                              "median_snr_db": _finite_median(trial_medians)})
        summaries.append({"class_id": target.class_id, "symbol": target.symbol,
                          "frequency_hz": target.frequency_hz,
                          "attempts": len(attempts), "valid": len(valid),
                          "correct_online": sum(t["prediction"] == target.class_id for t in valid),
                          "harmonics": harmonics})
    frame = [t.get("frame_diagnostics") or {} for t in trials]
    trial_qc = []
    for trial in trials:
        quality = trial.get("quality") or {}
        timing = trial.get("frame_diagnostics") or {}
        trial_qc.append({
            "trial_id": trial["trial_id"], "block_id": trial["block_id"],
            "class_id": trial["true_class"], "status": trial["status"],
            "reason": trial.get("reason"), "prediction": trial.get("prediction"),
            "signal_unit": quality.get("unit"),
            "signal_hard_fail": quality.get("hard_fail"),
            "signal_hard_reasons": quality.get("hard_reasons", []),
            "signal_warnings": quality.get("warnings", []),
            "valid_channels": quality.get("valid_channels"),
            "frame_hard_invalid": timing.get("hard_invalid"),
            "frame_anomalies": timing.get("anomaly_count"),
            "estimated_missed_frames": timing.get("estimated_missed_frames"),
        })
    return {
        "schema_version": 1,
        "session_id": session["session_id"],
        "source_npz": session["session_id"] + ".npz",
        "source_npz_sha256": npz_sha256,
        "mode": "cued",
        "target_count": len(TARGETS),
        "labels": [target.symbol for target in TARGETS],
        "freqs_hz": [target.frequency_hz for target in TARGETS],
        "channel_indices": session["config"]["channel_indices"],
        "channel_positions": session["config"]["channel_positions"],
        "channel_labels": [f"CH{index + 1}" for index in session["config"]["channel_indices"]],
        "electrode_positions_assumed": False,
        "online_decoder": session["algorithm"],
        "decoder_settings": session["decoder_settings"],
        "online_summary": session["summary"],
        "presentation_qc": {
            "trial_count_with_frame_diagnostics": sum(bool(f) for f in frame),
            "median_refresh_hz": _finite_median([f.get("refresh_hz") for f in frame]),
            "total_anomalous_intervals": sum(int(f.get("anomaly_count") or 0) for f in frame),
            "total_severe_intervals": sum(int(f.get("severe_interval_count") or 0) for f in frame),
        },
        "signal_qc": {
            "method": "各有效试次Hann周期图的局部PSD比；先跨通道、再跨试次取中位数",
            "scope": "描述性频谱对比，不是识别概率或独立验证准确率",
            "frequency_resolution_hz": (next((t["spectral_quality"].get("frequency_resolution_hz")
                                             for t in trials if t.get("spectral_quality", {}).get("available")), None)),
            "targets": summaries,
        },
        "trial_qc": trial_qc,
    }



__all__ = [name for name in globals() if not name.startswith('__')]

