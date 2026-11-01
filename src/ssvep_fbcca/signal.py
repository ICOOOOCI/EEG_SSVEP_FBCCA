from __future__ import annotations

from .common import *
from .config import *



def _validate_eeg_ct(data: np.ndarray, *, name: str = "EEG",
                     allow_empty_samples: bool = False) -> np.ndarray:
    """按内部统一的“通道 × 采样点”布局验证并返回 EEG 矩阵。"""
    x = np.asarray(data, dtype=float)
    if x.ndim != 2 or x.shape[0] < 1 or (
            not allow_empty_samples and x.shape[1] < 1):
        raise ValueError(f"{name}必须是二维C×T数组（通道数×采样点数）")
    return x

def sample_count(duration_s: float, fs: float) -> int:
    """[start, end)内的均匀采样点数；1.25s×250Hz需要313点而非静默截掉半点。

    最后一采样点必须严格小于请求终点；浮点容差只处理整数乘积舍入误差。
    """
    if not np.isfinite(duration_s) or duration_s <= 0 or not np.isfinite(fs) or fs <= 0:
        raise ValueError("时长和采样率必须为有限正数")
    return int(math.ceil(duration_s * fs - 1e-9))

def generate_reference_signals(frequencies_hz: Sequence[float], fs: float,
                               n_samples: int, n_harmonics: int = 5) -> np.ndarray:
    """原参考信号公式；返回(targets, 2*harmonics, samples)。"""
    frequencies = np.asarray(frequencies_hz, dtype=float)
    if (frequencies.ndim != 1 or not frequencies.size or not np.isfinite(frequencies).all()
            or np.any(frequencies <= 0) or len(np.unique(frequencies)) != len(frequencies)):
        raise ValueError("候选频率须为一维、不重复的有限正数")
    if isinstance(n_samples, bool) or not isinstance(n_samples, (int, np.integer)) or n_samples < 2:
        raise ValueError("n_samples 必须是大于等于2的整数")
    _validate_harmonics(n_harmonics, frequencies, fs)
    t = np.arange(n_samples, dtype=float) / fs
    harmonic = np.arange(1, n_harmonics + 1, dtype=float)
    phase = 2 * np.pi * frequencies[:, None, None] * harmonic[None, :, None] * t[None, None, :]
    return np.stack((np.sin(phase), np.cos(phase)), axis=2).reshape(
        len(frequencies), 2 * n_harmonics, n_samples)

def _normalised_rows(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """先按每行幅值缩放，再中心化/标准化，避免微伏/伏尺度及平方下溢。

    阈值仅用于“相对于该行原始幅值”的数值可分辨性，不是固定伏值下限。
    返回非恒值行及它们的索引；全零/全恒值输入明确报错。
    """
    x = _validate_eeg_ct(data, name="输入")
    if x.shape[1] < 2 or not np.isfinite(x).all():
        raise ValueError("输入须为有限二维数组(channels, samples)，至少2个样本")
    amplitude = np.max(np.abs(x), axis=1, keepdims=True)
    scaled = np.divide(x, amplitude, out=np.zeros_like(x), where=amplitude > 0)
    centered = scaled - scaled.mean(axis=1, keepdims=True)
    rms = np.sqrt(np.mean(centered * centered, axis=1))
    active = np.flatnonzero(rms > 64 * np.finfo(float).eps)
    if not len(active):
        raise DegenerateSignalError("全零或全部恒值通道，拒绝输出任意最高分类")
    return centered[active] / rms[active, None], active

def assess_signal_quality(data: np.ndarray, *, unit: str = "UNKNOWN",
                           max_abs: Optional[float] = None,
                           rail_min: Optional[float] = None,
                           rail_max: Optional[float] = None,
                           rail_margin: float = 0.0,
                           max_clip_fraction: float = 0.0,
                           jump_z: float = 12.0,
                          min_valid_channels: int = 6,
                          max_warning_bad_channels: int = 2) -> dict[str, Any]:
    """检查标准化前的原始 EEG，不改变 FBCCA 的输入和公式。

    绝对幅值和削顶只有在设备单位/轨值明确时才有物理意义；未知单位只产生
    警告，避免把任意 LSL 数值误当成微伏或伏特。返回的摘要可直接写入试次日志。
    rail_margin 使用与输入/轨值相同的数值单位；max_clip_fraction 是[0, 1)样本比例。
    """
    if not math.isfinite(rail_margin) or rail_margin < 0:
        raise ValueError("rail_margin 必须是非负有限的EEG数值余量")
    if not math.isfinite(max_clip_fraction) or not 0 <= max_clip_fraction < 1:
        raise ValueError("max_clip_fraction 必须是[0, 1)内的有限样本比例")
    try:
        x = _validate_eeg_ct(data, name="原始EEG")
    except ValueError as exc:
        raise InvalidTrial("原始EEG形状不足，无法进行质量检查") from exc
    if min(x.shape, default=0) < 2:
        raise InvalidTrial("原始EEG形状不足，无法进行质量检查")
    report: dict[str, Any] = {
        "unit": str(unit), "hard_fail": False, "hard_reasons": [], "warnings": [],
        "valid_channels": [], "bad_channels": [], "max_abs": [], "peak_to_peak": [],
        "clip_fraction": [], "jump_z": [],
    }
    if not np.isfinite(x).all():
        report["hard_fail"] = True
        report["hard_reasons"].append("EEG含NaN/Inf")
        return report

    centered = x - np.mean(x, axis=1, keepdims=True)
    rms = np.sqrt(np.mean(centered * centered, axis=1))
    scale = np.maximum(np.max(np.abs(x), axis=1), np.finfo(float).eps)
    active = rms > 64 * np.finfo(float).eps * scale
    report["valid_channels"] = np.flatnonzero(active).astype(int).tolist()
    report["bad_channels"] = np.flatnonzero(~active).astype(int).tolist()
    if len(report["bad_channels"]):
        report["warnings"].append(
            "恒值/无变化通道: " + ",".join(str(i + 1) for i in report["bad_channels"]))
    if len(report["bad_channels"]) > max_warning_bad_channels or len(report["valid_channels"]) < min_valid_channels:
        report["hard_fail"] = True
        report["hard_reasons"].append(
            f"有效通道仅{len(report['valid_channels'])}个，至少需要{min_valid_channels}个")

    max_abs_values = np.max(np.abs(x), axis=1)
    p2p_values = np.ptp(x, axis=1)
    report["max_abs"] = max_abs_values.tolist()
    report["peak_to_peak"] = p2p_values.tolist()
    unit_known = str(unit).upper() not in ("AUTO", "UNKNOWN", "UNVERIFIED", "")
    if max_abs is not None and unit_known:
        if not np.isfinite(max_abs) or max_abs <= 0:
            raise ValueError("max_abs 必须为有限正数")
        too_large = np.flatnonzero(max_abs_values > max_abs)
        if len(too_large):
            report["hard_fail"] = True
            report["hard_reasons"].append(
                "幅值超过阈值的通道: " + ",".join(str(i + 1) for i in too_large))

    clip_fraction = np.zeros(x.shape[0], dtype=float)
    if not unit_known:
        report["warnings"].append("单位未确认，绝对幅值/削顶仅记录未门控")
    elif rail_min is not None or rail_max is not None:
        if (rail_min is None or rail_max is None or not rail_min < rail_max
                or not np.isfinite(rail_min) or not np.isfinite(rail_max)):
            raise ValueError("rail_min/rail_max 必须同时为有限值且 min < max")
        if 2 * rail_margin >= rail_max - rail_min:
            raise ValueError("rail_margin 必须小于设备轨值跨度的一半")
        clipped = (x <= rail_min + rail_margin) | (x >= rail_max - rail_margin)
        clip_fraction = clipped.mean(axis=1)
        if np.any(clip_fraction > max_clip_fraction):
            report["hard_fail"] = True
            report["hard_reasons"].append(
                "检测到设备轨值削顶通道: " + ",".join(
                    str(i + 1) for i in np.flatnonzero(clip_fraction > max_clip_fraction)))
    elif max_abs is None:
        report["warnings"].append("单位或设备满量程阈值未配置，绝对幅值/削顶仅记录未门控")
    report["clip_fraction"] = clip_fraction.tolist()

    jump_values = np.zeros(x.shape[0], dtype=float)
    for i, row in enumerate(x):
        diff = np.diff(row)
        med = float(np.median(diff))
        mad = float(np.median(np.abs(diff - med)))
        denom = max(1.4826 * mad, np.finfo(float).eps)
        jump_values[i] = float(np.max(np.abs(diff - med)) / denom) if len(diff) else 0.0
    report["jump_z"] = jump_values.tolist()
    jump_channels = np.flatnonzero(jump_values > max(0.0, float(jump_z)))
    if len(jump_channels):
        report["warnings"].append(
            "存在异常相邻跳变通道: " + ",".join(str(i + 1) for i in jump_channels))
    return report

def mains_noise_diagnostics(data: np.ndarray, fs: float,
                            notch_hz: Optional[float] = 50.0) -> dict[str, Any]:
    """描述接收端滤波前的窄带功率，但不据此拒绝 EEG。

    汉宁窗周期图的功率比与通道单位无关。通道中位比例不低于 50% 仅表示
    输入由工频成分主导，并非经过验证的 SSVEP 质量或拒识阈值，也不能据此
    判定物理噪声源。关闭接收端陷波时仍观察默认的 49--51 Hz 频带；分母频带
    上限为 90 Hz 与奈奎斯特频率中的较小值，且不含奈奎斯特频点。分母频带
    不同时，所得比例不可直接比较。
    """
    from scipy.signal import periodogram

    x = _validate_eeg_ct(data, name="工频诊断EEG")
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("工频诊断采样率必须为有限正数")
    center = 50.0 if notch_hz is None else float(notch_hz)
    if not math.isfinite(center) or center <= 0:
        raise ValueError("工频诊断中心频率必须为有限正数")
    reference_high = min(90.0, fs / 2)
    report: dict[str, Any] = {
        "available": False,
        "method": "Hann periodogram; per-channel line-band / reference-band power",
        "input_scope": "uniform EEG before receiver notch and bandpass filtering",
        "sampling_rate_hz": float(fs),
        "receiver_notch_hz": notch_hz,
        "line_center_hz": center,
        "line_center_source": "default_50_hz_notch_off" if notch_hz is None else "receiver_notch_setting",
        "line_band_hz": [center - 1.0, center + 1.0],
        "reference_band_hz": [6.0, float(reference_high)],
        "nyquist_bin_excluded": True,
        "frequency_resolution_hz": float(fs / x.shape[1]),
        "line_fraction_per_channel": [None] * x.shape[0],
        "median_line_fraction": None,
        "valid_channel_indices": [],
        "line_dominated": False,
        "description_threshold": 0.5,
        "rejection_threshold_validated": False,
        "used_for_rejection": False,
        "warnings": [],
    }
    if x.shape[1] < 32 or not np.isfinite(x).all():
        report["unavailable_reason"] = "insufficient or non-finite samples"
        return report
    if (center - 1.0 < 6.0 or center + 1.0 > reference_high
            or center + 1.0 >= fs / 2):
        report["unavailable_reason"] = "full line band not covered below Nyquist / within the reference band"
        return report
    try:
        normalized, active = _normalised_rows(x)
    except DegenerateSignalError:
        report["unavailable_reason"] = "no varying channels"
        return report
    frequencies, psd = periodogram(normalized, fs=fs, window="hann",
                                  detrend="constant", axis=-1)
    line = (frequencies >= center - 1.0) & (frequencies <= center + 1.0)
    reference = ((frequencies >= 6.0) & (frequencies <= reference_high)
                 & (frequencies < fs / 2))
    if not np.any(line) or not np.any(reference):
        report["unavailable_reason"] = "no frequency bins in the diagnostic bands"
        return report
    denominator = psd[:, reference].sum(axis=-1)
    usable = denominator > np.finfo(float).tiny
    if not np.any(usable):
        report["unavailable_reason"] = "no measurable power in the reference band"
        return report
    fractions = np.clip(psd[usable][:, line].sum(axis=-1) / denominator[usable], 0.0, 1.0)
    indices = active[usable]
    for index, fraction in zip(indices, fractions):
        report["line_fraction_per_channel"][int(index)] = float(fraction)
    median = float(np.median(fractions))
    report.update(available=True, valid_channel_indices=indices.tolist(),
                  median_line_fraction=median, line_dominated=median >= 0.5)
    if report["line_dominated"]:
        report["warnings"].append(
            f"工频附近能量占比高：{center - 1:g}–{center + 1:g} Hz / 6–{reference_high:g} Hz"
            f"通道中位数={median:.1%}；仅作相对频谱提示，不用于拒识或物理幅值判断")
    return report

def _inverse_sqrt_covariance(covariance: np.ndarray, regularization: float) -> np.ndarray:
    """相对特征值下限；不再用固定绝对eps截断微小协方差。"""
    values, vectors = np.linalg.eigh((covariance + covariance.T) / 2)
    scale = float(np.max(values))
    if not np.isfinite(scale) or scale <= 0:
        raise DegenerateSignalError("协方差不含有效正特征值")
    floor = scale * max(float(regularization), np.finfo(float).eps * len(values))
    return (vectors * (1 / np.sqrt(np.maximum(values, floor)))) @ vectors.T

def _whiten_rows(data: np.ndarray, regularization: float) -> np.ndarray:
    """对输入行进行标准化和协方差白化。"""
    if not np.isfinite(regularization) or regularization < 0:
        raise ValueError("regularization 必须非负且有限")
    z, _ = _normalised_rows(data)
    covariance = z @ z.T / (z.shape[1] - 1)
    scale = float(np.trace(covariance)) / len(covariance)
    covariance += regularization * scale * np.eye(len(covariance))
    return (_inverse_sqrt_covariance(covariance, regularization) @ z
            / math.sqrt(z.shape[1] - 1))

def cca_correlation(eeg: np.ndarray, reference: np.ndarray, regularization: float = 1e-10) -> float:
    """修正后的最大典型相关系数；单位缩放不改变结果。"""
    x, y = np.asarray(eeg, float), np.asarray(reference, float)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError("CCA输入必须二维且样本数相等")
    cross = _whiten_rows(x, regularization) @ _whiten_rows(y, regularization).T
    return float(np.clip(np.linalg.svd(cross, compute_uv=False)[0], 0, 1))

def fbcca_fusion(eeg_subbands: np.ndarray, reference_signals: np.ndarray,
                 a: float = 1.25, b: float = 0.25, regularization: float = 1e-10
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """保留 score=sum((k**(-a)+b)*rho**2)，复用白化参考以减少重复计算。"""
    bands, refs = np.asarray(eeg_subbands, float), np.asarray(reference_signals, float)
    if (bands.ndim != 3 or refs.ndim != 3 or bands.shape[-1] != refs.shape[-1]
            or min(bands.shape) < 1 or min(refs.shape) < 1):
        raise ValueError("FBCCA形状必须为(bands, channels, samples)/(targets, components, samples)")
    if not np.isfinite(a) or not np.isfinite(b) or a < 0 or b < 0:
        raise ValueError("权重参数a、b须非负且有限")
    wy = [_whiten_rows(ref, regularization) for ref in refs]
    return _fusion_whitened(bands, wy, a, b, regularization)

def _fusion_whitened(bands: np.ndarray, wy: Sequence[np.ndarray],
                     a: float, b: float, regularization: float
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """批量计算各候选的小矩阵SVD；保持原最大典型相关和融合公式。"""
    if not np.isfinite(a) or not np.isfinite(b) or a < 0 or b < 0:
        raise ValueError("权重参数a、b须非负且有限")
    same_shape = len({ref.shape for ref in wy}) == 1
    transposed = np.stack(wy).swapaxes(-1, -2) if same_shape else None
    correlations = np.empty((len(bands), len(wy)))
    for k, band in enumerate(bands):
        wx = _whiten_rows(band, regularization)
        if transposed is not None:
            correlations[k] = np.clip(
                np.linalg.svd(wx @ transposed, compute_uv=False)[:, 0], 0, 1)
        else:
            for j, ref_w in enumerate(wy):
                correlations[k, j] = np.clip(
                    np.linalg.svd(wx @ ref_w.T, compute_uv=False)[0], 0, 1)
    weights = np.arange(1, len(bands) + 1, dtype=float) ** (-a) + b
    return weights @ (correlations ** 2), correlations, weights

@lru_cache(maxsize=32)
def _cached_reference_whitening(frequencies: tuple[float, ...], fs: float,
                                n_samples: int, n_harmonics: int,
                                regularization: float) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    """只缓存确定性的数学参考，不缓存EEG、标签或试次结果。"""
    refs = generate_reference_signals(frequencies, fs, n_samples, n_harmonics)
    white = tuple(_whiten_rows(ref, regularization) for ref in refs)
    refs.setflags(write=False)
    for ref in white:
        ref.setflags(write=False)
    return refs, white

@lru_cache(maxsize=32)
def _cached_filter_design(fs: float, notch_hz: Optional[float],
                          bands: tuple[tuple[float, float], ...]) -> tuple:
    """创建并缓存只读的陷波器和 Chebyshev-I 滤波器组系数。"""
    from scipy.signal import cheby1, iirnotch
    notch = None if notch_hz is None else iirnotch(notch_hz, 30.0, fs=fs)
    bank = tuple(cheby1(4, .5, [low, high], btype="bandpass", fs=fs, output="sos")
                 for low, high in bands)
    for coefficient in (*(() if notch is None else notch), *bank):
        coefficient.setflags(write=False)
    return notch, bank

def _slice_eeg_window(data: np.ndarray, fs: float, window_s: float,
                      onset_s: float, *, short_message: str) -> np.ndarray:
    """在调用方完成布局和采样率检查后，截取一个完整 EEG 窗。"""
    if not np.isfinite(window_s) or window_s <= 0 or not np.isfinite(onset_s) or onset_s < 0:
        raise ValueError("window_s须为正、onset_s须非负")
    start, length = round(onset_s * fs), sample_count(window_s, fs)
    if length < 32 or start + length > data.shape[1]:
        raise ValueError(short_message)
    return data[:, start:start + length].copy()

def _preprocess_validated_window(window: np.ndarray, fs: float, window_s: float,
                                 target_fs: float, notch_hz: Optional[float],
                                 bands: tuple[tuple[float, float], ...]
                                 ) -> tuple[np.ndarray, np.ndarray]:
    """对已截取且验证过的窗口进行重采样、陷波和滤波器组处理。"""
    from scipy.signal import filtfilt, resample_poly, sosfiltfilt

    if not math.isclose(fs, target_fs, rel_tol=0, abs_tol=1e-9):
        ratio = Fraction(target_fs / fs).limit_denominator(100000)
        actual_fs = fs * ratio.numerator / ratio.denominator
        if not math.isclose(actual_fs, target_fs, rel_tol=1e-8):
            raise ValueError("无法得到准确的重采样比")
        window = resample_poly(window, ratio.numerator, ratio.denominator, axis=-1)
    wanted = sample_count(window_s, target_fs)
    if window.shape[1] < wanted:
        raise ValueError("重采样后样本不足，不补零")
    window = window[:, :wanted]
    if notch_hz is not None:
        if not np.isfinite(notch_hz) or not 0 < notch_hz < min(fs, target_fs) / 2:
            raise ValueError("陷波频率须位于0与Nyquist之间")
    notch, filters = _cached_filter_design(target_fs, notch_hz, bands)
    if notch is not None:
        bn, an = notch
        window = filtfilt(bn, an, window, axis=-1)
    bank = [sosfiltfilt(sos.copy(), window, axis=-1) for sos in filters]
    return window, np.stack(bank)

def preprocess_lsl_window(data_ch_samples: np.ndarray, fs: float,
                          window_s: float = DEFAULT_WINDOW_S, onset_s: float = 0.0,
                          target_fs: float = TARGET_FS, notch_hz: Optional[float] = 50.0,
                          *, filter_bands: Optional[Sequence[Sequence[float]]] = None
                          ) -> tuple[np.ndarray, np.ndarray]:
    """保留原 Chebyshev-I 阶数4/纹波0.5dB和filtfilt；不是连续因果滤波。

    输入为均匀时间网格。onset_s仅为数组裁剪偏移；事件截窗后必须传0。
    零相位滤波只使用已到齐的本窗口及SciPy默认反射边界，未引入后续试次。
    """
    bands = _validate_filter_bands(filter_bands, fs, target_fs)
    data = _validate_eeg_ct(data_ch_samples, name="LSL数据")
    if not np.isfinite(data).all():
        raise ValueError("LSL数据须为有限(channels, samples)数组")
    window = _slice_eeg_window(data, fs, window_s, onset_s,
                               short_message="数据不足或窗口过短；不得补零凑齐窗口")
    _normalised_rows(window)
    return _preprocess_validated_window(window, fs, window_s, target_fs, notch_hz, bands)

def classify_eeg_window(data_ch_samples: np.ndarray, fs: float,
                        window_s: float = DEFAULT_WINDOW_S, onset_s: float = 0.0,
                        target_frequencies_hz: Optional[Sequence[float]] = None,
                        n_harmonics: int = N_HARMONICS, *, target_fs: float = TARGET_FS,
                        notch_hz: Optional[float] = 50.0, a: float = 1.25, b: float = .25,
                        regularization: float = 1e-10, min_valid_channels: int = 1,
                        filter_bands: Optional[Sequence[Sequence[float]]] = None) -> dict:
    """只接收EEG/算法参数，不接受真实目标，防止标签泄漏。

    FBCCA和标准CCA共享完全相同的EEG窗、通道、参考和第一子带。
    分数不是概率；本方法仍强制选择最高分，没有空闲检测或自由输入可靠性保证。
    """
    frequencies = (BENCHMARK_FREQUENCIES_HZ if target_frequencies_hz is None
                   else np.asarray(target_frequencies_hz, float))
    bands = _validate_filter_bands(filter_bands, fs, target_fs)
    _validate_harmonics(n_harmonics, frequencies, fs)
    data = _validate_eeg_ct(data_ch_samples, name="EEG")
    if not np.isfinite(window_s) or window_s <= 0 or not np.isfinite(onset_s) or onset_s < 0:
        raise ValueError("EEG须二维，窗长为正，偏移非负")
    window = _slice_eeg_window(data, fs, window_s, onset_s,
                               short_message="EEG时间窗覆盖不足")
    notch_processing = {"mode": "epoch", "notch_hz": notch_hz, "Q": 30.0,
                        "order": "uniform raw window -> normalize -> epoch notch -> subbands"}
    normalized, active = _normalised_rows(window)
    if isinstance(min_valid_channels, bool) or not isinstance(min_valid_channels, int) or min_valid_channels < 1:
        raise ValueError("min_valid_channels须为正整数")
    if len(active) < min_valid_channels:
        raise DegenerateSignalError(f"有效变化通道仅{len(active)}个，至少需要{min_valid_channels}个")
    window, bank = _preprocess_validated_window(normalized, fs, window_s, target_fs, notch_hz, bands)
    refs, white_refs = _cached_reference_whitening(
        tuple(float(f) for f in frequencies), float(target_fs), window.shape[-1],
        n_harmonics, float(regularization))
    scores, rho, weights = _fusion_whitened(bank, white_refs, a, b, regularization)
    best = int(np.argmax(scores))
    cca_scores = rho[0] ** 2
    cca_best = int(np.argmax(cca_scores))
    return {
        "prediction": best + 1, "frequency_hz": float(frequencies[best]),
        "scores": scores, "correlations": rho, "weights": weights,
        "cca_prediction": cca_best + 1, "cca_frequency_hz": float(frequencies[cca_best]),
        "cca_scores": cca_scores, "active_channels": active,
        "dropped_channels": np.setdiff1d(np.arange(data.shape[0]), active),
        "window_shape": window.shape, "bank_shape": bank.shape, "reference_shape": refs.shape,
        "processing_fs": target_fs,
        "filter_bands_hz": bands,
        "n_harmonics": n_harmonics,
        "notch_processing": notch_processing,
        "decoder": "FBCCA",
        "fbcca_scores": scores,
        "fbcca_prediction": best + 1,
    }



__all__ = [name for name in globals() if not name.startswith('__')]

