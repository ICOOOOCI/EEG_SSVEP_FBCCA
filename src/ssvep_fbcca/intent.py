"""自由输入的意图证据特征。

该模块不参与 FBCCA 分类，只从已经到齐的 EEG 窗计算拒识门控所需的
描述性频谱比例。比例不依赖输入的绝对物理单位，因此不会把未知 LSL
单位误当成微伏或伏特。
"""

from __future__ import annotations

from .common import *


def alpha_power_ratio(data: np.ndarray, fs: float,
                      *, alpha_band: tuple[float, float] = (8.0, 13.0),
                      reference_band: tuple[float, float] = (6.0, 45.0)) -> float:
    """返回所有通道中位数的 alpha/reference PSD 比例。

    仅用于自由输入的意图门控，不改变 FBCCA 的输入、滤波器组或参数。
    使用原始均匀 EEG 窗的 Hann 周期图；通道先各自计算比例，再取中位数，
    避免单个高幅值通道主导结果。
    """
    x = np.asarray(data, dtype=float)
    if (x.ndim != 2 or x.shape[0] < 1 or x.shape[1] < 8
            or not np.isfinite(x).all() or not np.isfinite(fs) or fs <= 0):
        raise ValueError("意图门控 EEG 必须是有限的通道×采样点数组")
    lo_a, hi_a = map(float, alpha_band)
    lo_r, hi_r = map(float, reference_band)
    if not (0 < lo_r < lo_a < hi_a <= hi_r < fs / 2):
        raise ValueError("alpha/reference 频带或采样率不合法")
    from scipy.signal import periodogram
    frequencies, psd = periodogram(
        x, fs=fs, window="hann", detrend="constant", scaling="density", axis=-1)
    alpha = (frequencies >= lo_a) & (frequencies < hi_a)
    reference = (frequencies >= lo_r) & (frequencies < hi_r)
    if not np.any(alpha) or not np.any(reference):
        raise ValueError("意图门控频带没有可用的频谱点")
    reference_power = np.mean(psd[:, reference], axis=1)
    alpha_power = np.mean(psd[:, alpha], axis=1)
    valid = np.isfinite(reference_power) & np.isfinite(alpha_power) & (reference_power > 0)
    if not np.any(valid):
        raise DegenerateSignalError("意图门控参考频带没有有限功率")
    ratios = alpha_power[valid] / reference_power[valid]
    if not np.isfinite(ratios).all():
        raise DegenerateSignalError("意图门控 alpha 比例不是有限值")
    return float(np.median(ratios))


__all__ = [name for name in globals() if not name.startswith('__')]
