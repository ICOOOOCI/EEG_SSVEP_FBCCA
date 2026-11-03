"""解码器适配器：把配置和 FBCCA 函数封装成可插拔端口。"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from .config import BENCHMARK_FREQUENCIES_HZ, Config
from .interfaces import Decoder
from .signal import classify_eeg_window


class FbccaDecoder:
    """默认 FBCCA/eCCA 解码器实现。

    采集器只持有 ``Decoder`` 端口；未来可替换为 CCA、深度模型或离线回放
    解码器，而无需改动 UI 和试次账本。
    """

    def __init__(self, cfg: Config, *, calibration: Optional[Any] = None,
                 target_frequencies_hz: Optional[Sequence[float]] = None):
        self.cfg = cfg
        self.calibration = calibration
        self.target_frequencies_hz = (None if target_frequencies_hz is None else
                                      tuple(float(value) for value in target_frequencies_hz))

    def decode(self, data: Any, fs: float, *, notch_context: Any = None) -> dict[str, Any]:
        cfg = self.cfg
        return classify_eeg_window(
            data, fs, window_s=cfg.window_s, onset_s=0.0,
            target_frequencies_hz=(BENCHMARK_FREQUENCIES_HZ if self.target_frequencies_hz is None
                                   else self.target_frequencies_hz),
            n_harmonics=cfg.n_harmonics, target_fs=cfg.target_fs,
            notch_hz=cfg.notch_hz, a=cfg.weight_a, b=cfg.weight_b,
            regularization=cfg.cca_regularization,
            min_valid_channels=cfg.min_valid_channels,
            filter_bands=cfg.filter_bands,
            line_regression_hz=cfg.line_regression_hz,
            calibration=self.calibration, notch_context=notch_context,
        )


__all__ = ["FbccaDecoder"]

