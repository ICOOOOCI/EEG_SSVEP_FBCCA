from __future__ import annotations

from .common import *
from dataclasses import fields as dataclass_fields

try:
    import tomllib as _toml_loader
except ModuleNotFoundError:  # Python 3.10；安装 tomli 后自动使用完整解析器
    try:
        import tomli as _toml_loader
    except ModuleNotFoundError:
        _toml_loader = None


def _read_toml(path: Path) -> dict[str, Any]:
    """读取配置文件；无 tomllib/tomli 时提供受限但无额外依赖的回退解析器。"""
    text = path.read_text(encoding="utf-8")
    if _toml_loader is not None:
        return _toml_loader.loads(text)
    import json

    result: dict[str, Any] = {}
    section: dict[str, Any] = result
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            section = result.setdefault(line[1:-1].strip(), {})
            continue
        if "=" not in line:
            raise ValueError(f"配置行缺少 '='：{raw_line}")
        key, raw_value = (item.strip() for item in line.split("=", 1))
        if raw_value in ("true", "false"):
            value: Any = raw_value == "true"
        elif raw_value.startswith(("\"", "[")):
            value = json.loads(raw_value)
        else:
            try:
                value = float(raw_value) if any(ch in raw_value for ch in ".eE") else int(raw_value)
            except ValueError:
                value = raw_value.strip("\"")
        section[key] = value
    return result



@dataclass
class Config:
    """集中保存会话、采集、解码、显示和持久化配置。"""

    @classmethod
    def from_toml(cls, path: str | Path) -> "Config":
        """从分区 TOML 加载配置，并拒绝拼写错误或未支持的字段。"""
        source = Path(path).expanduser()
        try:
            raw = _read_toml(source)
        except OSError as exc:
            raise ValueError(f"无法读取配置文件：{source}：{exc}") from exc
        values: dict[str, Any] = {}
        for key, value in raw.items():
            if isinstance(value, dict):
                values.update(value)
            else:
                values[key] = value
        allowed = {item.name for item in dataclass_fields(cls)}
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(f"配置文件包含未知字段：{', '.join(unknown)}")
        for name in ("channel_indices", "channel_positions", "window_size"):
            if isinstance(values.get(name), list):
                values[name] = tuple(values[name])
        cfg = cls(**values)
        cfg.validate()
        return cfg

    session_mode: str = "cued"
    free_prepare_s: float = 1.0
    protocol: str = "standard_5s"
    blocks: int = 3
    random_seed: int = 20260928
    cue_s: float = 2.0
    cued_settle_s: float = 0.5
    response_delay_s: float = 0.23
    blank_min_s: float = 0.5
    feedback_s: float = 0.5
    cued_rest_every: int = 40
    cued_rest_s: float = 30.0
    block_rest_s: float = 180.0
    cued_offline_max_delay_s: float = 0.30
    max_consecutive_invalid: int = 3

    eeg_stream_name: Optional[str] = "obci_eeg1"
    eeg_stream_type: str = "EEG"
    timestamp_mode: str = "lsl"
    continue_on_timestamp_anomaly: bool = True
    timestamp_max_jitter_s: float = 0.05
    timestamp_jitter_grace_s: float = 0.5
    timestamp_max_discontinuity_s: float = 0.25
    timestamp_slew_rate_s_per_s: float = 0.01
    channel_indices: tuple[int, ...] = tuple(range(8))
    # 历史序列化字段名保留；新会话内容是通道编号，不是头皮位置。
    channel_positions: tuple[str, ...] = tuple(f"CH{i + 1}" for i in range(8))
    buffer_s: float = 60.0
    connect_timeout_s: float = 10.0
    startup_timeout_s: float = 45.0
    warmup_s: float = 10.0
    data_wait_timeout_s: float = 8.0
    clock_refresh_s: float = 1.0
    max_clock_step_s: float = 0.002
    clock_slew_rate_s_per_s: float = 0.001
    max_receive_age_s: float = 2.0
    max_gap_factor: float = 1.5
    hard_gap_factor: float = 2.0
    rate_tolerance: float = 0.002
    hard_rate_tolerance: float = 0.005
    max_warning_gaps: int = 1
    max_warning_frame_anomalies: int = 1
    min_valid_channels: int = 6
    max_warning_bad_channels: int = 2
    quality_max_abs: Optional[float] = None
    quality_rail_min: Optional[float] = None
    quality_rail_max: Optional[float] = None
    quality_rail_margin: float = 0.0
    quality_max_clip_fraction: float = 0.0
    quality_jump_z: float = 12.0
    rejection_enabled: bool = False
    rejection_min_score: Optional[float] = None
    rejection_min_margin: Optional[float] = None
    record_root: str = "session_records"
    input_unit: str = "AUTO"
    upstream_filter_description: str = "OpenBCI GUI TimeSeriesRaw（用户指定；LSL元数据未必可核实）"
    upstream_lowpass_hz: Optional[float] = None
    device_timestamp_lag_s: float = 0.0
    sample_counter_channel: Optional[int] = None
    hardware_timestamp_channel: Optional[int] = None
    photodiode_channel: Optional[int] = None
    trigger_channel: Optional[int] = None
    timing_calibration_file: Optional[str] = None

    target_fs: float = 250.0
    n_harmonics: int = 3
    filter_bank_profile: str = "m3_5band"
    weight_a: float = 1.25
    weight_b: float = 0.25
    notch_hz: Optional[float] = 50.0
    notch_mode: str = "epoch"
    notch_history_s: float = 4.0
    line_regression_hz: Optional[float] = None
    cca_regularization: float = 1e-10
    participant_id: Optional[str] = None
    calibration_file: Optional[str] = None
    rejection_calibration_id: Optional[str] = None

    full_screen: bool = True
    screen_index: int = 0
    window_size: tuple[int, int] = (1920, 1080)
    viewing_distance_cm: float = 70.0
    screen_diagonal_inches: Optional[float] = None
    frame_long_factor: float = 1.5
    frame_short_factor: float = 0.5
    frame_rate_tolerance: float = 0.005
    marker_stream_name: str = "FBCCA_40_Keyboard_Events"
    output_box_height_px: float = 72.0
    output_box_margin_px: float = 24.0

    @property
    def channel_labels(self) -> tuple[str, ...]:
        """按真实LSL列号标注通道；不把历史位置名称当成当前接线。"""
        return tuple(f"CH{index + 1}" for index in self.channel_indices)

    @property
    def window_s(self) -> float:
        """返回当前实验协议对应的 EEG 分析窗长度（秒）。"""
        presets = {"standard_5s": 5.0, "standard_2s": 2.0, "debug_2s": 2.0,
                   "paper_window_1p25s": 1.25, "paper_offline_5s": 5.0}
        if self.protocol not in presets:
            raise ValueError(f"未知 protocol: {self.protocol}; 可选 {tuple(presets)}")
        return presets[self.protocol]

    @property
    def startup_buffer_s(self) -> float:
        """返回启动检查前必须积累的最短数据时长（秒）。"""
        history_s = self.notch_history_s if self.notch_mode == "history" else 0.0
        return max(self.warmup_s, history_s + 2.1)

    @property
    def stimulus_s(self) -> float:
        """返回响应延迟与分析窗之和，即单次刺激时长。"""
        return self.response_delay_s + self.window_s

    @property
    def cued_stimulus_s(self) -> float:
        """返回模式2为离线延迟分析保留的完整闪烁时长。"""
        return self.window_s + max(self.response_delay_s, self.cued_offline_max_delay_s)

    @property
    def filter_bands(self) -> tuple[tuple[float, float], ...]:
        """返回当前 FBCCA 配置使用的滤波器组频带。"""
        profiles = {"m3": M3_BANDS, "m3_5band": M3_5_BANDS,
                    "line_robust": LINE_ROBUST_BANDS}
        if self.filter_bank_profile not in profiles:
            raise ValueError("filter_bank_profile 必须为 m3、m3_5band 或 line_robust")
        return profiles[self.filter_bank_profile]

    @property
    def rejection_thresholds_configured(self) -> bool:
        """判断自由输入模式是否完整启用了双阈值拒识。"""
        return bool(
            self.rejection_enabled
            and self.rejection_min_score is not None
            and self.rejection_min_margin is not None
            and math.isfinite(float(self.rejection_min_score))
            and math.isfinite(float(self.rejection_min_margin))
        )

    def validate(self) -> None:
        """验证所有配置项及跨字段约束，不合法时抛出 ``ValueError``。"""
        _ = self.window_s
        if self.calibration_file and not (self.participant_id and self.participant_id.strip()):
            raise ValueError("加载个人模型必须设置 participant_id / --participant")
        if self.notch_mode not in ("epoch", "history"):
            raise ValueError("notch_mode 必须为 epoch 或 history")
        if (not math.isfinite(self.notch_history_s) or self.notch_history_s <= 0):
            raise ValueError("notch_history_s 必须为有限正数")
        if self.notch_mode == "history":
            if self.notch_hz is None or not math.isfinite(self.notch_hz) or not 0 < self.notch_hz < self.target_fs / 2:
                raise ValueError("history 模式必须设置有效 notch_hz")
            if self.calibration_file:
                raise ValueError("history 模式目前仅支持基础FBCCA；旧个人模板使用epoch预处理，不能混用")
            if self.line_regression_hz is not None:
                raise ValueError("history 模式不能叠加未验证的单窗工频回归")
            if self.rejection_enabled:
                raise ValueError("history 模式的拒识分数尚未标定，不能套用epoch模式阈值")
        if self.session_mode not in ("free", "cued"):
            raise ValueError("session_mode 必须为 free 或 cued")
        if self.timestamp_mode not in ("lsl", "auto", "strict", "openbci"):
            raise ValueError("timestamp_mode 必须为 lsl；旧值仅供历史会话离线读取")
        if not isinstance(self.continue_on_timestamp_anomaly, bool):
            raise ValueError("continue_on_timestamp_anomaly 必须为布尔值")
        if not math.isfinite(self.timestamp_max_jitter_s) or self.timestamp_max_jitter_s <= 0:
            raise ValueError("timestamp_max_jitter_s 必须为有限正数")
        if not math.isfinite(self.timestamp_jitter_grace_s) or self.timestamp_jitter_grace_s <= 0:
            raise ValueError("timestamp_jitter_grace_s 必须为有限正数")
        if (not math.isfinite(self.timestamp_max_discontinuity_s)
                or self.timestamp_max_discontinuity_s <= self.timestamp_max_jitter_s):
            raise ValueError("timestamp_max_discontinuity_s 必须大于持续偏移门限")
        if not 0 < self.timestamp_slew_rate_s_per_s < .1:
            raise ValueError("时间轴微调速率必须在0和0.1之间")
        if not math.isfinite(self.free_prepare_s) or self.free_prepare_s <= 0:
            raise ValueError("free_prepare_s 必须是有限正数")
        for name in ("blocks", "n_harmonics", "min_valid_channels", "max_consecutive_invalid",
                     "cued_rest_every"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} 必须是正整数")
        for name in ("cue_s", "cued_settle_s", "response_delay_s", "blank_min_s", "feedback_s",
                     "cued_rest_s", "block_rest_s", "cued_offline_max_delay_s"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} 必须非负且有限")
        for name in ("connect_timeout_s", "startup_timeout_s", "warmup_s", "data_wait_timeout_s",
                     "buffer_s", "clock_refresh_s", "max_clock_step_s", "max_receive_age_s",
                     "viewing_distance_cm"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} 必须为有限正数")
        if self.timestamp_mode == "lsl" and self.startup_timeout_s <= self.startup_buffer_s:
            raise ValueError("LSL启动超时必须长于启动检查所需数据时长")
        if self.screen_diagonal_inches is not None and (
                not math.isfinite(self.screen_diagonal_inches) or self.screen_diagonal_inches <= 0):
            raise ValueError("screen_diagonal_inches 必须为空或有限正数")
        if not 1 < self.max_gap_factor < 2:
            raise ValueError("max_gap_factor 必须在1和2之间，不得跨明确缺样插值")
        if not (1 < self.frame_long_factor < 2 and 0 < self.frame_short_factor < 1):
            raise ValueError("帧间隔阈值不合法")
        if not math.isfinite(self.frame_rate_tolerance) or not 0 < self.frame_rate_tolerance < 1:
            raise ValueError("frame_rate_tolerance 必须是(0, 1)内的有限比例")
        if not 0 < self.rate_tolerance < 0.1:
            raise ValueError("rate_tolerance 必须在0和0.1之间")
        if not math.isfinite(self.device_timestamp_lag_s):
            raise ValueError("device_timestamp_lag_s 必须有限")
        sync_indices = []
        for name in ("sample_counter_channel", "hardware_timestamp_channel",
                     "photodiode_channel", "trigger_channel"):
            index = getattr(self, name)
            if index is not None:
                if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                    raise ValueError(f"{name} 必须为空或从0开始的通道整数索引")
                if index in self.channel_indices or index in sync_indices:
                    raise ValueError(f"{name} 不得与分类EEG或其他同步通道重叠")
                sync_indices.append(index)
        for name in ("clock_slew_rate_s_per_s", "quality_rail_margin", "quality_jump_z"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} 必须非负且有限")
        if not math.isfinite(self.quality_max_clip_fraction) or not 0 <= self.quality_max_clip_fraction < 1:
            raise ValueError("quality_max_clip_fraction 必须是[0, 1)内的有限样本比例")
        if not self.hard_gap_factor > self.max_gap_factor:
            raise ValueError("hard_gap_factor 必须大于 max_gap_factor")
        for name in ("max_warning_gaps", "max_warning_frame_anomalies", "max_warning_bad_channels"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} 必须为非负整数")
        if (not math.isfinite(self.hard_rate_tolerance)
                or not self.hard_rate_tolerance >= self.rate_tolerance):
            raise ValueError("hard_rate_tolerance 必须有限且大于或等于 rate_tolerance")
        for name in ("quality_max_abs", "quality_rail_min", "quality_rail_max"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value)):
                raise ValueError(f"{name} 必须有限")
        if self.quality_max_abs is not None and self.quality_max_abs <= 0:
            raise ValueError("quality_max_abs 必须为正")
        if ((self.quality_rail_min is None) != (self.quality_rail_max is None)
                or (self.quality_rail_min is not None and self.quality_rail_min >= self.quality_rail_max)):
            raise ValueError("quality_rail_min/max 必须同时设置且 min < max")
        if (self.quality_rail_min is not None
                and 2 * self.quality_rail_margin >= self.quality_rail_max - self.quality_rail_min):
            raise ValueError("quality_rail_margin 必须小于设备轨值跨度的一半")
        for name in ("rejection_min_score", "rejection_min_margin"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} 必须非负且有限")
        band_high = max(high for _, high in self.filter_bands)
        _validate_fs(self.target_fs, band_high)
        _validate_line_regression_hz(self.line_regression_hz, self.target_fs)
        if self.filter_bank_profile == "line_robust" and self.n_harmonics != 3:
            raise ValueError("line_robust 复测方案必须使用 n_harmonics=3")
        if self.filter_bank_profile == "m3_5band" and self.n_harmonics != 3:
            raise ValueError("m3_5band 方案必须使用 n_harmonics=3")
        _normalise_channel_indices(self.channel_indices)
        if len(self.channel_indices) != len(self.channel_positions):
            raise ValueError("channel_indices 和 channel_positions 数量必须一致")
        if not 1 <= self.min_valid_channels <= len(self.channel_indices):
            raise ValueError("min_valid_channels 超出所选通道数")
        history_s = self.notch_history_s if self.notch_mode == "history" else 0.0
        if self.buffer_s <= self.window_s + self.response_delay_s + self.data_wait_timeout_s + 2 + history_s:
            raise ValueError("内存缓冲区过短，可能覆盖仍待分类的 EEG")
        _validate_harmonics(self.n_harmonics, BENCHMARK_FREQUENCIES_HZ, self.target_fs)
        if self.upstream_lowpass_hz is not None:
            if not math.isfinite(self.upstream_lowpass_hz) or self.upstream_lowpass_hz < band_high:
                raise ValueError(f"已知发布端低通小于{band_high:g} Hz，不能使用当前频带")

CONFIG = Config()

def configure_fbcca_profile(cfg: Config, profile: str) -> Config:
    """同时选择子带、谐波和论文融合权重，避免只切换M3频带却仍用3次谐波。

    窗长和刺激流程独立于算法配置；m3不表示设备、空间布局和时序已严格复现。
    原始采样率在接收/分类入口校验，不能通过升采样绕过Nyquist限制。
    """
    harmonics = {"m3": 5, "m3_5band": 3, "line_robust": 3}
    if profile not in harmonics:
        raise ValueError("FBCCA配置必须为 m3、m3_5band 或 line_robust")
    configured = replace(cfg, filter_bank_profile=profile,
                         n_harmonics=harmonics[profile], weight_a=1.25, weight_b=.25,
                         line_regression_hz=None)
    configured.validate()
    return configured

@dataclass(frozen=True)
class Target:
    """描述一个键盘目标及其网格位置和刺激频率。"""

    class_id: int
    symbol: str
    row: int
    col: int
    frequency_hz: float
    phase_rad: float = 0.0

    @property
    def display_label(self) -> str:
        """返回适合显示在键帽上的文本。"""
        return {"SPACE": "", "BACK": "<-"}.get(self.symbol, self.symbol)

KEY_ROWS = (
    tuple("1234567890"),
    tuple("QWERTYUIOP"),
    (*tuple("ASDFGHJKL"), "BACK"),
    ("SPACE", *tuple("ZXCVBNM,.")),
)

FREQUENCIES_HZ_BY_ROW = (
    (8.0, 12.2, 8.4, 12.6, 8.8, 13.0, 9.2, 13.4, 9.6, 13.8),
    (10.0, 14.2, 10.4, 14.6, 10.8, 15.0, 11.2, 15.4, 11.6, 15.8),
    (12.0, 8.2, 12.4, 8.6, 12.8, 9.0, 13.2, 9.4, 13.6, 9.8),
    (14.0, 10.2, 14.4, 10.6, 14.8, 11.0, 15.2, 11.4, 15.6, 11.8),
)

TARGETS = tuple(Target(i + 1, symbol, r, c, FREQUENCIES_HZ_BY_ROW[r][c])
                for i, (r, c, symbol) in enumerate(
                    (r, c, symbol) for r, row in enumerate(KEY_ROWS) for c, symbol in enumerate(row)))

BENCHMARK_FREQUENCIES_HZ = np.asarray([t.frequency_hz for t in TARGETS])

M3_BANDS = tuple((float(8 * n - 2), 90.0) for n in range(1, 8))

M3_5_BANDS = tuple((float(8 * n - 2), 90.0) for n in range(1, 6))

LINE_ROBUST_BANDS = tuple((float(8 * n - 2), 48.0) for n in range(1, 4))

DEFAULT_WINDOW_S = 2.0

TARGET_FS = 250.0

N_HARMONICS = 3

def _normalise_channel_indices(channel_indices: Sequence[int]) -> np.ndarray:
    """规范化并验证非负且不重复的 EEG 通道索引。"""
    indices = np.asarray(channel_indices)
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError("channel_indices 必须是一维非空数组")
    if not np.issubdtype(indices.dtype, np.integer):
        if not np.all(np.isfinite(indices)) or not np.all(indices == np.round(indices)):
            raise ValueError("channel_indices 必须是整数")
    indices = indices.astype(int)
    if np.any(indices < 0) or len(np.unique(indices)) != len(indices):
        raise ValueError("通道索引必须非负且不重复")
    return indices

def _validate_fs(fs: float, upper_hz: Optional[float] = 90.0) -> None:
    """验证采样率为正数，并在需要时检查奈奎斯特上限。"""
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("采样率必须为有限正数")
    if upper_hz is not None and fs <= 2 * upper_hz:
        raise ValueError(f"原始/处理采样率必须大于{2 * upper_hz:g} Hz，以支持{upper_hz:g} Hz子带上限")

def _validate_filter_bands(filter_bands: Optional[Sequence[Sequence[float]]],
                           fs: float, target_fs: float) -> tuple[tuple[float, float], ...]:
    """验证滤波器组频带及原始、目标采样率。"""
    bands = M3_BANDS if filter_bands is None else tuple(tuple(band) for band in filter_bands)
    if not bands or any(len(band) != 2 for band in bands):
        raise ValueError("滤波组必须包含(low, high)频带")
    for low, high in bands:
        if not np.isfinite([low, high]).all() or not 0 < low < high:
            raise ValueError("滤波频带必须满足 0 < low < high")
    high = max(high for _, high in bands)
    _validate_fs(fs, high)
    _validate_fs(target_fs, high)
    return bands

def _validate_harmonics(n: int, frequencies: np.ndarray, fs: float) -> None:
    """验证谐波数，并保证最高参考谐波低于奈奎斯特频率。"""
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("谐波数必须为正整数")
    if not np.isfinite(fs) or fs <= 0 or np.max(frequencies) * n >= fs / 2:
        raise ValueError("最高参考谐波必须低于Nyquist频率")

def _validate_line_regression_hz(value: Optional[float], fs: float) -> None:
    """验证可选的工频回归频率。"""
    if value is None:
        return
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.integer, np.floating))
            or not np.isfinite(value) or not 0 < value < fs / 2):
        raise ValueError("line_regression_hz须为None或低于Nyquist频率的有限正数")



__all__ = [name for name in globals() if not name.startswith('__')]

