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
        for name in ("fbcca_filter_bands_hz", "chat_letter_filter_bands_hz", "chat_number_filter_bands_hz"):
            if values.get(name) is not None:
                values[name] = tuple(tuple(band) for band in values[name])
        cfg = cls(**values)
        cfg.validate()
        return cfg

    # 模式三使用 DeepSeek Chat Completions；密钥只从环境变量读取。
    deepseek_model: str = "deepseek-chat"
    deepseek_api_url: str = "https://api.deepseek.com/chat/completions"
    deepseek_api_key_env: str = "DEEPSEEK_API_KEY"
    deepseek_api_key_file: Optional[str] = None
    deepseek_timeout_s: float = 60.0
    # 模式三用户通过 BCI 逐字输入（如 ALS 患者）；回复需简短、纯文本、便于朗读。
    deepseek_system_prompt: Optional[str] = (
        "The user has ALS and types with a brain-computer interface, one letter at a "
        "time, in capital letters, often without punctuation and with occasional typos. "
        "Interpret short or misspelled messages generously. Reply in the user's language "
        "with short, clear, plain text (no Markdown, lists, tables or emoji) because the "
        "reply is shown on screen and read aloud. When you need more information, ask one "
        "simple question that can be answered with a word or two."
    )
    # 模式三无障碍选项（面向无法用手操作的用户，如 ALS）。
    chatbox_voice_enabled: bool = True          # 朗读回复和语音提示
    chatbox_voice_rate: int = -1                # 语速 -10..10，略慢更易听清
    chatbox_speak_selections: bool = True       # 每次识别后读出所选键
    chatbox_feedback_s: float = 0.6             # 所选键高亮确认时长
    chatbox_reading_chars_per_s: float = 14.0   # 无语音进度时的估计滚动速度
    chatbox_reply_hold_s: float = 3.0           # 读完后停留时间，再恢复闪烁输入
    chatbox_predictions_enabled: bool = True    # 字母页顶部三个候选词键（关闭则为30个目标）
    chatbox_llm_predictions: bool = True        # 候选词同时参考 DeepSeek 上下文预测
    chatbox_prediction_wait_s: float = 1.0      # 每轮开始前最多等待 DeepSeek 候选的时间
    # 不打算输入时的保护：复用 rejection_* 阈值拦截低置信轮次。
    chatbox_rejection_enabled: bool = True
    # 滑动窗口：窗长、识别间隔、分数平滑权重（1表示不平滑）。
    # 各界面每个有效窗口直接选中，不做连续窗口计数。
    window_s: float = 5.0
    decode_step_s: float = 0.25
    score_ema: float = 1.0
    # 模式3两页独立窗口；chat_window_s 保留为字母页字段名。
    # 保留2026-10-21末版本（675db9a）的算法和拒识阈值，分析窗按要求统一为5秒。
    chat_window_s: float = 5.0
    chat_number_window_s: float = 5.0
    chat_score_ema: float = 1.0
    chat_min_score: float = 0.25934562367893293
    # 第一/第二名分差门槛。2026-10-19 提示式数据（40 类，5 s 窗）：0 时放行窗口中 71% 正确，
    # 0.05 时 88% 正确；被拒的窗口不丢弃本轮，只是等后续滑动窗口。
    chat_min_margin: float = 0.05
    # 每轮闪烁开始后先留给用户移动视线的时间；这段 EEG 不进入任何分析窗。
    chat_gaze_shift_s: float = 1.0
    # 按频率的分数归一化文件（python -m ssvep_fbcca.score_norm 生成）。可用时模式3改按
    # z 分数选键，并用下面两个 z 门槛代替 chat_min_score/chat_min_margin。
    chat_score_norm_file: Optional[str] = None
    chat_norm_min_z: float = 4.0
    chat_norm_min_margin: float = 0.5
    # 模式1方向键的二次确认及行驶看门狗。
    chair_confirm_settle_s: float = 0.45
    chair_confirm_timeout_s: float = 7.0
    chair_watchdog_s: float = 1.0
    response_delay_s: float = 0.23
    blank_min_s: float = 0.5
    feedback_s: float = 0.5

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
    rejection_max_alpha_ratio: Optional[float] = None
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
    # 保留分界面配置接口，各界面默认恢复2026-10-21的3谐波、5子带FBCCA。
    menu_n_harmonics: int = 3
    chair_n_harmonics: int = 3
    chat_n_harmonics: int = 3
    chat_weight_a: float = 1.25
    chat_weight_b: float = 0.25
    chat_cca_regularization: float = 1e-10
    chat_letter_filter_bands_hz: tuple[tuple[float, float], ...] = (
        (6., 90.), (14., 90.), (22., 90.), (30., 90.), (38., 90.))
    chat_number_filter_bands_hz: tuple[tuple[float, float], ...] = (
        (6., 90.), (14., 90.), (22., 90.), (30., 90.), (38., 90.))
    weight_a: float = 1.25
    weight_b: float = 0.25
    notch_hz: Optional[float] = 50.0
    # 10月21日实现保留的可选工频回归，原启动配置未启用。
    line_regression_hz: Optional[float] = None
    cca_regularization: float = 1e-10
    # None 保持原来的 M3 五子带；显式配置时允许离线验证其他 FBCCA 频带。
    fbcca_filter_bands_hz: Optional[tuple[tuple[float, float], ...]] = None
    participant_id: Optional[str] = None
    rejection_calibration_id: Optional[str] = None

    full_screen: bool = True
    screen_index: int = 0
    window_size: tuple[int, int] = (1920, 1080)
    viewing_distance_cm: float = 70.0
    screen_diagonal_inches: Optional[float] = None
    stimulus_size_deg: float = 4.0
    stimulus_center_pitch_deg: float = 6.0
    frame_long_factor: float = 1.5
    frame_short_factor: float = 0.5
    frame_rate_tolerance: float = 0.005
    marker_stream_name: str = "FBCCA_40_Keyboard_Events"

    @property
    def channel_labels(self) -> tuple[str, ...]:
        """按真实LSL列号标注通道；不把历史位置名称当成当前接线。"""
        return tuple(f"CH{index + 1}" for index in self.channel_indices)

    @property
    def startup_buffer_s(self) -> float:
        """返回启动检查前必须积累的最短数据时长（秒）。"""
        return max(self.warmup_s, 2.1)

    @property
    def menu_profile(self) -> "Config":
        """仅主菜单覆盖谐波数，不修改共享配置。"""
        return replace(self, n_harmonics=self.menu_n_harmonics)

    @property
    def chair_profile(self) -> "Config":
        """模式1独立的谐波数；窗口和门控沿用原设置。"""
        return replace(self, n_harmonics=self.chair_n_harmonics)

    @property
    def chat_profile(self) -> "Config":
        """兼容旧接口：返回模式3字母页配置；实时两页使用 chat_page_profile。"""
        return self.chat_page_profile(1)

    def chat_page_profile(self, page: int) -> "Config":
        """按当前页面生成独立解码配置，切页不会修改共享配置。"""
        if isinstance(page, bool) or page not in (1, 2):
            raise ValueError("ChatBox page 必须为1（字母）或2（数字）")
        return replace(self, n_harmonics=self.chat_n_harmonics,
                       window_s=self.chat_window_s if page == 1 else self.chat_number_window_s,
                       fbcca_filter_bands_hz=(self.chat_letter_filter_bands_hz if page == 1 else
                                              self.chat_number_filter_bands_hz),
                       weight_a=self.chat_weight_a, weight_b=self.chat_weight_b,
                       cca_regularization=self.chat_cca_regularization,
                       score_ema=self.chat_score_ema,
                       rejection_min_score=self.chat_min_score,
                       rejection_min_margin=self.chat_min_margin)

    @property
    def filter_bands(self) -> tuple[tuple[float, float], ...]:
        """返回 FBCCA 使用的滤波器组频带。"""
        return (M3_5_BANDS if self.fbcca_filter_bands_hz is None else
                tuple(tuple(band) for band in self.fbcca_filter_bands_hz))

    @property
    def rejection_thresholds_configured(self) -> bool:
        """判断 ChatBox 是否完整启用了分数、分差和频谱拒识。"""
        return bool(
            self.rejection_enabled
            and self.rejection_min_score is not None
            and self.rejection_min_margin is not None
            and self.rejection_max_alpha_ratio is not None
            and math.isfinite(float(self.rejection_min_score))
            and math.isfinite(float(self.rejection_min_margin))
            and math.isfinite(float(self.rejection_max_alpha_ratio))
        )

    def validate(self) -> None:
        """验证所有配置项及跨字段约束，不合法时抛出 ``ValueError``。"""
        for name in ("deepseek_model", "deepseek_api_url", "deepseek_api_key_env"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} 不能为空")
        if (self.deepseek_api_key_file is not None
                and (not isinstance(self.deepseek_api_key_file, str)
                     or not self.deepseek_api_key_file.strip())):
            raise ValueError("deepseek_api_key_file 必须是非空路径或空值")
        if not self.deepseek_api_url.startswith(("http://", "https://")):
            raise ValueError("deepseek_api_url 必须是 HTTP(S) URL")
        if (not math.isfinite(self.deepseek_timeout_s)
                or self.deepseek_timeout_s <= 0):
            raise ValueError("deepseek_timeout_s 必须为有限正数")
        if self.deepseek_system_prompt is not None and not isinstance(self.deepseek_system_prompt, str):
            raise ValueError("deepseek_system_prompt 必须是字符串或空值")
        for name in ("chatbox_voice_enabled", "chatbox_speak_selections",
                     "chatbox_predictions_enabled", "chatbox_llm_predictions",
                     "chatbox_rejection_enabled"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} 必须为 true 或 false")
        if (isinstance(self.chatbox_voice_rate, bool) or not isinstance(self.chatbox_voice_rate, int)
                or not -10 <= self.chatbox_voice_rate <= 10):
            raise ValueError("chatbox_voice_rate 必须为 -10 到 10 的整数")
        for name in ("chatbox_feedback_s", "chatbox_reading_chars_per_s", "chatbox_reply_hold_s",
                     "chatbox_prediction_wait_s"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0 or (name == "chatbox_reading_chars_per_s" and value == 0):
                raise ValueError(f"{name} 必须为有限非负数（语速须为正数）")
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
        for name in ("score_ema", "chat_score_ema"):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} 必须在(0, 1]内")
        for name in ("window_s", "chat_window_s", "chat_number_window_s"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} 必须为有限正数")
        if self.chat_score_norm_file is not None and (
                not isinstance(self.chat_score_norm_file, str) or not self.chat_score_norm_file.strip()):
            raise ValueError("chat_score_norm_file 必须是非空路径或空值")
        for name in ("chat_min_score", "chat_min_margin", "chat_gaze_shift_s",
                     "chat_norm_min_z", "chat_norm_min_margin"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} 必须非负且有限")
        for name in ("decode_step_s", "chair_confirm_timeout_s", "chair_watchdog_s"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} 必须为有限正数")
        if not math.isfinite(self.chair_confirm_settle_s) or self.chair_confirm_settle_s < 0:
            raise ValueError("chair_confirm_settle_s 必须非负且有限")
        for name in ("n_harmonics", "menu_n_harmonics", "chair_n_harmonics", "chat_n_harmonics",
                     "min_valid_channels"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} 必须是正整数")
        for name in ("response_delay_s", "blank_min_s", "feedback_s"):
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
        if not (math.isfinite(self.stimulus_size_deg)
                and math.isfinite(self.stimulus_center_pitch_deg)
                and 0 < self.stimulus_size_deg < self.stimulus_center_pitch_deg < 90):
            raise ValueError("刺激视觉角度须满足 0 < size < center_pitch < 90")
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
        if (self.rejection_max_alpha_ratio is not None
                and (not math.isfinite(self.rejection_max_alpha_ratio)
                     or self.rejection_max_alpha_ratio <= 0)):
            raise ValueError("rejection_max_alpha_ratio 必须为正且有限")
        _validate_filter_bands(self.filter_bands, self.target_fs, self.target_fs)
        for bands in (self.chat_letter_filter_bands_hz, self.chat_number_filter_bands_hz):
            _validate_filter_bands(bands, self.target_fs, self.target_fs)
        band_high = max(high for bands in (self.filter_bands, self.chat_letter_filter_bands_hz,
                                           self.chat_number_filter_bands_hz) for _, high in bands)
        for name in ("weight_a", "weight_b", "cca_regularization", "chat_weight_a", "chat_weight_b",
                     "chat_cca_regularization"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} 必须非负且有限")
        _normalise_channel_indices(self.channel_indices)
        if len(self.channel_indices) != len(self.channel_positions):
            raise ValueError("channel_indices 和 channel_positions 数量必须一致")
        if not 1 <= self.min_valid_channels <= len(self.channel_indices):
            raise ValueError("min_valid_channels 超出所选通道数")
        if self.buffer_s <= max(self.window_s, self.chat_window_s, self.chat_number_window_s) + self.response_delay_s + self.data_wait_timeout_s + 2:
            raise ValueError("内存缓冲区过短，可能覆盖仍待分类的 EEG")
        _validate_harmonics(self.n_harmonics, BENCHMARK_FREQUENCIES_HZ, self.target_fs)
        _validate_line_regression_hz(self.line_regression_hz, self.target_fs)
        for name in ("menu_n_harmonics", "chair_n_harmonics", "chat_n_harmonics"):
            _validate_harmonics(getattr(self, name), BENCHMARK_FREQUENCIES_HZ, self.target_fs)
        if self.upstream_lowpass_hz is not None:
            if not math.isfinite(self.upstream_lowpass_hz) or self.upstream_lowpass_hz < band_high:
                raise ValueError(f"已知发布端低通小于{band_high:g} Hz，不能使用当前频带")

CONFIG = Config(
    # 与2026-10-21启动配置一致；旧阈值不构成对当前界面的新验证。
    rejection_enabled=True,
    rejection_min_score=0.25934562367893293,
    rejection_min_margin=0.0,
    rejection_max_alpha_ratio=2.7631716193293068,
    rejection_calibration_id="offline_20261019_target12_screen_around_eyes",
)

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

M3_5_BANDS = tuple((float(8 * n - 2), 90.0) for n in range(1, 6))

DEFAULT_WINDOW_S = 5.0

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
    bands = M3_5_BANDS if filter_bands is None else tuple(tuple(band) for band in filter_bands)
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
    """验证10月21日FBCCA实现中的可选工频回归频率。"""
    if value is None:
        return
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.integer, np.floating))
            or not np.isfinite(value) or not 0 < value < fs / 2):
        raise ValueError("line_regression_hz须为None或低于Nyquist频率的有限正数")



__all__ = [name for name in globals() if not name.startswith('__')]

