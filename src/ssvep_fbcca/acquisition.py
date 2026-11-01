from __future__ import annotations

from .common import *
from .config import *
from .signal import *
from .interfaces import EegSource



@dataclass
class Epoch:
    """保存一个对齐后的 EEG 事件窗及其源时间轴和诊断信息。"""

    data: np.ndarray
    fs: float
    raw_timestamps: np.ndarray
    local_timestamps: np.ndarray
    clock_corrections: np.ndarray
    diagnostics: dict[str, Any]
    source_data: np.ndarray
    uniform_timestamps: np.ndarray
    source_timestamps: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        """规范化数组字段，并验证数据与各时间轴的维度。"""
        self.data = _validate_eeg_ct(self.data, name="Epoch.data",
                                     allow_empty_samples=True)
        self.source_data = _validate_eeg_ct(
            self.source_data, name="Epoch.source_data", allow_empty_samples=True)
        self.raw_timestamps = np.asarray(self.raw_timestamps, dtype=float)
        self.local_timestamps = np.asarray(self.local_timestamps, dtype=float)
        self.clock_corrections = np.asarray(self.clock_corrections, dtype=float)
        self.uniform_timestamps = np.asarray(self.uniform_timestamps, dtype=float)
        self.source_timestamps = np.asarray(
            self.raw_timestamps.copy() if self.source_timestamps is None else self.source_timestamps,
            dtype=float)
        self.validate_dimensions()

    def validate_dimensions(self) -> None:
        """验证 C×T 数据矩阵及对应的一维时间轴。"""
        if self.data.shape[0] != self.source_data.shape[0]:
            raise ValueError("Epoch.data 与 Epoch.source_data 通道数不一致")
        for name, timestamps in (
                ("raw_timestamps", self.raw_timestamps),
                ("source_timestamps", self.source_timestamps),
                ("local_timestamps", self.local_timestamps),
                ("clock_corrections", self.clock_corrections)):
            if timestamps.ndim != 1:
                raise ValueError(f"Epoch.{name}必须是一维时间轴")
            if len(timestamps) != self.source_data.shape[1]:
                raise ValueError(
                    f"Epoch.{name}长度必须等于source_data的采样点数")
        if self.uniform_timestamps.ndim != 1:
            raise ValueError("Epoch.uniform_timestamps必须是一维时间轴")
        if len(self.uniform_timestamps) != self.data.shape[1]:
            raise ValueError(
                "Epoch.uniform_timestamps长度必须等于data的采样点数")

def _parse_stream_metadata(info: Any, indices: Sequence[int]) -> dict:
    """解析 LSL 流 XML 元数据并提取选定通道与发布端滤波信息。"""
    xml = info.as_xml
    if callable(xml):
        xml = xml()
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as exc:
        raise RuntimeError("无法解析LSL流元数据") from exc
    channels = root.findall("./desc/channels/channel")
    selected = []
    for i in indices:
        node = channels[i] if i < len(channels) else None
        selected.append({"lsl_index": i, "label": node.findtext("label", "UNKNOWN") if node is not None else "UNKNOWN",
                         "unit": node.findtext("unit", "UNKNOWN") if node is not None else "UNKNOWN"})
    filter_nodes = []
    for element in root.findall("./desc//*"):
        if any(s in element.tag.lower() for s in ("filter", "lowpass", "highpass", "notch")):
            text = " ".join(element.itertext()).strip()
            if text:
                filter_nodes.append(f"{element.tag}: {text}")
    def value(name: str, legacy_name: str) -> Any:
        """兼容读取新版属性和旧版零参数访问器。"""
        item = getattr(info, name, None)
        return item if item is not None else getattr(info, legacy_name)()
    return {"name": value("name", "name"), "type": value("stype", "type"),
            "source_id": value("source_id", "source_id"),
            "channel_count": value("n_channels", "channel_count"),
            "nominal_fs": float(value("sfreq", "nominal_srate")),
            "selected_channels": selected, "publisher_filter_metadata": filter_nodes or ["UNKNOWN"],
            "raw_xml": xml}

class MNEStreamWindow:
    """从 MNE-LSL 环形缓冲读取事件窗；生产环境与 acquire 共用同一把锁。"""

    timestamp_mode = "lsl"

    def __init__(self, stream: Any, indices: Sequence[int], device_lag_s: float,
                 *, lock: Any = None):
        """绑定 MNE-LSL 流、分类通道和可选的设备时延补偿。"""
        self.stream = stream
        self.indices = list(indices)
        self.n_channels = len(indices)
        self.device_lag_s = device_lag_s
        self.total_received = 0
        self.lock = threading.RLock() if lock is None else lock

    def snapshot(self) -> tuple[np.ndarray, ...]:
        """在线程锁内复制当前环形缓冲及其处理后时间戳。"""
        with self.lock:
            data, timestamps = self.stream.get_data(picks=self.indices)
            x = np.asarray(data, dtype=float).copy()
            received = np.asarray(timestamps, dtype=float).copy()
        if (received.ndim != 1
                or x.shape != (self.n_channels, len(received))):
            raise InvalidTrial("MNE-LSL缓冲数据与时间戳形状不一致")
        nonempty = np.flatnonzero(received != 0)
        x, received = x[:, nonempty], received[nonempty]
        local = received - self.device_lag_s
        corrections = np.zeros(len(received), dtype=float)
        valid = np.isfinite(x).all(axis=0)
        return x, received, local, corrections, valid

    @property
    def latest(self) -> float:
        """返回缓冲区最新样本的本地时间；无数据时返回负无穷。"""
        with self.lock:
            _, received = self.stream.get_data(
                winsize=1 / float(self.stream.info["sfreq"]), picks=[self.indices[0]])
            return float(received[-1] - self.device_lag_s) if len(received) and received[-1] else -math.inf

    def window_snapshot(self, start: float, end: float) -> tuple[np.ndarray, ...]:
        """从一次缓冲快照中返回覆盖指定半开区间的支撑样本。"""
        return self._window_from_snapshot(self.snapshot(), start, end)

    @staticmethod
    def _window_from_snapshot(snapshot: tuple[np.ndarray, ...], start: float,
                              end: float) -> tuple[np.ndarray, ...]:
        """从已复制的快照中截取事件窗及左右插值支撑点。"""
        x, received, local, corrections, valid = snapshot
        if len(local) < 2 or local[0] > start or local[-1] < end:
            raise InvalidTrial("事件窗口数据未到齐或已被MNE-LSL缓冲覆盖")
        left = int(np.searchsorted(local, start, side="right") - 1)
        right = int(np.searchsorted(local, end, side="left"))
        support = slice(left, right + 1)
        return (x[:, support], received[support], local[support],
                corrections[support], valid[support], received[support].copy())


    def epoch(self, start: float, duration_s: float, nominal_fs: float,
              max_gap_factor: float = 1.5, rate_tolerance: float = .002,
              hard_gap_factor: float = 2.0, hard_rate_tolerance: float = .005,
              max_warning_gaps: int = 1) -> Epoch:
        """校验指定事件窗的时间轴，并插值到声明采样率的均匀网格。"""
        _validate_fs(nominal_fs, None)
        if not np.isfinite(start) or not np.isfinite(duration_s) or duration_s <= 0:
            raise ValueError("事件时间/窗口长度不合法")
        end = start + duration_s
        snapshot = self.snapshot()
        x, raw, local, corrections, valid, source = self._window_from_snapshot(snapshot, start, end)
        grid = start + np.arange(sample_count(duration_s, nominal_fs)) / nominal_fs
        diagnostics = {
            "timestamp_mode": self.timestamp_mode,
            "nominal_fs_hz": float(nominal_fs),
            "requested_window_start": float(start), "requested_window_end": float(end),
            "source_support_samples": len(local),
            "source_samples_within_window": int(np.count_nonzero((local >= start) & (local < end))),
            "expected_uniform_samples": sample_count(duration_s, nominal_fs),
            "rate_warning_tolerance": float(rate_tolerance),
            "rate_hard_tolerance": float(hard_rate_tolerance),
        }
        def failure_epoch(message: str) -> Epoch:
            """构造保留源数据和失败诊断的空事件窗。"""
            return Epoch(
                data=np.empty((self.n_channels, 0), dtype=float), fs=nominal_fs,
                raw_timestamps=raw.copy(), local_timestamps=local.copy(),
                clock_corrections=corrections.copy(),
                diagnostics={**diagnostics, "hard_reason": message, "uniform_samples": 0},
                source_data=x.copy(), uniform_timestamps=np.empty(0, dtype=float),
                source_timestamps=source.copy())
        if len(local) < 3 or not np.all(valid):
            message = "窗口含NaN/Inf或有效样本不足"
            raise InvalidTrial(message, epoch=failure_epoch(message))
        dt_raw, dt_source, dt_local = np.diff(raw), np.diff(source), np.diff(local)
        nominal_dt = 1 / nominal_fs
        if np.any(dt_source <= 0) or np.any(dt_local <= 0):
            message = "窗口时间戳不严格递增"
            raise InvalidTrial(message, epoch=failure_epoch(message))
        interval = np.maximum(dt_source, dt_local)
        gap_mask = interval > nominal_dt * max_gap_factor
        hard_gap_mask = interval >= nominal_dt * hard_gap_factor
        warnings: list[str] = ["LSL时间戳已同步/去抖；无设备样本计数器，不能逐点确认丢样或重复样本"]
        warning_gap_count = int(np.count_nonzero(gap_mask))
        hard_gap_count = int(np.count_nonzero(hard_gap_mask))
        if hard_gap_count:
            message = ("LSL处理后窗口存在明确缺样级时间戳间隔："
                       f"warning_gaps={warning_gap_count}, hard_gaps={hard_gap_count}")
            raise InvalidTrial(message, epoch=failure_epoch(message))
        if warning_gap_count:
            level = "较多" if warning_gap_count > max_warning_gaps else "轻微"
            warnings.append(f"存在{warning_gap_count}个{level}异常采样间隔；未达到明确缺样阈值，继续处理")
        measured_fs = (len(local) - 1) / (local[-1] - local[0])
        source_measured_fs = (len(source) - 1) / (source[-1] - source[0])
        local_rate_error = abs(measured_fs / nominal_fs - 1)
        source_rate_error = abs(source_measured_fs / nominal_fs - 1)
        diagnostics.update(
            timestamp_estimated_fs=float(measured_fs),
            source_clock_estimated_fs=float(source_measured_fs),
            local_rate_error_fraction=float(local_rate_error),
            source_rate_error_fraction=float(source_rate_error),
            timestamp_nominal_grid_offset_peak_to_peak_s=float(np.ptp(
                local - local[0] - np.arange(len(local)) / nominal_fs)),
            sample_count_interpretation=("count of received samples within postprocessed timestamps; "
                                         "does not establish device sample loss"),
        )
        maximum_rate_error = max(local_rate_error, source_rate_error)
        if maximum_rate_error > hard_rate_tolerance:
            warnings.append(
                f"时间戳采样率偏差较大但仅警告：source={source_measured_fs:.3f}Hz, "
                f"local={measured_fs:.3f}Hz, declared={nominal_fs:g}Hz")
        elif maximum_rate_error > rate_tolerance:
            warnings.append(
                f"时间戳采样率轻微偏差：source={source_measured_fs:.3f}Hz, local={measured_fs:.3f}Hz")
        _validate_fs(min(measured_fs, source_measured_fs), None)
        if grid.size < 32 or grid[-1] > local[-1] or grid[0] < local[0]:
            message = "窗口不满足重采样时间覆盖"
            raise InvalidTrial(message, epoch=failure_epoch(message))
        uniform = np.asarray([np.interp(grid, local, channel) for channel in x], dtype=float)
        epoch = Epoch(data=uniform, fs=nominal_fs, raw_timestamps=raw.copy(),
                     local_timestamps=local.copy(), clock_corrections=corrections.copy(), diagnostics={
            **diagnostics,
            "source_support_samples": len(local), "uniform_samples": len(grid),
            "timestamp_estimated_fs": float(measured_fs), "source_clock_estimated_fs": float(source_measured_fs),
            "max_source_interval_s": float(dt_source.max()), "max_local_interval_s": float(dt_local.max()),
            "timestamp_mode": self.timestamp_mode,
            "timestamp_semantics": "LSL postprocessed local_clock timestamps; original outlet stamps unavailable",
            "raw_nonmonotonic_count": int(np.count_nonzero(dt_raw <= 0)),
            "raw_min_interval_s": float(dt_raw.min()), "raw_max_interval_s": float(dt_raw.max()),
            "timestamp_adjustment_max_abs_s": float(np.max(np.abs(source - raw))),
            "source_support_start": float(local[0]), "source_support_end": float(local[-1]),
            "uniform_last_sample": float(grid[-1]), "end_is_exclusive": True,
            "clock_correction_min_s": float(corrections.min()), "clock_correction_max_s": float(corrections.max()),
            "clock_correction_scope": "already applied inside LSL; saved values are zero placeholders",
            "interpolation": "LSL-postprocessed timestamp grid; no extrapolation; device sample loss unverified",
            "warnings": warnings,
            "warning_gap_count": warning_gap_count,
            "hard_gap_count": hard_gap_count,
            "rate_mismatch_handling": "warning_only",
        }, source_data=x.copy(), uniform_timestamps=grid.copy(), source_timestamps=source.copy())
        return epoch

class ContinuousLSL:
    """管理 MNE-LSL 连续采集、缓冲、健康检查和事件窗读取。"""

    def __init__(self, cfg: Config):
        """根据配置初始化尚未连接的连续采集器。"""
        self.cfg = cfg
        self.stream: Any = None
        self.buffer: Optional[MNEStreamWindow] = None
        self.metadata: dict = {}
        self.fs = math.nan
        self.error: Optional[BaseException] = None
        self.stop_event = threading.Event()
        self.clock: Callable[[], float] = time.monotonic
        self.clock_updates: list[dict[str, float]] = []
        self.estimated_fs = math.nan
        self.timestamp_mode = "lsl"
        self.last_arrival = time.monotonic()
        self.first_timestamp: Optional[float] = None
        self.last_timestamp = -math.inf
        self._stream_lock = threading.RLock()
        self._acquisition_thread: Optional[threading.Thread] = None
        self.recorder: Optional[SessionRecorder] = None

    def _timestamp_summary(self) -> dict[str, Any]:
        """汇总当前 LSL 时间戳处理方式及其可验证范围。"""
        return {
            "mode": "mne_lsl_streamlsl",
            "flags": ["clocksync", "dejitter", "monotonize"],
            "clock_domain": "mne_lsl.lsl.local_clock",
            "manual_time_correction_applied": False,
            "sample_index_reconstruction": False,
            "acquisition_buffer": "MNE-LSL StreamLSL",
            "buffer_consistency": "manual acquire and copied snapshots share one lock",
            "original_outlet_timestamps_available": False,
            "saved_raw_timestamp_field": "legacy name; contains LSL-postprocessed timestamps",
            "device_timestamp_lag_s": self.cfg.device_timestamp_lag_s,
            "accepted_samples_total": self.buffer.total_received if self.buffer is not None else 0,
            "sample_integrity": ("configured device counter saved separately; semantics and continuity unverified"
                                 if self.cfg.sample_counter_channel is not None else
                                 "device sample counter unavailable in selected EEG stream"),
            "physical_latency": "unmeasured; GUI and display delays require independent validation",
        }

    def start(self) -> None:
        """连接唯一匹配的 EEG 流，启动采集线程并完成预热检查。"""
        if self.cfg.timestamp_mode != "lsl":
            raise RuntimeFault("ACQUISITION", "实时采集仅支持LSL内建时间处理；旧重建模式已停用")
        try:
            from mne_lsl.lsl import local_clock, resolve_streams
            from mne_lsl.stream import StreamLSL
        except (ImportError, RuntimeError, OSError) as exc:
            raise RuntimeFault("LSL_LIBRARY", "不能加载MNE-LSL/liblsl：" + str(exc)) from exc
        self.clock = local_clock
        name = self.cfg.eeg_stream_name or None
        stype = self.cfg.eeg_stream_type
        streams = resolve_streams(timeout=self.cfg.connect_timeout_s, name=name, stype=stype)
        if not streams:
            visible = resolve_streams(timeout=min(1.0, self.cfg.connect_timeout_s))
            details = "; ".join(
                f"{sinfo.name!r}(type={sinfo.stype!r}, channels={sinfo.n_channels}, "
                f"fs={sinfo.sfreq:g}Hz)"
                for sinfo in visible
            ) or "无"
            raise RuntimeFault("LSL_NOT_FOUND",
                f"未找到LSL EEG流（name={name or '任意'}, type={stype}）。当前可见流：{details}\n"
                "设备连接成功不代表已开启LSL输出。本程序通过LSL接收EEG，不直接连接USB设备。\n"
                "若使用OpenBCI GUI：先Start Data Stream，再在Networking选择LSL，"
                "数据选择TimeSeriesRaw、Type填EEG，点击Start LSL Stream，保持GUI运行后重试。"
            )
        if len(streams) != 1:
            names = ", ".join(repr(sinfo.name) for sinfo in streams)
            raise RuntimeFault("LSL_CONNECTION",
                f"找到多个匹配的EEG流（{names}），请确认OpenBCI GUI只发布一条{name or stype}流"
            )
        self.stream = StreamLSL(self.cfg.buffer_s, name=streams[0].name,
                                stype=streams[0].stype,
                                source_id=streams[0].source_id or None)
        try:
            self.stream.connect(acquisition_delay=None,
                                processing_flags=("clocksync", "dejitter", "monotize"),
                                timeout=self.cfg.connect_timeout_s)
            info = self.stream.sinfo
            self.fs = float(info.sfreq)
            _validate_fs(self.fs, max(high for _, high in self.cfg.filter_bands))
            if max(self.cfg.channel_indices) >= info.n_channels:
                raise RuntimeError(f"EEG流通道不足：需要{len(self.cfg.channel_indices)}路，实际{info.n_channels}路")
            self.metadata = _parse_stream_metadata(info, self.cfg.channel_indices)
            self.metadata["all_channels"] = _parse_stream_metadata(
                info, range(info.n_channels))["selected_channels"]
            if self.recorder is not None:
                self.recorder.start_continuous(self.metadata)
            self.metadata["timestamp_processing"] = self._timestamp_summary()
            print("[时间轴] MNE-LSL StreamLSL后台采集与环形缓冲；"
                  "LSL内建时钟同步、去抖和单调化。\n"
                  f"收集{self.cfg.startup_buffer_s:g}秒数据用于启动检查，收齐即继续；"
                  "原始发布端时间戳在此模式下不可恢复。"
                  "设备和屏幕的物理延迟仍需独立测量。", flush=True)
            self.buffer = MNEStreamWindow(self.stream, self.cfg.channel_indices,
                                          self.cfg.device_timestamp_lag_s,
                                          lock=self._stream_lock)
            self.last_arrival = time.monotonic()
            self.stream.add_callback(self._observe_chunk)
            if self.recorder is not None:
                self.recorder.capture_active = True
            self._acquisition_thread = threading.Thread(
                target=self._acquire_loop, name="EEG-acquisition", daemon=True)
            self._acquisition_thread.start()
        except BaseException:
            self.close()
            raise
        try:
            deadline = time.monotonic() + self.cfg.startup_timeout_s
            while time.monotonic() < deadline:
                self.check_health()
                with self._stream_lock:
                    first = self.first_timestamp
                    last = self.last_timestamp
                    received = self.buffer.total_received
                if (first is not None and received > 2
                        and last - first >= self.cfg.startup_buffer_s):
                    self.estimated_fs = (received - 1) / (last - first)
                    startup_error = abs(self.estimated_fs / self.fs - 1)
                    if startup_error > self.cfg.hard_rate_tolerance:
                        print(f"[时间轴警告] 启动LSL采样率{self.estimated_fs:.3f}Hz，"
                              f"与声明值{self.fs:g}Hz偏差较大；仅警告并继续。", flush=True)
                    elif startup_error > self.cfg.rate_tolerance:
                        print(f"[时间轴警告] 启动LSL采样率{self.estimated_fs:.3f}Hz，"
                              f"与声明值{self.fs:g}Hz轻微偏差。", flush=True)
                    _validate_fs(self.estimated_fs, max(high for _, high in self.cfg.filter_bands))
                    return
                self.stop_event.wait(.02)
            raise RuntimeFault("ACQUISITION", "已找到LSL流，但启动阶段未收到足够的连续EEG数据")
        except BaseException:
            self.close()
            raise

    def _acquire_loop(self) -> None:
        """串行更新 MNE-LSL 缓冲；不在 acquire 的中间状态发布快照。"""
        while not self.stop_event.is_set():
            try:
                with self._stream_lock:
                    if self.stop_event.is_set():
                        return
                    if self.stream is None or not self.stream.connected:
                        raise RuntimeError("MNE-LSL采集流已断开")
                    self.stream.acquire()
                    if not self.stream.connected:
                        raise RuntimeError("MNE-LSL采集异常后断开")
                    if self.error is not None:
                        return
            except BaseException as exc:
                if self.error is None:
                    self.error = exc
                return
            self.stop_event.wait(.01)

    def _observe_chunk(self, data: np.ndarray, ts: np.ndarray, info: Any) -> tuple[np.ndarray, np.ndarray]:
        """保存回调所见的所有通道并检查健康状态；磁盘写入在独立线程完成。"""
        try:
            stamps = np.asarray(ts, dtype=float)
            if (data.ndim != 2 or len(stamps) != len(data)
                    or data.shape[1] <= max(self.cfg.channel_indices)):
                raise InvalidTrial("MNE-LSL采集块形状或时间戳无效")
            if len(stamps) and self.recorder is not None:
                self.recorder.record_continuous(data, stamps, self.clock())
            if not np.isfinite(stamps).all():
                raise InvalidTrial("MNE-LSL时间戳含NaN/Inf")
            if len(stamps) and (np.any(np.diff(stamps) <= 0) or stamps[0] <= self.last_timestamp):
                raise InvalidTrial("MNE-LSL时间戳回退或重复，停止以避免刺激与EEG错配")
            if len(stamps):
                if self.first_timestamp is None:
                    self.first_timestamp = float(stamps[0])
                self.last_timestamp = float(stamps[-1])
                self.last_arrival = time.monotonic()
                if self.buffer is not None:
                    self.buffer.total_received += len(stamps)
        except BaseException as exc:
            self.error = exc
        return data, ts

    def check_health(self) -> None:
        """检查采集、记录线程、流连接状态和最近样本年龄。"""
        if self.recorder is not None:
            self.recorder.check_recording_health()
        if self.error is not None:
            raise RuntimeFault("ACQUISITION", f"采集线程异常：{self.error}") from self.error
        if self.stop_event.is_set():
            raise RuntimeFault("ACQUISITION", "采集已停止")
        with self._stream_lock:
            if self.stream is not None:
                if not self.stream.connected:
                    raise RuntimeFault("ACQUISITION", "MNE-LSL采集线程已断开")
                if time.monotonic() - self.last_arrival > self.cfg.max_receive_age_s:
                    raise RuntimeFault("ACQUISITION", "EEG流停止发送，已停止实验")

    def wait_epoch(self, onset: float, cancel: threading.Event) -> Epoch:
        """等待目标事件窗到齐，随后返回经过时间轴校验的事件窗。"""
        assert self.buffer is not None
        start = onset + self.cfg.response_delay_s
        end = start + self.cfg.window_s
        deadline = time.monotonic() + self.cfg.data_wait_timeout_s
        while self.buffer.latest < end:
            if cancel.is_set():
                raise AbortSession("已取消等待EEG")
            self.check_health()
            if time.monotonic() >= deadline:
                raise InvalidTrial(f"EEG未覆盖窗口终点{end:.6f}，该试次不分类")
            cancel.wait(.01)
        self.check_health()
        return self.buffer.epoch(
            start, self.cfg.window_s, self.fs, self.cfg.max_gap_factor,
            self.cfg.rate_tolerance, self.cfg.hard_gap_factor,
            self.cfg.hard_rate_tolerance, self.cfg.max_warning_gaps)

    def close(self) -> None:
        """停止采集线程并安全断开 LSL 流；重复调用不会重复断开。"""
        self.stop_event.set()
        if self._acquisition_thread is not None:
            if self._acquisition_thread.ident is not None:
                self._acquisition_thread.join(timeout=max(2.0, self.cfg.connect_timeout_s))
            if self._acquisition_thread.is_alive():
                raise RuntimeFault("ACQUISITION", "EEG接收线程未及时退出；未并发断开仍在使用的流")
            self._acquisition_thread = None
        if self.recorder is not None:
            self.recorder.capture_active = False
        with self._stream_lock:
            if self.stream is not None:
                try:
                    if self.stream.connected:
                        self.stream.disconnect()
                finally:
                    self.stream = None
        self.metadata["timestamp_processing"] = self._timestamp_summary()

def decode_trial(trial_id: int, onset: float, collector: EegSource, cfg: Config,
                 cancel: threading.Event) -> dict:
    """分类工作线程；函数签名有意不包含真实类别。"""
    epoch = collector.wait_epoch(onset, cancel)
    if cancel.is_set():
        raise AbortSession("已取消分类")
    if cfg.input_unit == "AUTO":
        units = [str(item.get("unit", "UNKNOWN")) for item in
                 collector.metadata.get("selected_channels", [])]
        quality_unit = units[0] if units and len(set(units)) == 1 else "UNKNOWN"
    else:
        quality_unit = cfg.input_unit
    quality = assess_signal_quality(
        epoch.source_data, unit=quality_unit, max_abs=cfg.quality_max_abs,
        rail_min=cfg.quality_rail_min, rail_max=cfg.quality_rail_max,
        rail_margin=cfg.quality_rail_margin,
        max_clip_fraction=cfg.quality_max_clip_fraction, jump_z=cfg.quality_jump_z,
        min_valid_channels=cfg.min_valid_channels,
        max_warning_bad_channels=cfg.max_warning_bad_channels)
    quality["mains_noise"] = mains_noise_diagnostics(epoch.data, epoch.fs, cfg.notch_hz)
    quality["warnings"].extend(quality["mains_noise"]["warnings"])
    epoch.diagnostics["quality"] = quality
    epoch.diagnostics.setdefault("warnings", []).extend(quality["warnings"])
    if quality["hard_fail"]:
        raise InvalidTrial("EEG信号质量不合格：" + "; ".join(quality["hard_reasons"]), epoch=epoch)
    if cancel.is_set():
        raise AbortSession("已取消分类")
    started = time.monotonic()
    from threadpoolctl import threadpool_limits
    from .decoding import FbccaDecoder
    with threadpool_limits(limits=1):
        decoder = getattr(collector, "decoder", None)
        if decoder is None:
            decoder = FbccaDecoder(cfg)
        result = decoder.decode(epoch.data, epoch.fs)
    if cancel.is_set():
        raise AbortSession("已取消分类，未提交结果")
    epoch.diagnostics["notch_processing"] = result["notch_processing"]
    result.update(trial_id=trial_id, epoch=epoch, quality=quality,
                  computation_s=time.monotonic() - started)
    return result



__all__ = [name for name in globals() if not name.startswith('__')]

