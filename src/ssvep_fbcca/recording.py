from __future__ import annotations

from .common import *
from .config import *
from .events import *
from .quality import *
from .calibration import *



class ContinuousRecording:
    """有界队列 + 约一秒的原子分块；不占用呈现线程，不插值、不滤波。

    recording_sample_index 仅为本程序接收到的数据编号，不冒充设备采样计数器。
    分块在成功封装前一直保留；进程意外退出后也可单独读取已完成的分块。
    """

    def __init__(self, folder: Path, channels: int, fs: float):
        """创建分块目录、有界队列和后台写入线程。"""
        self.folder, self.channels, self.fs = folder, channels, fs
        self.folder.mkdir(parents=True, exist_ok=False)
        self.pending: queue.Queue = queue.Queue(maxsize=512)
        self.error: Optional[BaseException] = None
        self.closed = False
        self.accepted_samples = 0
        self.submitted_samples = 0
        self.saved_samples = 0
        self.chunk_count = 0
        self.data_dtype: Optional[np.dtype] = None
        self.parts: list[Path] = []
        self.thread = threading.Thread(target=self._write, name="EEG-file-writer", daemon=True)
        self.thread.start()

    def append(self, data: np.ndarray, stamps: np.ndarray, arrival: float) -> None:
        """复制一个采集回调块并将其无阻塞地提交给写入线程。"""
        if self.closed or self.error is not None:
            raise RuntimeFault("SAVE", f"连续数据写入已停止：{self.error}")
        if data.shape != (len(stamps), self.channels):
            raise RuntimeFault("SAVE", "连续数据通道数变化，拒绝错误拼接")
        if self.data_dtype is None:
            self.data_dtype = data.dtype
        if data.dtype != self.data_dtype:
            raise RuntimeFault("SAVE", "连续数据类型变化，拒绝有损转换")
        self.submitted_samples += len(stamps)
        item = (np.array(data, copy=True), stamps.copy(),
                float(arrival), self.accepted_samples, self.chunk_count)
        try:
            self.pending.put_nowait(item)
        except queue.Full as exc:
            self.error = RuntimeError("连续录制队列已满；停止采集，不能宣称完整保存")
            raise RuntimeFault("SAVE", str(self.error)) from exc
        self.accepted_samples += len(stamps)
        self.chunk_count += 1

    def _write(self) -> None:
        """将队列中的采集块聚合为约一秒一个的原子 NPZ 分块。"""
        batch: list = []
        count = 0

        def flush() -> None:
            """将当前批次原子写入磁盘并更新已保存计数。"""
            nonlocal count
            if not batch:
                return
            path = self.folder / f"block_{len(self.parts):06d}.npz"
            SessionRecorder._atomic_npz(path, {
                "source_data": np.concatenate([item[0] for item in batch]).T,
                "lsl_postprocessed_timestamps": np.concatenate([item[1] for item in batch]),
                "callback_lsl_clock": np.concatenate([np.full(len(item[1]), item[2]) for item in batch]),
                "recording_sample_index": np.concatenate([
                    np.arange(item[3], item[3] + len(item[1]), dtype=np.int64) for item in batch]),
                "receive_chunk_index": np.concatenate([
                    np.full(len(item[1]), item[4], dtype=np.int64) for item in batch]),
            })
            self.parts.append(path)
            self.saved_samples += count
            batch.clear()
            count = 0
        try:
            while True:
                try:
                    item = self.pending.get(timeout=.5)
                except queue.Empty:
                    flush()
                    continue
                if item is None:
                    flush()
                    return
                batch.append(item)
                count += len(item[1])
                if count >= max(1, round(self.fs)):
                    flush()
        except BaseException as exc:
            self.error = exc

    def close(self) -> None:
        """排空队列并等待写入线程结束，保留失败时的已有分块。"""
        if self.closed:
            if self.thread.is_alive():
                raise RuntimeFault("SAVE", "连续录制线程仍在写入，未开始封装")
            return
        self.closed = True
        if self.thread.is_alive():
            try:
                self.pending.put(None, timeout=5)
            except queue.Full as exc:
                raise RuntimeFault("SAVE", "连续录制队列无法排空；保留分块") from exc
            self.thread.join(timeout=30)
        if self.thread.is_alive():
            raise RuntimeFault("SAVE", "连续录制线程未及时退出；保留分块")

    def summary(self) -> dict:
        """返回连续录制的样本、分块、数据类型和完整性摘要。"""
        return {"submitted_samples": self.submitted_samples,
                "queued_samples": self.accepted_samples, "saved_samples": self.saved_samples,
                "received_chunks": self.chunk_count, "saved_blocks": len(self.parts),
                "source_dtype": str(self.data_dtype) if self.data_dtype is not None else None,
                "complete_for_submitted_samples": bool(self.closed and not self.thread.is_alive()
                    and self.error is None and self.submitted_samples == self.saved_samples),
                "error": str(self.error) if self.error else None,
                "scope": "Samples delivered to the MNE-LSL callback; upstream loss unverified"}

class SessionRecorder:
    """连续数据与逐试次落盘；两种模式均封装完整会话ZIP。"""

    schema_version = 5

    def __init__(self, cfg: Config, *, root: Optional[str | Path] = None,
                 context: Optional[dict[str, Any]] = None):
        """准备会话路径、清单、程序快照和可选的时间校准证据。"""
        base = Path(root if root is not None else cfg.record_root)
        if not base.is_absolute():
            base = Path(__file__).resolve().parent / base
        timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        self.session_dir = base / f"session_{timestamp}_{uuid.uuid4().hex[:8]}"
        self.manifest_path = self.session_dir / "manifest.json"
        self.progress_path = self.session_dir / "session.json"
        self.session_path = base / (self.session_dir.name + ".npz")
        self.archive_path = base / (self.session_dir.name + ".zip")
        self._started = False
        self.cfg = cfg
        self.export_artifacts: dict[str, str] = {}
        self._records: list[TrialRecord] = []
        self.continuous: Optional[ContinuousRecording] = None
        self.capture_active = False
        self.continuous_metadata: dict = {}
        self.timing_calibration: dict = {"status": "not_provided", "measurement": None}
        if cfg.timing_calibration_file:
            path = Path(cfg.timing_calibration_file).expanduser()
            if not path.is_absolute():
                path = ROOT / path
            content = path.read_bytes()
            measurement = json.loads(content)
            json.dumps(measurement, allow_nan=False)
            if not isinstance(measurement, dict):
                raise ValueError("时间校准文件必须为JSON对象")
            self.timing_calibration = {"status": "provided_not_independently_verified",
                "source_file": str(path), "sha256": hashlib.sha256(content).hexdigest(),
                "measurement": measurement}
        self.program = {
            "program_version": "dual-mode-continuous-recording-6", "export_schema_version": 6,
            "record_schema_version": self.schema_version,
            "source_file": Path(__file__).name,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "python_version": sys.version, "numpy_version": np.__version__,
            "runtime_environment": environment_snapshot(),
        }
        self.manifest: dict[str, Any] = {
            "schema_version": self.schema_version,
            "session_dir": str(self.session_dir),
            "created_utc": timestamp,
            "context": _json_safe(context or {}),
            "trials": [],
        }

    def start_continuous(self, metadata: dict) -> None:
        """验证流元数据并启动包含全部 LSL 通道的连续录制。"""
        if self.continuous is not None:
            raise RuntimeError("连续录制不能重复启动")
        channels = int(metadata["channel_count"])
        for key in SYNC_CHANNELS:
            index = getattr(self.cfg, key + "_channel")
            if index is not None and index >= channels:
                raise RuntimeFault("SAVE", f"{key}通道{index}超出当前LSL流的{channels}列")
        self._start_on_first_trial()
        self.continuous_metadata = {"stream": metadata, "config": vars(self.cfg).copy(),
            "timing_calibration": self.timing_calibration, "program": self.program,
            "array_layout": "channels x samples; all source columns in original order",
            "timestamp_semantics": "LSL clocksync/dejitter/monotonize; NOT hardware capture time",
            "callback_lsl_clock_semantics": "one local LSL clock read per callback, repeated per sample",
            "recording_sample_index_semantics": "receiver row number; NOT device sample counter",
            "recording_scope": "first acquisition callback through collector shutdown, including warmup",
            "source_data_scope": "all LSL channels before receiver filtering; upstream processing may apply"}
        self._atomic_json(self.session_dir / "continuous_metadata.json", self.continuous_metadata)
        self.continuous = ContinuousRecording(self.session_dir / "continuous_blocks",
                                               channels, float(metadata["nominal_fs"]))

    def record_continuous(self, data: np.ndarray, stamps: np.ndarray, arrival: float) -> None:
        """将一个原始 LSL 回调块转交给连续录制器。"""
        if self.continuous is None:
            raise RuntimeFault("SAVE", "连续录制尚未初始化")
        self.continuous.append(data, stamps, arrival)

    def check_recording_health(self) -> None:
        """在后台连续写入失败时抛出标准保存故障。"""
        if self.continuous is not None and self.continuous.error is not None:
            raise RuntimeFault("SAVE", f"连续数据保存失败：{self.continuous.error}")

    def _start_on_first_trial(self) -> None:
        """惰性创建会话目录和初始清单。"""
        if self._started:
            return
        self.session_dir.mkdir(parents=True, exist_ok=False)
        self._atomic_json(self.manifest_path, self.manifest)
        self._started = True

    @staticmethod
    def _atomic_csv(path: Path, columns: Sequence[str], rows: Any) -> None:
        """写入并同步临时 CSV，完成后原子替换目标文件。"""
        temp = path.with_name(path.name + f".tmp-{uuid.uuid4().hex}")
        with temp.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)

    def _export_timing(self, payload: dict) -> dict[str, Path]:
        """磁盘映射合并连续分块；缺失硬件信息以空数组和状态记录，绝不推算填充。"""
        stem = self.session_dir.name
        paths = {key: self.session_dir / f"{stem}_{suffix}" for key, suffix in {
            "continuous": "continuous_eeg.npz", "signals": "sync_signals.npz",
            "timing_calibration": "timing_calibration.json", "sync_report": "sync_report.json",
            "frames": "frames.csv", "events": "events.csv", "metadata": "metadata.json",
            "readme": "recording_readme.txt"}.items()}
        recording = self.continuous
        summary = recording.summary() if recording is not None else {
            "submitted_samples": 0, "queued_samples": 0, "saved_samples": 0,
            "complete_for_submitted_samples": False, "error": "continuous acquisition not started"}
        n = summary["saved_samples"]
        channels = recording.channels if recording is not None else 0
        stream = payload["acquisition_metadata"]
        signal_status = {}
        timing_stats: dict = {"available": False, "physical_latency_measured": False}
        with tempfile.TemporaryDirectory(prefix=".continuous-export-", dir=self.session_dir) as folder, ExitStack() as mappings:
            def array(name: str, shape: tuple, dtype: Any = float) -> np.ndarray:
                """为大型导出数组创建空数组或临时内存映射。"""
                if not all(shape):
                    return np.empty(shape, dtype=dtype)
                mapped = np.lib.format.open_memmap(Path(folder) / (name + ".npy"), mode="w+",
                                                   dtype=dtype, shape=shape)
                mappings.callback(mapped._mmap.close)
                return mapped
            data = array("source_data", (channels, n),
                         recording.data_dtype if recording is not None and recording.data_dtype is not None else float)
            stamps = array("lsl_postprocessed_timestamps", (n,))
            local = array("lag_corrected_lsl_timestamps", (n,))
            arrivals = array("callback_lsl_clock", (n,))
            indices = array("recording_sample_index", (n,), np.int64)
            chunks = array("receive_chunk_index", (n,), np.int64)
            offset = 0
            for part in recording.parts if recording is not None else []:
                with np.load(part, allow_pickle=False) as saved:
                    count = len(saved["lsl_postprocessed_timestamps"])
                    end = offset + count
                    data[:, offset:end] = saved["source_data"]
                    for name, dest in (("lsl_postprocessed_timestamps", stamps),
                                       ("callback_lsl_clock", arrivals),
                                       ("recording_sample_index", indices),
                                       ("receive_chunk_index", chunks)):
                        dest[offset:end] = saved[name]
                    local[offset:end] = stamps[offset:end] - self.cfg.device_timestamp_lag_s
                    offset = end
            if offset != n:
                raise RuntimeError("连续分块样本总数不符，保留原始分块")
            continuous_metadata = {**self.continuous_metadata, "recording": summary,
                "stream": stream, "selected_eeg_channel_indices": self.cfg.channel_indices,
                "applied_device_timestamp_lag_s": self.cfg.device_timestamp_lag_s,
                "original_outlet_timestamps_available": False}
            self._atomic_npz(paths["continuous"], {
                "source_data": data, "lsl_postprocessed_timestamps": stamps,
                "lag_corrected_lsl_timestamps": local, "callback_lsl_clock": arrivals,
                "recording_sample_index": indices, "receive_chunk_index": chunks,
                "selected_eeg_channel_indices": np.asarray(self.cfg.channel_indices, dtype=np.int64),
                "metadata_json": np.array(json.dumps(_json_safe(continuous_metadata), ensure_ascii=False)),
            })
            signals: dict[str, Any] = {"recording_sample_index": indices,
                "lsl_postprocessed_timestamps": stamps, "lag_corrected_lsl_timestamps": local}
            channel_metadata = stream.get("all_channels", [])
            for key in SYNC_CHANNELS:
                index = getattr(self.cfg, key + "_channel")
                available = index is not None and index < channels and n > 0
                values = data[index] if available else np.empty(0, dtype=float)
                signals[key] = values
                finite = values[np.isfinite(values)]
                signal_status[key] = {
                    "status": "recorded_unvalidated" if available else "unavailable",
                    "lsl_channel_index": index, "samples": len(values),
                    "channel_metadata": channel_metadata[index] if available and index < len(channel_metadata) else None,
                    "source": "user-configured column in the same EEG LSL stream" if available else None,
                    "reason": None if available else ("channel not configured" if index is None else "no recorded samples"),
                    "finite_samples": len(finite),
                    "minimum": float(finite.min()) if len(finite) else None,
                    "maximum": float(finite.max()) if len(finite) else None,
                    "hardware_semantics_verified": False,
                }
            signals["metadata_json"] = np.array(json.dumps({
                "signals": signal_status, "hardware_values": "as received, no unit/clock conversion",
                "missing_values": "empty arrays mean unavailable, never synthesized",
                "alignment": "same received rows; device sampling relationship requires validation",
            }, ensure_ascii=False))
            self._atomic_npz(paths["signals"], signals)
            if n:
                delta = np.diff(stamps)
                finite_delta = delta[np.isfinite(delta)]
                monotonic = bool(np.isfinite(stamps).all() and np.all(delta > 0))
                timing_stats = {"available": True, "physical_latency_measured": False,
                    "first_lsl_timestamp": float(stamps[0]), "last_lsl_timestamp": float(stamps[-1]),
                    "strictly_increasing": monotonic,
                    "nonfinite_timestamp_count": int((~np.isfinite(stamps)).sum()),
                    "nonpositive_interval_count": int((delta <= 0).sum()),
                    "median_interval_s": float(np.median(finite_delta)) if len(finite_delta) else None,
                    "maximum_interval_s": float(finite_delta.max()) if len(finite_delta) else None,
                    "timestamp_gap_count": int((delta > self.cfg.max_gap_factor / recording.fs).sum()),
                    "estimated_rate_hz": float((n - 1) / (stamps[-1] - stamps[0])) if monotonic and n > 1 else None,
                    "scope": "postprocessed LSL timestamps; not a hardware loss or jitter measurement"}
            else:
                monotonic = False
            trials = []
            for record in self._records:
                start, end = record.requested_window_start, record.requested_window_end
                covered = bool(monotonic and start is not None and end is not None
                               and local[0] <= start < end <= local[-1])
                trials.append({"trial_id": record.trial_id, "status": record.status,
                    "stimulus_onset_lsl": record.stimulus_onset,
                    "stimulus_offset_lsl": record.stimulus_offset,
                    "requested_window_start_lsl": start, "requested_window_end_lsl": end,
                    "software_window_bounds_covered": covered,
                    "recording_start_index": int(np.searchsorted(local, start)) if covered else None,
                    "recording_end_index_exclusive": int(np.searchsorted(local, end)) if covered else None,
                    "frame_timing_valid": record.frame_diagnostics.get("valid"),
                    "physical_alignment": "unverified"})
        calibration = {**self.timing_calibration,
            "applied_device_timestamp_lag_s": self.cfg.device_timestamp_lag_s,
            "correction_formula": "lag_corrected_lsl_timestamp = lsl_postprocessed_timestamp - device_timestamp_lag_s",
            "zero_lag_meaning": "no correction applied; not a measured zero latency",
            "calibration_file_auto_applied": False,
            "display_latency_correction_applied_s": 0.0,
            "response_delay_s": self.cfg.response_delay_s,
            "response_delay_meaning": "analysis window parameter; not a measured hardware delay",
            "measured_physical_offset_s": None, "measured_physical_jitter_s": None,
            "measured_clock_drift_ppm": None,
            "measurement_scope": "provided evidence is archived verbatim; no automatic physical calibration"}
        self._atomic_json(paths["timing_calibration"], calibration)
        self._atomic_json(paths["sync_report"], {
            "schema_version": 1, "physical_alignment": "unverified",
            "physical_alignment_reason": "需验证光电/触发通道含义、设备采样关系，并进行事件匹配与延迟校准",
            "recording": summary, "software_timestamp_diagnostics": timing_stats,
            "signals": signal_status, "trials": trials,
            "original_outlet_timestamps": {"status": "unavailable", "reason": "MNE-LSL postprocessing enabled"},
            "calibration_file": paths["timing_calibration"].name,
            "acquisition_timestamp_processing": stream.get("timestamp_processing", {}),
            "limitations": ["软件时间戳正常不证明实际出光与采样对齐", "硬件通道未经配置时不推算、不填零冒充测量",
                            "recording_sample_index不是硬件采样序号", "软件间隔统计不能证明设备无丢样"]})
        self._atomic_csv(paths["events"], ("timestamp_lsl", "event", "trial_id", "target_id", "details_json"),
            ({"timestamp_lsl": e.get("timestamp"), "event": e["kind"], "trial_id": e["trial_id"],
              "target_id": e.get("target_id"), "details_json": json.dumps(e.get("details", {}), ensure_ascii=False)}
             for e in payload["events"]))
        def frame_rows():
            """逐行生成每个试次的显示帧时间记录。"""
            for entry in self.manifest["trials"]:
                record = json.loads((self.session_dir / entry["metadata_file"]).read_text(encoding="utf-8"))
                with np.load(self.session_dir / entry["arrays_file"], allow_pickle=False) as saved:
                    flips = saved["frame_flip_times"]
                    clocks = saved["frame_lsl_times"]
                target = record["true_class"]
                for i, stamp in enumerate(flips):
                    yield {"trial_id": record["trial_id"], "frame_index": i,
                        "flip_time_psychopy": stamp,
                        "flip_callback_lsl": clocks[i] if i < len(clocks) else None,
                        "interval_s": stamp - flips[i-1] if i else None,
                        "is_offset_flip": bool(record["stimulus_offset"] is not None and i == len(flips)-1),
                        "stimulus_onset_lsl": record["stimulus_onset"],
                        "target_id": target,
                        "target_frequency_hz": TARGETS[target-1].frequency_hz if target else None}
        self._atomic_csv(paths["frames"], ("trial_id", "frame_index", "flip_time_psychopy", "flip_callback_lsl", "interval_s",
            "is_offset_flip", "stimulus_onset_lsl", "target_id", "target_frequency_hz"), frame_rows())
        self._atomic_json(paths["metadata"], {**payload, "targets": _json_safe(TARGETS),
            "session_id": stem, "created_utc": self.manifest["created_utc"],
            "recording": summary, "timing_calibration": calibration,
            "files": {key: path.name for key, path in paths.items()},
            "frame_clock": "PsychoPy flip return and local LSL clock in callOnFlip are separate readings; neither measures photons"})
        paths["readme"].write_text(
            "EEG会话录制说明\n"
            "同名会话NPZ：原有逐试次EEG、处理网格、分类结果和完整试次元数据。\n"
            "continuous_eeg.npz：从首个采集回调到采集停止的所有LSL通道，包含预热、刺激及间歇。\n"
            "模式2可用events.csv中的stimulus_onset从连续EEG截取每次闪烁开始后的数据；程序不自动估计延迟。\n"
            "模式2闪烁覆盖5秒分析窗及最多300毫秒离线候选起点。\n"
            "source_data为通道×样本，按原流列顺序；selected_eeg_channel_indices给出分类通道。\n"
            "lsl_postprocessed_timestamps已由LSL校时、去抖和单调化，不是原始发布端或硬件时间戳。\n"
            "callback_lsl_clock是采集回调时读取的本机时钟，同一回调内重复，不是逐样本到达或采样时间。\n"
            "recording_sample_index只是接收记录序号，不代表设备采样序号。\n"
            "sync_signals.npz：配置的硬件计数、硬件时间戳、光电和触发列；未获取时为空数组。\n"
            "同步列保留收到的值，不擅自换算单位、校时或解释计数回绕。独立AUX流本版本未连接。\n"
            "events.csv：LSL软件事件；frames.csv：PsychoPy逐帧时间及翻帧回调LSL时间，均非实际出光时刻。\n"
            "timing_calibration.json：外部校准证据及本次实际补偿；0补偿不代表测得0延迟。\n"
            "sync_report.json：记录完整性、软件时间间隔、硬件字段可用性和逐试次覆盖检查。\n"
            "metadata.json：配置、设备元数据、显示信息、程序版本和源代码哈希。\n"
            "光电信号存在也不自动代表已校准；本程序不根据分类准确率猜测物理延迟。\n"
            "异常退出：session目录中的continuous_blocks分块可单独用numpy.load读取。\n"
            "这些分块最多保留到最近一次成功落盘，崩溃前内存中的队列可能未写入。\n",
            encoding="utf-8")
        return paths

    @staticmethod
    def _atomic_json(path: Path, payload: Any) -> None:
        """以 UTF-8 写入并同步临时 JSON，随后原子替换目标文件。"""
        temp = path.with_name(path.name + f".tmp-{uuid.uuid4().hex}")
        with temp.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(_json_safe(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)

    @staticmethod
    def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
        """写入并同步临时压缩 NPZ，随后原子替换目标文件。"""
        temp = path.with_name(path.name + f".tmp-{uuid.uuid4().hex}")
        with temp.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)

    def write_trial(self, record: TrialRecord) -> dict[str, str]:
        """原子保存单个试次的数组、元数据并更新会话清单。"""
        self._start_on_first_trial()
        trial_stem = f"trial_{record.trial_id:04d}"
        metadata_path = self.session_dir / f"{trial_stem}.json"
        arrays_path = self.session_dir / f"{trial_stem}.npz"
        record.artifact_refs = {"metadata": str(metadata_path), "arrays": str(arrays_path)}
        result = record.result or {}
        epoch = result.get("epoch")
        notch_context_metadata = None
        arrays: dict[str, np.ndarray] = {
            "frame_flip_times": np.asarray(record.frame_flip_times, dtype=float),
            "frame_lsl_times": np.asarray(record.frame_lsl_times, dtype=float),
        }
        frame_intervals = record.frame_diagnostics.get("psychopy_frame_intervals_s")
        if frame_intervals is not None:
            arrays["psychopy_frame_intervals_s"] = np.asarray(frame_intervals, dtype=float)
        if epoch is not None:
            epoch.validate_dimensions()
            arrays.update({
                "source_data": np.asarray(epoch.source_data, dtype=float),
                "source_lsl_timestamps": np.asarray(epoch.source_timestamps, dtype=float),
                "source_raw_timestamps": np.asarray(epoch.raw_timestamps, dtype=float),
                "source_timestamps": np.asarray(epoch.source_timestamps, dtype=float),
                "timestamp_adjustments": epoch.source_timestamps - epoch.raw_timestamps,
                "source_local_timestamps": np.asarray(epoch.local_timestamps, dtype=float),
                "clock_corrections": np.asarray(epoch.clock_corrections, dtype=float),
                "uniform_data": np.asarray(epoch.data, dtype=float),
                "uniform_timestamps": np.asarray(epoch.uniform_timestamps, dtype=float),
            })
            notch_context = getattr(epoch, "notch_context", None)
            if notch_context is not None:
                arrays.update({
                    "notch_context_data": np.asarray(notch_context.data, dtype=float),
                    "notch_context_local_timestamps": np.asarray(notch_context.local_timestamps, dtype=float),
                    "notch_context_uniform_timestamps": np.asarray(notch_context.uniform_timestamps, dtype=float),
                    "notch_context_source_offset": np.asarray(notch_context.source_offset, dtype=np.int64),
                    "notch_context_epoch_end": np.asarray(notch_context.epoch_end, dtype=float),
                    "notch_context_requested_history_samples": np.asarray(
                        notch_context.requested_history_samples, dtype=np.int64),
                })
                context_ts = arrays["notch_context_local_timestamps"]
                notch_context_metadata = {
                    "source_offset": int(notch_context.source_offset),
                    "epoch_end": float(notch_context.epoch_end),
                    "requested_history_samples": int(notch_context.requested_history_samples),
                    "saved_samples": int(notch_context.data.shape[1]),
                    "saved_start": float(context_ts[0]) if len(context_ts) else None,
                    "saved_end": float(context_ts[-1]) if len(context_ts) else None,
                    "data_scope": "Selected LSL channels before receiver filtering; past context and current epoch",
                    "timestamp_semantics": "LSL-postprocessed local timestamps; not device hardware timestamps",
                }
        for key in ("scores", "correlations", "weights", "cca_scores", "fbcca_scores", "ecca_features",
                    "history_notched_data"):
            if result.get(key) is not None:
                arrays[key] = np.asarray(result[key], dtype=float)
        self._atomic_npz(arrays_path, arrays)
        frame_diag = dict(record.frame_diagnostics)
        frame_diag.pop("intervals_s", None)
        trial_metadata = {
            "schema_version": self.schema_version,
            "trial_id": record.trial_id, "block_id": record.block_id,
            "mode": record.mode, "true_class": record.true_class,
            "true_symbol": (TARGETS[record.true_class - 1].symbol
                             if record.true_class is not None else None),
            "true_frequency_hz": (TARGETS[record.true_class - 1].frequency_hz
                                   if record.true_class is not None else None),
            "status": record.status, "reason": record.reason,
            "text_applied": record.text_applied,
            "start": record.start, "end": record.end,
            "duration_s": max(0.0, record.end - record.start),
            "cue_onset": record.cue_onset,
            "cue_progression": "automatic_timed" if record.mode == "cued" else None,
            "cue_confirmed_at": record.cue_confirmed_at, "fixation_onset": record.fixation_onset,
            "stimulus_onset": record.stimulus_onset,
            "stimulus_offset": record.stimulus_offset,
            "requested_window_start": record.requested_window_start,
            "requested_window_end": record.requested_window_end,
            "actual_stimulus_s": record.actual_stimulus_s,
            "phase_times": record.phase_times,
            "frame_diagnostics": frame_diag,
            "prediction": result.get("prediction"),
            "decoder": result.get("decoder", "FBCCA"),
            "calibration_id": result.get("calibration_id"),
            "calibration_fallback": result.get("calibration_fallback"),
            "fbcca_prediction": result.get("fbcca_prediction"),
            "frequency_hz": result.get("frequency_hz"),
            "filter_bands_hz": result.get("filter_bands_hz"),
            "n_harmonics": result.get("n_harmonics"),
            "line_regression_hz": result.get("line_regression_hz", self.cfg.line_regression_hz),
            "notch_mode": getattr(self.cfg, "notch_mode", "epoch"),
            "notch_history_s": getattr(self.cfg, "notch_history_s", 4.0),
            "notch_processing": result.get("notch_processing"),
            "notch_context": notch_context_metadata,
            "rejection_diagnostics": result.get("rejection_diagnostics"),
            "cca_prediction": result.get("cca_prediction"),
            "cca_frequency_hz": result.get("cca_frequency_hz"),
            "computation_s": result.get("computation_s"),
            "quality": result.get("quality"),
            "sampling_rate_hz": epoch.fs if epoch is not None else None,
            "epoch_diagnostics": epoch.diagnostics if epoch is not None else None,
            "artifact_refs": record.artifact_refs,
            "arrays_file": arrays_path.name,
            "array_keys": sorted(arrays),
        }
        self._atomic_json(metadata_path, trial_metadata)
        entry = {"trial_id": record.trial_id, "status": record.status,
                 "metadata_file": metadata_path.name, "arrays_file": arrays_path.name}
        self.manifest["trials"].append(entry)
        self._atomic_json(self.manifest_path, self.manifest)
        self._records.append(record)
        return dict(record.artifact_refs)

    @staticmethod
    def _npz_entry(archive: zipfile.ZipFile, name: str, value: Any) -> None:
        """将一个无 pickle 的 NPY 数组成员写入 NPZ 归档。"""
        with archive.open(name + ".npy", "w", force_zip64=True) as handle:
            np.lib.format.write_array(handle, np.asarray(value), allow_pickle=False)

    def _export_session(self, payload: dict[str, Any]) -> str:
        """逐试次合并原始EEG与元数据，生成可直接读取的NPZ。"""
        stem = self.session_dir.name
        self.session_path = self.session_dir / f"{stem}.npz"
        trials = []
        for entry in self.manifest["trials"]:
            with (self.session_dir / entry["metadata_file"]).open(encoding="utf-8") as handle:
                trials.append(json.load(handle))
        with tempfile.TemporaryDirectory(prefix=".export-", dir=self.session_dir) as folder:
            staging = Path(folder)
            raw_path = staging / f"{stem}.npz"
            with zipfile.ZipFile(raw_path, "w", compression=zipfile.ZIP_DEFLATED) as raw:
                has_eeg = []
                for trial in trials:
                    prefix = f"trial_{trial['trial_id']:04d}"
                    with np.load(self.session_dir / trial["arrays_file"], allow_pickle=False) as arrays:
                        has_eeg.append("source_data" in arrays and arrays["source_data"].shape[1] > 0)
                        for key in arrays.files:
                            self._npz_entry(raw, f"{prefix}_{key}", arrays[key])
                        true_class = trial["true_class"]
                        reference = true_class if true_class is not None else trial["prediction"]
                        spectral = spectral_quality(
                            arrays["uniform_data"] if "uniform_data" in arrays else np.empty((0, 0)),
                            trial["sampling_rate_hz"] or math.nan, reference, self.cfg.n_harmonics)
                        spectral["reference_kind"] = ("true_label" if true_class is not None else
                                                      "prediction" if reference is not None else "none")
                    trial["spectral_quality"] = spectral
                    trial["artifact_refs"] = {"session_file": self.session_path.name,
                                              "array_prefix": prefix + "_"}
                    trial.pop("arrays_file", None)
                metadata = {
                    **payload, "program": self.program, "session_id": stem,
                    "targets": _json_safe(TARGETS), "trials": trials,
                    "algorithm": "FB-eCCA" if self.cfg.calibration_file else "FBCCA",
                    "trained_model": self.manifest["context"].get("trained_model"),
                    "data_scope": (
                        "Recorded raw trial windows plus per-trial raw notch history; not a continuous session recording"
                        if getattr(self.cfg, "notch_mode", "epoch") == "history" else
                        "Recorded trial windows only; not a continuous session recording"),
                    "array_layout": "channels x samples; each trial uses trial_NNNN_ prefixed keys",
                    "label_convention": "1-based true class; -1 means unknown (including free mode)",
                    "source_data_scope": "Selected LSL channels before receiver filtering; upstream processing may apply",
                    "channel_labels": list(self.cfg.channel_labels),
                    "electrode_positions_assumed": False,
                    "continuous_data_file": stem + "_continuous_eeg.npz",
                    "sync_report_file": stem + "_sync_report.json",
                }
                metadata.pop("export_status", None)
                metadata.pop("manifest_file", None)
                for key, value in {
                    "trial_ids": np.asarray([t["trial_id"] for t in trials], dtype=np.int64),
                    "labels": np.asarray([t["true_class"] if t["true_class"] is not None else -1
                                          for t in trials], dtype=np.int64),
                    "has_eeg": np.asarray(has_eeg, dtype=bool),
                    "channel_indices": np.asarray(self.cfg.channel_indices, dtype=np.int64),
                    "channel_positions": np.asarray(self.cfg.channel_positions, dtype=str),
                    "channel_labels": np.asarray(self.cfg.channel_labels, dtype=str),
                    "sampling_rates_hz": np.asarray([t["sampling_rate_hz"] or math.nan for t in trials]),
                    "target_fs_hz": self.cfg.target_fs,
                    "metadata_json": json.dumps(_json_safe(metadata), ensure_ascii=False, allow_nan=False),
                }.items():
                    self._npz_entry(raw, key, value)
            with zipfile.ZipFile(raw_path) as archive:
                if archive.testzip() is not None:
                    raise RuntimeError("会话NPZ校验失败，保留逐试次文件")
            with raw_path.open("r+b") as handle:
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(raw_path, self.session_path)
        self.export_artifacts["session"] = str(self.session_path)
        return str(self.session_path)

    def finalize(self, *, summary: dict[str, Any], events: Sequence[EventStamp],
                 acquisition_metadata: dict[str, Any], acquisition_review: dict[str, Any],
                 clock_updates: Sequence[Any], display_info: dict[str, Any],
                 typed_text: str, stop_reason: str) -> Optional[str]:
        """停止写入、导出会话文件、封装 ZIP，并清理已成功封装的临时文件。"""
        if not self._started:
            return None
        if self.capture_active:
            raise RuntimeFault("SAVE", "采集线程尚未停止，不能封装正在写入的连续记录；保留分块")
        if self.continuous is not None:
            self.continuous.close()
        payload = _json_safe({
            "schema_version": self.schema_version,
            "manifest_file": self.manifest_path.name,
            "config": vars(self.cfg), "summary": summary,
            "decoder_settings": {"filter_bank_profile": self.cfg.filter_bank_profile,
                                 "filter_bands_hz": self.cfg.filter_bands,
                                 "n_harmonics": self.cfg.n_harmonics,
                                 "line_regression_hz": self.cfg.line_regression_hz,
                                 "notch_mode": getattr(self.cfg, "notch_mode", "epoch"),
                                 "notch_history_s": getattr(self.cfg, "notch_history_s", 4.0),
                                 "notch_hz": self.cfg.notch_hz,
                                 "notch_Q": 30.0,
                                 "notch_pipeline": (
                                     "raw sample-order history through epoch end -> notch filtfilt -> "
                                     "interpolation to epoch grid -> row normalization -> "
                                     "epoch filter bank; no second epoch notch"
                                     if getattr(self.cfg, "notch_mode", "epoch") == "history" else
                                     "uniform raw epoch -> optional line regression -> row normalization -> "
                                     "epoch notch filtfilt -> epoch filter bank")},
            "events": events, "acquisition_metadata": acquisition_metadata,
            "acquisition_review": acquisition_review, "clock_updates": clock_updates,
            "display_info": display_info, "typed_text": typed_text,
            "stop_reason": stop_reason,
            "program": self.program, "export_status": "pending",
            "continuous_recording": self.continuous.summary() if self.continuous else None,
        })
        self._atomic_json(self.progress_path, payload)
        self.manifest["session_file"] = self.progress_path.name
        self._atomic_json(self.manifest_path, self.manifest)
        try:
            timing_files = self._export_timing(payload)
            self._export_session(payload)
            sidecars: dict[str, str] = {}
            if self.cfg.session_mode == "cued":
                sidecars = export_cued_sidecars(self.session_path)
                package_cued_session(self.session_path, destination=self.archive_path,
                    extra_sources=[path for key, path in timing_files.items() if key != "events"])
            else:
                package_recording_files([self.session_path, *timing_files.values()], self.archive_path)
            self.export_artifacts = {"session": str(self.archive_path)}
        except Exception as exc:
            payload["export_status"] = "failed"
            payload["export_error"] = f"{type(exc).__name__}: {exc}"
            self._atomic_json(self.progress_path, payload)
            self.manifest["export_status"] = "failed"
            self._atomic_json(self.manifest_path, self.manifest)
            raise
        for record in self._records:
            record.artifact_refs = {"session_archive": str(self.archive_path),
                 "session_member": self.session_path.name,
                 "array_prefix": f"trial_{record.trial_id:04d}_"}
        if self.continuous is not None and self.continuous.error is not None:
            return str(self.archive_path)
        for path in {self.session_path, *timing_files.values(), *(Path(p) for p in sidecars.values())}:
            path.unlink()
        if self.continuous is not None:
            for path in self.continuous.parts:
                path.unlink()
            self.continuous.folder.rmdir()
            (self.session_dir / "continuous_metadata.json").unlink()
        for entry in self.manifest["trials"]:
            (self.session_dir / entry["metadata_file"]).unlink()
            (self.session_dir / entry["arrays_file"]).unlink()
        self.progress_path.unlink()
        self.manifest_path.unlink()
        self.session_dir.rmdir()
        return str(self.archive_path)

def export_cued_sidecars(session_path: str | Path) -> dict[str, str]:
    """为模式2会话NPZ导出质控JSON、可加载模型PKL和事件CSV。

    可对旧版单文件会话NPZ再次运行；原NPZ保持原位且不会被修改。
    """
    session_path = Path(session_path)
    with np.load(session_path, allow_pickle=False) as archive:
        session = json.loads(archive["metadata_json"].item())
        if session.get("config", {}).get("session_mode") != "cued":
            raise ValueError("只有模式2提示测试会话可生成质控和个人模型文件")
        saved_config = dict(session["config"])
        saved_config.pop("quick_entry_mode", None)
        cfg = Config(**saved_config)
        bundle = _cued_model_bundle(archive, session, cfg)
    hasher = hashlib.sha256()
    with session_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    digest = hasher.hexdigest()
    bundle["source_npz_sha256"] = digest
    qc = _cued_quality_report(session, bundle, digest)
    stem = session_path.stem
    qc_path = session_path.with_name(stem + "_qc.json")
    model_path = session_path.with_name(stem + "_model.pkl")
    events_path = session_path.with_name(stem + "_events.csv")
    SessionRecorder._atomic_json(qc_path, qc)

    temp = model_path.with_name(model_path.name + f".tmp-{uuid.uuid4().hex}")
    with temp.open("wb") as handle:
        pickle.dump(bundle, handle, protocol=5)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, model_path)

    trial_by_id = {t["trial_id"]: t for t in session["trials"]}
    fieldnames = ("timestamp_lsl", "event", "trial_id", "block_id", "target_id",
                  "target_symbol", "target_frequency_hz", "status", "prediction",
                  "prediction_symbol", "cca_prediction", "details_json")
    temp = events_path.with_name(events_path.name + f".tmp-{uuid.uuid4().hex}")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for event in session["events"]:
            trial = trial_by_id.get(event["trial_id"], {})
            target_id = event.get("target_id")
            prediction = trial.get("prediction")
            writer.writerow({
                "timestamp_lsl": event.get("timestamp"),
                "event": event["kind"],
                "trial_id": event["trial_id"],
                "block_id": trial.get("block_id"),
                "target_id": target_id,
                "target_symbol": TARGETS[target_id - 1].symbol if target_id else None,
                "target_frequency_hz": TARGETS[target_id - 1].frequency_hz if target_id else None,
                "status": trial.get("status"),
                "prediction": prediction,
                "prediction_symbol": TARGETS[prediction - 1].symbol if prediction else None,
                "cca_prediction": trial.get("cca_prediction"),
                "details_json": json.dumps(event.get("details") or {}, ensure_ascii=False,
                                           separators=(",", ":")),
            })
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, events_path)
    return {"qc": str(qc_path), "model": str(model_path), "events": str(events_path)}

def package_cued_session(session_path: str | Path, *,
                         destination: Optional[str | Path] = None,
                         remove_sources: bool = False,
                         extra_sources: Sequence[Path] = ()) -> str:
    """兼容旧四文件会话，也可附加完整连续记录，核对后才可移除源文件。"""
    session_path = Path(session_path)
    stem = session_path.stem
    sources = (session_path,
               session_path.with_name(stem + "_qc.json"),
               session_path.with_name(stem + "_model.pkl"),
               session_path.with_name(stem + "_events.csv"), *extra_sources)
    archive_path = Path(destination) if destination is not None else session_path.with_suffix(".zip")
    return package_recording_files(sources, archive_path, remove_sources=remove_sources)

def package_recording_files(sources: Sequence[Path], archive_path: Path, *,
                            remove_sources: bool = False) -> str:
    """封装会话文件，逐项核对哈希后可选择移除源文件。"""
    if len({path.name for path in sources}) != len(sources):
        raise ValueError("会话ZIP存在重复文件名")
    if archive_path.exists():
        raise FileExistsError(f"会话ZIP已存在，不覆盖：{archive_path}")
    if not all(path.is_file() for path in sources):
        missing = [str(path) for path in sources if not path.is_file()]
        raise FileNotFoundError("会话文件不齐，无法封装：" + ", ".join(missing))
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    temp = archive_path.with_name(archive_path.name + f".tmp-{uuid.uuid4().hex}")
    try:
        with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED,
                             compresslevel=6, allowZip64=True) as archive:
            for path in sources:
                archive.write(path, arcname=path.name)
        with zipfile.ZipFile(temp) as archive:
            if archive.namelist() != [path.name for path in sources] or archive.testzip() is not None:
                raise RuntimeError("会话ZIP目录或CRC校验失败")
            for path in sources:
                original_hash = hashlib.sha256()
                archived_hash = hashlib.sha256()
                with path.open("rb") as source, archive.open(path.name) as member:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        original_hash.update(chunk)
                    for chunk in iter(lambda: member.read(1024 * 1024), b""):
                        archived_hash.update(chunk)
                if original_hash.digest() != archived_hash.digest():
                    raise RuntimeError(f"会话ZIP内容核对失败：{path.name}")
        with temp.open("r+b") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, archive_path)
    finally:
        temp.unlink(missing_ok=True)
    if remove_sources:
        for path in sources:
            path.unlink()
    return str(archive_path)



__all__ = [name for name in globals() if not name.startswith('__')]

