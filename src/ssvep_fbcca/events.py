from __future__ import annotations

from .common import *
from .config import *



@dataclass
class EventStamp:
    """记录一个实验事件及其试次、目标、详情和 LSL 时间戳。"""

    kind: str
    trial_id: int
    target_id: Optional[int] = None
    details: dict = field(default_factory=dict)
    timestamp: Optional[float] = None

class EventMarkers:
    """callOnFlip回调只记本地LSL时间并入队；网络发送不占用呈现线程。

    EEG截窗直接使用这个事件对象的时间，不经由Marker流绕一圈，也不重复校时。
    Marker线程随后仍以原始事件时间作为LSL timestamp，而非发送时刻。
    """

    def __init__(self, cfg: Config, clock: Callable[[], float], outlet: Any = None):
        """创建标记流，并启动异步发送线程。"""
        self.clock, self.error = clock, None
        self.events: list[EventStamp] = []
        self.pending: queue.SimpleQueue = queue.SimpleQueue()
        self.closed = False
        if outlet is None:
            from mne_lsl.lsl import StreamInfo, StreamOutlet
            info = StreamInfo(cfg.marker_stream_name, "Markers", 1, 0,
                              "string", f"fbcca-keyboard-{uuid.uuid4()}")
            desc = info.desc
            desc.append_child_value("clock", "local_clock in PsychoPy callOnFlip")
            desc.append_child_value("physical_display_latency", "UNMEASURED")
            desc.append_child_value("event_delivery", "queued; original timestamp preserved")
            target_node = desc.append_child("targets")
            for target in TARGETS:
                node = target_node.append_child("target")
                for key, value in vars(target).items():
                    node.append_child_value(key, str(value))
            outlet = StreamOutlet(info, chunk_size=1)
        self.outlet = outlet
        self.thread = threading.Thread(target=self._send, name="LSL-markers", daemon=True)
        self.thread.start()

    def mark(self, event: EventStamp) -> None:
        """在调用线程立即记时，并将事件加入异步发送队列。"""
        if self.closed or event.timestamp is not None:
            raise RuntimeError("事件重复标记或事件模块已经关闭")
        event.timestamp = self.clock()
        self.events.append(event)
        self.pending.put(event)

    def _send(self) -> None:
        """持续发送排队事件，并保留首次发送异常供主线程检查。"""
        while True:
            event = self.pending.get()
            if event is None:
                return
            try:
                payload = json.dumps({"event": event.kind, "trial_id": event.trial_id,
                    "target_id": event.target_id, "details": event.details}, ensure_ascii=True)
                self.outlet.push_sample([payload], timestamp=event.timestamp)
            except Exception as exc:
                self.error = exc

    def close(self) -> None:
        """通知标记线程退出并等待其结束。"""
        if not self.closed:
            self.closed = True
            self.pending.put(None)
            self.thread.join(timeout=2)
            if self.thread.is_alive():
                self.error = RuntimeError("Marker发送线程未及时退出")

@dataclass
class TrialRecord:
    """保存单个试次从呈现到分类、质控和落盘的完整状态。"""

    trial_id: int
    block_id: int
    true_class: Optional[int]
    status: str = "started"
    reason: str = ""
    start: float = 0.0
    end: float = 0.0
    cue_onset: Optional[float] = None
    stimulus_onset: Optional[float] = None
    stimulus_offset: Optional[float] = None
    phase_times: list[tuple[str, float]] = field(default_factory=list)
    frame_flip_times: list[float] = field(default_factory=list)
    frame_lsl_times: list[float] = field(default_factory=list)
    frame_diagnostics: dict = field(default_factory=dict)
    result: Optional[dict] = None
    text_applied: bool = False
    mode: str = "cued"
    requested_window_start: Optional[float] = None
    requested_window_end: Optional[float] = None
    actual_stimulus_s: Optional[float] = None
    artifact_refs: dict[str, str] = field(default_factory=dict)
    cue_confirmed_at: Optional[float] = None
    fixation_onset: Optional[float] = None

class TrialLedger:
    """每个trial_id最多提交一次；有效和无效试次都计数。"""

    def __init__(self):
        """初始化空试次集合和自由输入文本。"""
        self.records: list[TrialRecord] = []
        self.ids: set[int] = set()
        self.typed_text = ""

    def apply_prediction(self, pred: int) -> None:
        """将预测类别对应的字符、空格或退格应用到输入文本。"""
        if not 1 <= pred <= len(TARGETS):
            raise ValueError("预测类别超出范围")
        symbol = TARGETS[pred - 1].symbol
        if symbol == "BACK":
            self.typed_text = self.typed_text[:-1]
        elif symbol == "SPACE":
            self.typed_text += " "
        else:
            self.typed_text += symbol

    def commit(self, record: TrialRecord) -> None:
        """校验已结束试次的状态与模式约束，并保证只提交一次。"""
        if record.trial_id in self.ids:
            raise RuntimeError(f"试次{record.trial_id}重复提交")
        if record.status not in ("valid", "rejected", "invalid", "aborted"):
            raise RuntimeError("未结束的试次不能提交")
        if record.mode == "free":
            if record.true_class is not None:
                raise ValueError("自由输入没有真实目标，true_class必须为None")
        elif record.mode == "cued":
            if record.true_class is None or not 1 <= record.true_class <= len(TARGETS):
                raise ValueError("提示测试的真实类别超出范围")
        else:
            raise ValueError("未知输入模式")
        if record.status == "valid":
            if record.result is None or record.result.get("trial_id") != record.trial_id:
                raise RuntimeError("分类结果缺失或属于其他试次")
            pred = record.result["prediction"]
            if (not 1 <= pred <= len(TARGETS)
                    or not 1 <= record.result["fbcca_prediction"] <= len(TARGETS)
                    or not 1 <= record.result["cca_prediction"] <= len(TARGETS)):
                raise ValueError("分类结果超出范围")
            if not record.text_applied:
                raise RuntimeError("有效试次必须先显式写入字符，再提交")
        elif record.text_applied:
            raise RuntimeError("拒识/无效/中止试次不得写入字符")
        self.ids.add(record.trial_id)
        self.records.append(record)

def _json_safe(value: Any) -> Any:
    """将试次元数据转换为不依赖 pickle 的 JSON 值。"""
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    if isinstance(value, (np.integer, np.floating)):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if hasattr(value, "__dict__"):
        return _json_safe(vars(value))
    return str(value)

SYNC_CHANNELS = ("sample_counter", "hardware_timestamp", "photodiode", "trigger")



__all__ = [name for name in globals() if not name.startswith('__')]

