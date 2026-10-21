from __future__ import annotations

import re

from .common import *
from .config import *
from .signal import *
from .acquisition import *
from .events import *
from .recording import *
from .analysis import *
from .interfaces import EegSource, EventSink, TrialSink
from .intent import *
from .llm import ChatBackendError, ChatMessage, DeepSeekChatBackend
from .voice import Speaker, make_speaker
from .predict import WordPredictor, apply_suggestion, llm_suggestions, merge_suggestions
from .wheelchair import DEFAULT_DIRECTION_BY_CLASS, WheelchairBridgeClient



KEY_BASE_SIZE = 140.0

KEY_BASE_GAP = 30.0

GRID_WIDTH_FRACTION = .96

GRID_HEIGHT_FRACTION = .58

LAYOUT_REFERENCE_SIZE = (1470.0, 960.0)
# 窗口模式重新排布所需的最小客户区（更小时保持原布局），以及尺寸停止变化多久后才重新排布。
MIN_WINDOW_SIZE = (800, 520)
RESIZE_SETTLE_S = 0.25

# Mode selection uses three separated frequencies. The decoder only compares
# these three references, using the same FBCCA settings as the main keyboard.
MODE_FREQUENCIES_HZ = (8.0, 10.0, 12.0)
MODE_NAMES = ("MODE 1", "MODE 2", "MODE 3")
# Mode menu follows the supplied reference: a nearly black blue-gray canvas,
# pale lavender selection tiles, and high-contrast black labels.
MODE_MENU_BG = (-.88, -.84, -.78)
MODE_TILE = (.90, .84, .96)
MODE_TILE_BORDER = (-.96, -.96, -.94)
MODE_LABEL = (-1.0, -1.0, -1.0)

# ChatBox palette follows the supplied keyboard reference: a near-black
# blue-gray canvas, pale lavender keys, dark labels, and restrained borders.
# PsychoPy RGB values are in the -1..1 range. Flicker frames are still generated
# by the existing SSVEP pipeline; these colors are for the stable display only.
CHAT_BG = (-.91, -.90, -.86)
CHAT_SURFACE = (.82, .76, .86)       # pale lavender/pink key face
CHAT_SURFACE_DARK = (-.78, -.76, -.72)
CHAT_BORDER = (-.42, -.40, -.38)
CHAT_TEXT = (1.0, 1.0, 1.0)
CHAT_KEY_TEXT = (-.92, -.93, -.95)    # near-black for legible key labels
CHAT_KEY_BORDER = (-.98, -.97, -.94)
CHAT_MUTED = (.64, .62, .66)
CHAT_WARNING = (1.0, .76, -.30)

# ChatBox has its own two-page keyboard. Mode 1 and mode 2 continue to use
# TARGETS and the original 40-key layout above this page-specific layer.
# SUGGEST1-3 are the word-prediction keys (candidate bar above Q-P). They are
# appended so the letter keys keep their class order, and are left out entirely
# when chatbox_predictions_enabled is false (the page then has 30 targets).
CHATBOX_SUGGESTION_SYMBOLS = ("SUGGEST1", "SUGGEST2", "SUGGEST3")
CHATBOX_PAGE1_SYMBOLS = tuple("QWERTYUIOPASDFGHJKLZXCVBNM") + (
    "123", "SPACE", "SEND", "BACKSPACE") + CHATBOX_SUGGESTION_SYMBOLS
CHATBOX_PAGE2_SYMBOLS = tuple("1234567890") + (
    ",", ".", "BACK_MENU", "BACKSPACE", "ABC", "SPACE", "SEND", "REPLAY", "REST")

# IME-style geometry in key-pitch units: (x column, row, width in key widths).
# Row 0 is the top keyboard row; the page switch key sits bottom-left.
CHATBOX_PAGE1_GRID = {
    **{symbol: (-4.5 + index, 0, 1.0) for index, symbol in enumerate("QWERTYUIOP")},
    **{symbol: (-4.0 + index, 1, 1.0) for index, symbol in enumerate("ASDFGHJKL")},
    **{symbol: (-3.0 + index, 2, 1.0) for index, symbol in enumerate("ZXCVBNM")},
    "BACKSPACE": (4.5, 2, 1.4),
    "123": (-4.5, 3, 1.4), "SPACE": (0.0, 3, 3.6), "SEND": (4.5, 3, 1.4),
    # Candidate bar: three wide keys, well apart, above the top letter row.
    "SUGGEST1": (-3.3, -1.1, 2.6), "SUGGEST2": (0.0, -1.1, 2.6), "SUGGEST3": (3.3, -1.1, 2.6),
}
CHATBOX_PAGE2_GRID = {
    **{symbol: (-4.5 + index, 0, 1.0) for index, symbol in enumerate("1234567890")},
    ",": (-1.5, 1, 1.0), "REPLAY": (0.0, 1, 1.4), ".": (1.5, 1, 1.0),
    "BACK_MENU": (-4.5, 2, 1.4), "REST": (0.0, 2, 1.4), "BACKSPACE": (4.5, 2, 1.4),
    "ABC": (-4.5, 3, 1.4), "SPACE": (0.0, 3, 3.6), "SEND": (4.5, 3, 1.4),
}
# Frequencies are permuted from the 8.0-15.8 Hz / 0.2 Hz benchmark grid by an
# offline search that penalises similar frequencies on nearby keys. On the
# letter page, touching keys differ by >=1.4 Hz and any two keys within 0.4 Hz
# have at least one key between them; the number page is spaced further apart.
CHATBOX_PAGE1_FREQUENCY_BY_SYMBOL = {
    "Q": 8.8, "W": 15.4, "E": 13.4, "R": 10.8, "T": 13.0,
    # I moved off 10.0 Hz (2026-10-21 ChatBox collection): a narrow 10.0 Hz EEG component
    # not locked to the flicker outscored the true key on 13.6% of other keys' trials at
    # 5 s (grid median 2.0%); 10.2 Hz is the only free frequency that keeps the rules above.
    "Y": 10.4, "U": 12.6, "I": 10.2, "O": 12.0, "P": 14.2,
    "A": 12.4, "S": 9.8, "D": 8.2, "F": 15.8, "G": 8.4,
    "H": 15.6, "J": 8.0, "K": 15.2, "L": 8.6,
    "Z": 14.8, "X": 11.6, "C": 14.0, "V": 11.8, "B": 13.8, "N": 11.0, "M": 13.2,
    "BACKSPACE": 11.2, "123": 10.6, "SPACE": 9.2, "SEND": 14.6,
    # Suggestion keys use unused grid frequencies; every adjacent key differs by >=1.0 Hz.
    "SUGGEST1": 12.2, "SUGGEST2": 15.0, "SUGGEST3": 9.0,
}
CHATBOX_PAGE2_FREQUENCY_BY_SYMBOL = {
    "1": 8.6, "2": 13.6, "3": 11.0, "4": 15.6, "5": 13.0,
    "6": 9.0, "7": 15.4, "8": 8.0, "9": 13.2, "0": 9.4,
    ",": 8.2, ".": 11.6, "BACK_MENU": 15.8, "BACKSPACE": 15.2,
    "ABC": 10.4, "SPACE": 14.6, "SEND": 10.6,
    # REPLAY re-reads the last reply; 10.2 Hz differs from every nearby key by >=1.2 Hz.
    "REPLAY": 10.2,
    # REST turns the keyboard off; 12.8 Hz differs from every nearby key by >=1.2 Hz.
    "REST": 12.8,
}
# Rest mode: only one WAKE key flickers. It is decoded against decoy frequencies
# that are not shown (spread over the alpha band), so idle alpha activity wins a
# decoy instead of waking the keyboard; it must win on consecutive rounds.
CHATBOX_WAKE_FREQUENCY_HZ = 15.0
CHATBOX_WAKE_DECOYS_HZ = (8.6, 10.2, 11.8, 13.4)
CHATBOX_FREQUENCY_BY_SYMBOL = {1: CHATBOX_PAGE1_FREQUENCY_BY_SYMBOL,
                               2: CHATBOX_PAGE2_FREQUENCY_BY_SYMBOL}

def _chatbox_display_label(symbol: str) -> str:
    """将 ChatBox 页面内部动作名称转换为屏幕标签。"""
    return {"BACK_MENU": "MENU", "BACKSPACE": "DEL", "ABC": "ABC",
            "SPACE": "SPACE", "SEND": "SEND", "123": "123",
            "REPLAY": "REPLAY", "REST": "REST", "SUGGEST1": " ", "SUGGEST2": " ",
            "SUGGEST3": " "}.get(symbol, symbol)

# Spoken confirmation after each recognised key, so users who cannot move can
# hear what was selected without looking away from the keyboard.
CHATBOX_SPOKEN_ACTIONS = {"SPACE": "space", "BACKSPACE": "delete", "123": "numbers",
                          "ABC": "letters", "BACK_MENU": "menu", ",": "comma",
                          ".": "period", "REST": "rest"}

def _is_wide_char(char: str) -> bool:
    """中日韩字符在屏幕上约占两个拉丁字符宽度。"""
    code = ord(char)
    return (0x2E80 <= code <= 0x9FFF or 0xAC00 <= code <= 0xD7AF
            or 0xF900 <= code <= 0xFAFF or 0xFF00 <= code <= 0xFFEF)

# Punctuation that must not begin a wrapped line.
_NO_LINE_START = set("，。！？、；：）」』》,.!?;:)")

def wrap_with_offsets(text: str, width: int) -> list[tuple[str, int]]:
    """按显示宽度折行并返回 (行文本, 行首字符下标)，用于让滚动跟随朗读位置。"""
    lines: list[tuple[str, int]] = []
    start, length = 0, len(text)
    while start < length:
        while start < length and text[start] == " ":
            start += 1
        if start >= length:
            break
        used, index, breakpoint = 0, start, None
        while index < length:
            step = 2 if _is_wide_char(text[index]) else 1
            if used + step > width:
                break
            if index > start and text[index] not in _NO_LINE_START and (
                    text[index] == " " or _is_wide_char(text[index])
                    or _is_wide_char(text[index - 1])):
                breakpoint = index
            used += step
            index += 1
        if index >= length:
            lines.append((text[start:].rstrip(), start))
            break
        can_break_here = text[index] not in _NO_LINE_START and (
            text[index] == " " or _is_wide_char(text[index]) or _is_wide_char(text[index - 1]))
        end = index if can_break_here else (breakpoint or index)
        lines.append((text[start:end].rstrip(), start))
        start = end
    return lines

def plain_chat_text(text: str) -> str:
    """去掉 Markdown 记号并合并空白，使屏幕显示和语音朗读都是干净的纯文本。"""
    text = re.sub(r"```.*?\n|```", " ", text, flags=re.S)
    text = re.sub(r"(?m)^\s{0,3}(#{1,6}|[-*+]|\d+[.)])\s+", " ", text)
    text = re.sub(r"[*_`#>]+", "", text)
    return " ".join(text.split())

def _keyboard_ui_scale(width: float, height: float) -> float:
    """顶部输出框和辅助文字随客户区等比例缩放。"""
    return min(width / LAYOUT_REFERENCE_SIZE[0], height / LAYOUT_REFERENCE_SIZE[1])

def _keyboard_side_margin(width: float) -> float:
    """输出框和键盘共用左右边界，并给最外侧提示框留出空间。"""
    return width * (1 - GRID_WIDTH_FRACTION) / 2

def _display_cm_per_pixel(cfg: Config, window_size: Sequence[float],
                          display_size: Optional[Sequence[float]] = None) -> Optional[float]:
    """对角线来自人工填写；窗口模式需提供整屏的 PsychoPy pix 坐标尺寸。"""
    if cfg.screen_diagonal_inches is None:
        return None
    if display_size is None:
        if not cfg.full_screen:
            raise ValueError("窗口模式的物理尺寸换算需要整块显示器的像素尺寸")
        display_size = window_size
    if len(display_size) != 2 or not all(math.isfinite(v) and v > 0 for v in display_size):
        raise ValueError("显示器宽高必须为有限正数")
    return cfg.screen_diagonal_inches * 2.54 / math.hypot(*display_size)

def calculate_keyboard_layout(cfg: Config, window_size: Sequence[float],
                              display_size: Optional[Sequence[float]] = None
                              ) -> tuple[np.ndarray, float, tuple[float, float], float]:
    """按截图的键间比例居中排布；不同屏幕只做统一等比例缩放。"""
    cfg.validate()
    width, height = map(float, window_size)
    if not all(math.isfinite(value) and value > 0 for value in (width, height)):
        raise ValueError("窗口宽高必须为有限正数")
    rows, cols = len(KEY_ROWS), max(map(len, KEY_ROWS))
    ui_scale = _keyboard_ui_scale(width, height)
    top_reserved = (cfg.output_box_height_px + cfg.output_box_margin_px + 70.0) * ui_scale
    bottom_reserved = 70.0 * ui_scale
    available_width = width - 2 * _keyboard_side_margin(width)
    available_height = height - top_reserved - bottom_reserved
    nominal_width = cols * KEY_BASE_SIZE + (cols - 1) * KEY_BASE_GAP
    nominal_height = rows * KEY_BASE_SIZE + (rows - 1) * KEY_BASE_GAP
    scale = min(available_width / nominal_width,
                GRID_HEIGHT_FRACTION * height / nominal_height,
                available_height / nominal_height)
    if scale <= 0:
        raise RuntimeError("屏幕客户区太小，无法容纳键盘与顶部输出框")
    key, gap = KEY_BASE_SIZE * scale, KEY_BASE_GAP * scale
    grid_height = rows * key + (rows - 1) * gap
    center_y = min(0.0, height / 2 - top_reserved - grid_height / 2)
    positions = np.asarray([
        ((target.col - (cols - 1) / 2) * (key + gap),
         center_y + ((rows - 1) / 2 - target.row) * (key + gap))
        for target in TARGETS
    ])
    return positions, key, (gap, gap), scale

def keyboard_layout_info(cfg: Config, window_size: Sequence[float],
                         display_size: Optional[Sequence[float]] = None) -> dict[str, Any]:
    """保存请求值、实际几何及估算视角；不把主观转头体验当作测量结果。"""
    positions, key, (gap_x, gap_y), scale = calculate_keyboard_layout(cfg, window_size, display_size)
    cm_per_pixel = _display_cm_per_pixel(cfg, window_size, display_size)
    corners = [target for target in TARGETS if target.row in (0, len(KEY_ROWS) - 1)
               and target.col in (0, len(KEY_ROWS[0]) - 1)]
    physical = None
    if cm_per_pixel is not None:
        def angle(length: float) -> float:
            """将屏幕上的像素长度换算为居中视角。"""
            return math.degrees(2 * math.atan(length * cm_per_pixel / (2 * cfg.viewing_distance_cm)))
        physical = {
            "basis": "declared diagonal and distance; square pixels; eye opposite screen center",
            "cm_per_pix_unit": cm_per_pixel,
            "key_size_cm": key * cm_per_pixel, "gap_cm": gap_x * cm_per_pixel,
            "key_centered_visual_angle_deg": angle(key), "gap_centered_visual_angle_deg": angle(gap_x),
            "grid_width_cm": (10 * key + 9 * gap_x) * cm_per_pixel,
            "grid_height_cm": (4 * key + 3 * gap_y) * cm_per_pixel,
            "corners": [{"class_id": t.class_id, "symbol": t.symbol,
                "center_eccentricity_deg": math.degrees(math.atan(
                    math.hypot(*positions[t.class_id - 1]) * cm_per_pixel / cfg.viewing_distance_cm)),
                "outer_edge_eccentricity_deg": math.degrees(math.atan(
                    math.hypot(*(np.abs(positions[t.class_id - 1]) + key / 2))
                    * cm_per_pixel / cfg.viewing_distance_cm))} for t in corners],
        }
    return {
        "layout_strategy": "centered_proportional_fit", "layout_schema_version": 3,
        "coordinate_space": "psychopy_pix_client",
        "requested_layout": {"rows": len(KEY_ROWS), "cols": max(map(len, KEY_ROWS)),
            "gap_to_key_ratio": KEY_BASE_GAP / KEY_BASE_SIZE,
            "max_grid_width_fraction": GRID_WIDTH_FRACTION,
            "max_grid_height_fraction": GRID_HEIGHT_FRACTION},
        "key_size_pix_units": key, "gap_pix_units": gap_x,
        "column_gap_pix_units": gap_x, "row_gap_pix_units": gap_y,
        "layout_scale_from_140px_reference": scale,
        "ui_scale_from_reference": _keyboard_ui_scale(*map(float, window_size)),
        "key_positions_pix_units": positions.tolist(),
        "layout_rows": len(KEY_ROWS), "layout_cols": max(map(len, KEY_ROWS)),
        "target_viewing_distance_cm": cfg.viewing_distance_cm,
        "declared_screen_diagonal_inches": cfg.screen_diagonal_inches,
        "display_size_pix_units": list(display_size if display_size is not None else window_size),
        "physical_estimate": physical,
    }

def _create_keyboard_window(visual: Any, cfg: Config) -> Any:
    """全屏使用目标屏幕尺寸；窗口模式可拖动缩放，布局使用 PsychoPy pix 坐标。"""
    import pyglet
    screens = pyglet.canvas.get_display().get_screens()
    if not 0 <= cfg.screen_index < len(screens):
        raise ValueError(f"显示器编号 {cfg.screen_index} 不可用；检测到 {len(screens)} 个显示器")
    screen = screens[cfg.screen_index]
    if cfg.full_screen:
        size = (screen.width, screen.height)
    else:
        # 请求尺寸超过当前屏幕时缩到屏幕内，避免标题栏或边缘跑出屏幕。
        size = (max(MIN_WINDOW_SIZE[0], min(cfg.window_size[0], screen.width - 40)),
                max(MIN_WINDOW_SIZE[1], min(cfg.window_size[1], screen.height - 80)))
    # PsychoPy 不暴露 pyglet 的 resizable 参数；仅在创建窗口期间替换类以开启拖动缩放。
    pyglet_window = pyglet.window.Window
    if not cfg.full_screen:
        class ResizableWindow(pyglet_window):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                kwargs["resizable"] = True
                super().__init__(*args, **kwargs)
        pyglet.window.Window = ResizableWindow
    try:
        win = visual.Window(size=size, fullscr=cfg.full_screen, screen=cfg.screen_index,
            winType="pyglet", useRetina=True,
            units="pix", color=MODE_MENU_BG, colorSpace="rgb", waitBlanking=True,
            allowGUI=not cfg.full_screen, checkTiming=False, autoLog=False)
    finally:
        pyglet.window.Window = pyglet_window
    win.keyboard_display_size = np.asarray((screen.width, screen.height), dtype=float)
    if not cfg.full_screen:
        _enable_live_resize(win)
    return win

def _enable_live_resize(win: Any) -> None:
    """窗口被拖动缩放时同步 PsychoPy 的客户区、帧缓冲、视口与投影。"""
    # 不调用 set_minimum_size：pyglet 1.4 在 Windows 上延迟处理 WM_GETMINMAXINFO 会访问失效指针而崩溃；
    # 过小的窗口改由 _relayout_if_resized 忽略。
    handle = win.backend.winHandle
    scale = win.getContentScaleFactor()
    psychopy_on_resize = handle.on_resize
    win.last_resize_at = 0.0

    def on_resize(width: int, height: int) -> None:
        psychopy_on_resize(width, height)
        if width <= 0 or height <= 0:  # 最小化
            return
        buffer = np.round(np.asarray((width, height)) * scale).astype(int)
        win.clientSize[:] = (width, height)
        win.backend._frameBufferSize[:] = buffer
        win.viewport = win.scissor = (0, 0, int(buffer[0]), int(buffer[1]))
        win.resetEyeTransform()
        win.last_resize_at = time.monotonic()

    handle.on_resize = on_resize

def _keyboard_coordinate_size(win: Any) -> np.ndarray:
    """返回与 PsychoPy pix 刺激位置/尺寸相同的客户区坐标范围。"""
    size = np.asarray(win.clientSize, dtype=float)
    if size.shape != (2,) or not np.all(np.isfinite(size)) or np.any(size <= 0):
        raise RuntimeError(f"无效的窗口客户区尺寸：{size}")
    return size

class MainThreadGraphicsGC:
    """窗口存活时，将循环垃圾回收限制在有 GL 上下文的主线程。

    pyglet 1.5 的纹理析构可能直接调用 glDeleteTextures。自动 GC 若被
    EEG/分类线程触发，会在没有当前 GL 上下文的线程内造成原生崩溃。
    引用计数仍正常工作；循环垃圾在试次间主动回收，避免长期输入累积。
    """

    def __init__(self) -> None:
        """确认当前位于主线程，并暂时关闭自动循环垃圾回收。"""
        self._check_thread()
        self.was_enabled = gc.isenabled()
        self.closed = False
        gc.disable()

    @staticmethod
    def _check_thread() -> None:
        """要求图形资源操作发生在 Python 主线程。"""
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("图形资源必须在主线程回收")

    def collect(self, win: Any) -> None:
        """在窗口 GL 上下文有效且未记录帧间隔时主动回收循环垃圾。"""
        self._check_thread()
        if self.closed:
            return
        if win.recordFrameIntervals:
            raise RuntimeError("刺激时序记录期间不得执行图形垃圾回收")
        win._setCurrent()
        gc.collect()

    def close(self) -> None:
        """恢复进入上下文前的垃圾回收启用状态。"""
        self._check_thread()
        if self.closed:
            return
        self.closed = True
        if self.was_enabled:
            gc.enable()

class PsychoPyKeyboard:
    """负责 SSVEP 键盘的显示、交互、试次编排和结果反馈。"""

    def __init__(self, cfg: Config, collector: EegSource, markers: Optional[EventSink],
                 ledger: TrialLedger, recorder: Optional[TrialSink] = None,
                 *, static_only: bool = False,
                 wheelchair: Optional[WheelchairBridgeClient] = None):
        """绑定会话组件并在主线程创建 PsychoPy 窗口和视觉刺激。"""
        MainThreadGraphicsGC._check_thread()
        from psychopy import visual, event, logging
        self.cfg, self.collector, self.markers, self.ledger = cfg, collector, markers, ledger
        self.recorder = recorder
        self.wheelchair = wheelchair
        self.static_only = static_only
        self.fatal_fault: Optional[dict] = None
        self.trial_faults: list[dict] = []
        self.visual, self.event = visual, event
        self.clock = collector.clock
        self.cancel = threading.Event()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="FBCCA-decoder")
        # LLM requests get their own worker so a slow reply never blocks EEG decoding.
        self.chat_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="DeepSeek-chat")
        self.prediction_executor = ThreadPoolExecutor(max_workers=1,
                                                      thread_name_prefix="DeepSeek-predict")
        self.win: Any = None
        self.display_info: dict = {}
        self._pause_enabled = False
        self._rest_beep: Any = None
        self._active_decode_cancel: Optional[threading.Event] = None
        self._decode_future: Any = None
        self._chat_future: Any = None
        self._chat_history: list[ChatMessage] = []
        self._chat_page = 1
        self._unread_reply = False
        self._typing_during_reply = False  # caregiver pressed SPACE while a reply was pending
        self._resting = False                # keyboard off; only the WAKE key flickers
        self._rejected_in_a_row = 0
        # Word prediction: local predictor, current candidate words, optional DeepSeek request.
        self._predictor = WordPredictor()
        self._suggestions = ["", "", ""]
        self._prediction_backend: Any = None
        self._prediction_future: Any = None
        self._prediction_key: Any = None
        self._prediction_started = 0.0
        self._speaker: Speaker = Speaker()
        self._chat_backend: Any = None
        self._output_cache: Optional[str] = None
        self._closed = False
        self._graphics_gc = MainThreadGraphicsGC()
        try:
            logging.console.setLevel(logging.ERROR)
            self._build_window()
        except BaseException:
            try:
                self.close()
            except Exception:
                pass
            raise

    def _build_window(self) -> None:
        """创建全部视觉元素，测量刷新率并完成静态渲染时序检查。"""
        visual, cfg = self.visual, self.cfg
        self.win = _create_keyboard_window(visual, cfg)
        self.win.mouseVisible = False
        self.mode_rgb_frames = np.empty((0, 3, 3))
        self._screen_mode = cfg.session_mode
        layout = self._build_visuals()
        self._finish_window_setup(layout)

    def _build_visuals(self) -> dict[str, Any]:
        """按当前客户区尺寸创建全部视觉元素；窗口尺寸变化后可再次调用以重新排布。"""
        visual, cfg = self.visual, self.cfg
        width, height = _keyboard_coordinate_size(self.win)
        layout = keyboard_layout_info(cfg, (width, height), self.win.keyboard_display_size)
        self.positions = np.asarray(layout["key_positions_pix_units"])
        key, gap_y = layout["key_size_pix_units"], layout["row_gap_pix_units"]
        scale = key / KEY_BASE_SIZE
        ui_scale = layout["ui_scale_from_reference"]
        self.key_size = key
        cue_width = self._cue_width = min(36.0 * scale, max(0.0, gap_y - 12 * scale) * 9 / 5)
        self._show_cue_arrow = cue_width >= 4
        cue_height = cue_width * 5 / 9
        self.cue_offset = 6 * scale + cue_height / 2
        self._static_colors = np.tile((.72, .58, .82), (len(TARGETS), 1))
        self._cue_colors = np.tile((-.33, -.37, -.25), (len(TARGETS), len(TARGETS), 1))
        for index in range(len(TARGETS)):
            self._cue_colors[index, index] = self._static_colors[index]
        self.key_borders = visual.ElementArrayStim(self.win, nElements=len(TARGETS), units="pix",
            xys=self.positions, sizes=(key + 4 * scale, key + 4 * scale), elementTex=None, elementMask=None,
            colors=np.tile((-.88, -.89, -.84), (len(TARGETS), 1)), colorSpace="rgb", autoLog=False)
        self.squares = visual.ElementArrayStim(self.win, nElements=len(TARGETS), units="pix",
            xys=self.positions, sizes=(key, key), elementTex=None, elementMask=None,
            colors=self._static_colors, colorSpace="rgb", autoLog=False)
        self._static_label_color = (-.64, -.67, -.58)
        self._cue_label_color = (.63, .57, .69)
        self._label_cue_target_id: Optional[int] = None
        self.labels = []
        for target, pos in zip(TARGETS, self.positions):
            label_height = key * (.32 if len(target.display_label) > 1 else .42)
            self.labels.append(visual.TextStim(self.win, text=target.display_label, pos=pos,
                height=label_height, units="pix", color=self._static_label_color,
                colorSpace="rgb", bold=False, autoLog=False))
        self.outline_shadow = visual.Rect(self.win, width=key + 14 * scale, height=key + 14 * scale,
            fillColor=None, lineColor="black", lineWidth=max(4, 10 * scale), units="pix", autoLog=False)
        self.outline = visual.Rect(self.win, width=key + 14 * scale, height=key + 14 * scale,
            fillColor=None, lineColor="#FFE45C", lineWidth=max(2, 6 * scale), units="pix", autoLog=False)
        self.triangle = (visual.ShapeStim(self.win,
            vertices=((-cue_width/2, -cue_height/2), (cue_width/2, -cue_height/2), (0, cue_height/2)),
            fillColor="#FFE45C", lineColor="black", lineWidth=max(1, 2 * scale), units="pix", autoLog=False)
            if self._show_cue_arrow else None)
        output_height = cfg.output_box_height_px * ui_scale
        output_margin = cfg.output_box_margin_px * ui_scale
        output_y = height/2 - output_margin - output_height/2
        output_width = width - 2 * _keyboard_side_margin(width)
        output_text_width = output_width - 48 * ui_scale
        self.output_box = visual.Rect(self.win, width=output_width, height=output_height,
            pos=(0, output_y), fillColor=(-0.82, -0.82, -0.82), lineColor="white", lineWidth=2*ui_scale,
            units="pix", autoLog=False)
        output_font_height = min(28, max(16, cfg.output_box_height_px*0.42)) * ui_scale
        self.output_text = visual.TextStim(self.win, text="", pos=(0, output_y),
            height=output_font_height, wrapWidth=output_text_width,
            font="Courier New", color="white", alignText="left", anchorHoriz="center", autoLog=False)
        self.output_caption = visual.TextStim(
            self.win, text="",
            pos=(-output_width / 2 + 22 * ui_scale, output_y + output_height * .30),
            height=13 * ui_scale, font="Arial", color="white",
            bold=True, alignText="left", anchorHoriz="left", autoLog=False)
        self.output_box.autoDraw = False
        self.output_text.autoDraw = False
        self.output_caption.autoDraw = False
        self._output_max_chars = max(12, int(output_text_width / output_font_height))
        self._sync_output_text(force=True)
        self.header = visual.TextStim(self.win, text="",
            pos=(0, output_y - output_height/2 - 22*ui_scale),
            height=24*ui_scale, wrapWidth=width-40*ui_scale, color="white", autoLog=False)
        self.cue_title = visual.TextStim(self.win, text="TARGET", pos=self.header.pos,
            height=34*ui_scale, wrapWidth=width-40*ui_scale,
            color="#FFE45C", bold=True, autoLog=False)
        self.footer = visual.TextStim(self.win, text="",
            pos=(0, -height/2+18*ui_scale), height=18*ui_scale, wrapWidth=width-40*ui_scale,
            color="white", autoLog=False)
        self.message = visual.TextStim(self.win, text="", pos=(0, -15*ui_scale),
            height=26*ui_scale, wrapWidth=width*.88, color="white", autoLog=False)
        menu_font = "Arial"
        self.mode_title = visual.TextStim(self.win, text="SELECT MODE",
            pos=(0, height*.30), height=42*ui_scale, font=menu_font,
            color="white", autoLog=False)
        mode_square_size = min(300*ui_scale, width*.17, height*.28)
        mode_gap = max(48*ui_scale, mode_square_size*.82)
        mode_total_width = 3 * mode_square_size + 2 * mode_gap
        if mode_total_width > width*.86:
            mode_square_size *= width*.86 / mode_total_width
            mode_gap = max(36*ui_scale, mode_square_size*.82)
        mode_y = 0.0
        mode_x = mode_square_size + mode_gap
        self.mode_positions = np.asarray(((-mode_x, mode_y), (0.0, mode_y),
                                          (mode_x, mode_y)), dtype=float)
        self.mode_key_size = mode_square_size
        self.mode_static_colors = np.tile(MODE_TILE, (3, 1))
        self.mode_squares = visual.ElementArrayStim(self.win, nElements=3, units="pix",
            xys=self.mode_positions, sizes=(mode_square_size, mode_square_size),
            elementTex=None, elementMask=None, colors=self.mode_static_colors,
            colorSpace="rgb", autoLog=False)
        self.mode_borders = visual.ElementArrayStim(self.win, nElements=3, units="pix",
            xys=self.mode_positions, sizes=(mode_square_size + 4*ui_scale,
                                            mode_square_size + 4*ui_scale), elementTex=None,
            elementMask=None, colors=np.tile(MODE_TILE_BORDER, (3, 1)),
            colorSpace="rgb", autoLog=False)
        self.mode_outline = visual.Rect(self.win, width=mode_square_size + 18*ui_scale,
            height=mode_square_size + 18*ui_scale, fillColor=None, lineColor="#FFE45C",
            lineWidth=max(1, 3*ui_scale), units="pix", autoLog=False)
        self.mode_labels = []
        mode_label_text = ("MODE 1\nFREE INPUT", "MODE 2\nCUED TEST",
                           "MODE 3\nDEEPSEEK\nCHATBOX")
        for index, label_text in enumerate(mode_label_text):
            self.mode_labels.append(visual.TextStim(
                self.win, text=label_text, pos=self.mode_positions[index],
                height=mode_square_size*.13, wrapWidth=mode_square_size*.9,
                font=menu_font, color=MODE_LABEL, bold=True, autoLog=False))
        self.mode_status = visual.TextStim(self.win, text="",
            pos=(0, -height/2 + 125*ui_scale), height=20*ui_scale,
            font=menu_font, color="#BDBDC6", autoLog=False)
        self.mode_hint = visual.TextStim(self.win,
            text="",
            pos=(0, -height/2 + 82*ui_scale), height=20*ui_scale,
            font=menu_font, color="white", autoLog=False)
        self.mode_warning = visual.TextStim(self.win,
            text="Visual flicker from 8–15.8 Hz will start  ·  Press ESC if uncomfortable",
            pos=(0, -height/2 + 46*ui_scale), height=18*ui_scale,
            font=menu_font, color="#FFCB52", autoLog=False)
        self.startup_title = visual.TextStim(self.win, text="Checking display",
            pos=(0, height/2 - 64*ui_scale), height=34*ui_scale,
            font=menu_font, color="white", autoLog=False)
        self.startup_dot = visual.Circle(self.win, radius=7*ui_scale,
            pos=(-158*ui_scale, height/2 - 64*ui_scale),
            fillColor="#FFCB52", lineColor="#FFCB52", units="pix", autoLog=False)
        self.startup_subtitle = visual.TextStim(self.win, text="Please wait · No flicker",
            pos=(0, height/2 - 105*ui_scale), height=19*ui_scale,
            font=menu_font, color="#BDBDC6", autoLog=False)
        processing_key_size = min(120*ui_scale, key*1.1)
        self._processing_key_size = processing_key_size
        self._processing_cued_positions = ((0, -height*.005), (0, -height*.075),
                                          (0, -height*.135))
        self._processing_free_positions = ((0, height*.055), (0, -height*.015),
                                          (0, -height*.075))
        self.processing_key = visual.Rect(self.win, width=processing_key_size,
            height=processing_key_size, pos=(0, height*.10),
            fillColor=(-.20, -.24, -.10), lineColor="#BEB8C5",
            lineWidth=max(1, 2*ui_scale), units="pix", autoLog=False)
        self.processing_key_label = visual.TextStim(self.win, text="",
            pos=self.processing_key.pos, height=processing_key_size*.43,
            color="white", font=menu_font, autoLog=False)
        self.processing_dot = visual.Circle(self.win, radius=7*ui_scale,
            pos=(0, -height*.005), fillColor="#FFCB52", lineColor="#FFCB52",
            units="pix", autoLog=False)
        self.processing_title = visual.TextStim(self.win, text="Processing EEG",
            pos=(0, -height*.075), height=34*ui_scale,
            font=menu_font, color="white", bold=True, autoLog=False)
        self.processing_subtitle = visual.TextStim(self.win, text="Please wait for the result",
            pos=(0, -height*.135), height=20*ui_scale,
            font=menu_font, color="#BDBDC6", autoLog=False)
        self.result_verdict = visual.TextStim(self.win, text="",
            pos=(0, height*.11), height=68*ui_scale,
            font=menu_font, color="#92E5B6", bold=True, autoLog=False)
        self.result_pair = visual.TextStim(self.win, text="",
            pos=(0, height*.005), height=34*ui_scale, wrapWidth=width*.9,
            font=menu_font, color="white", autoLog=False)
        self.result_frequency = visual.TextStim(self.win, text="",
            pos=(0, -height*.055), height=19*ui_scale,
            font=menu_font, color="#BDBDC6", autoLog=False)
        self.result_divider = visual.Line(self.win,
            start=(-min(width*.13, 160*ui_scale), -height*.105),
            end=(min(width*.13, 160*ui_scale), -height*.105),
            lineColor="#777781", lineWidth=max(1, ui_scale), units="pix", autoLog=False)
        self.result_details = visual.TextStim(self.win, text="",
            pos=(0, -height*.18), height=19*ui_scale, wrapWidth=width*.85,
            font=menu_font, color="#BDBDC6", autoLog=False)
        self.result_warning = visual.TextStim(self.win, text="",
            pos=(0, -height*.29), height=16*ui_scale, wrapWidth=width*.85,
            font=menu_font, color="#FFCB52", autoLog=False)
        self.pause_title = visual.TextStim(self.win, text="Paused",
            pos=(0, height*.015), height=56*ui_scale,
            font=menu_font, color="white", autoLog=False)
        self.pause_subtitle = visual.TextStim(self.win, text="",
            pos=(0, -height*.055), height=20*ui_scale, wrapWidth=width*.85,
            font=menu_font, color="white", autoLog=False)
        self.rest_title = visual.TextStim(self.win, text="Rest your eyes",
            pos=(0, height*.10), height=44*ui_scale,
            font=menu_font, color="white", autoLog=False)
        self.rest_timer = visual.TextStim(self.win, text="",
            pos=(0, -height*.015), height=112*ui_scale,
            font=menu_font, color="white", autoLog=False)
        self.rest_remaining_label = visual.TextStim(self.win, text="remaining",
            pos=(0, -height*.12), height=22*ui_scale,
            font=menu_font, color="#BDBDC6", autoLog=False)
        self.rest_auto_label = visual.TextStim(self.win, text="Resumes automatically",
            pos=(0, -height*.19), height=18*ui_scale,
            font=menu_font, color="#BDBDC6", autoLog=False)
        # Mode 3 splits the window: chat history and the draft box fill the top
        # half, and the SSVEP keyboard fills the bottom half like a phone IME.
        chat_w = width * .92
        chat_left = -chat_w / 2
        chat_top, chat_bottom = height * .435, height * .257
        chat_pad = 18 * ui_scale
        self.chat_title = visual.TextStim(self.win, text="MODE 3  ·  DEEPSEEK CHATBOX",
            pos=(chat_left, height * .468), height=18*ui_scale, font=menu_font,
            color="#FFE45C", bold=True, alignText="left", anchorHoriz="left", autoLog=False)
        self.chat_panel = visual.Rect(self.win, width=chat_w, height=chat_top - chat_bottom,
            pos=(0, (chat_top + chat_bottom) / 2), fillColor=CHAT_SURFACE_DARK,
            lineColor=CHAT_BORDER, lineWidth=max(1.5, 2*ui_scale), units="pix", autoLog=False)
        self.chat_history_caption = visual.TextStim(
            self.win, text="CONVERSATION", pos=(chat_left + chat_pad, chat_top - chat_pad),
            height=13*ui_scale, font=menu_font, color="#FFE45C", bold=True,
            alignText="left", anchorHoriz="left", autoLog=False)
        self.chat_history_rule = visual.Line(
            self.win, start=(chat_left + chat_pad, chat_top - 2.1 * chat_pad),
            end=(-chat_left - chat_pad, chat_top - 2.1 * chat_pad),
            lineColor=CHAT_BORDER, lineWidth=max(1, ui_scale), units="pix", autoLog=False)
        chat_font = 18 * ui_scale
        chat_text_top = chat_top - 2.6 * chat_pad
        chat_text_w = chat_w - 2 * chat_pad
        self.chat_history_text = visual.TextStim(self.win, text="",
            pos=(chat_left + chat_pad, chat_text_top),
            height=chat_font, wrapWidth=chat_text_w, font=menu_font, color=CHAT_TEXT,
            alignText="left", anchorHoriz="left", anchorVert="top", autoLog=False)
        # Arial averages roughly half an em per glyph; PsychoPy line spacing is ~1.2 em.
        self._chat_wrap_chars = max(20, int(chat_text_w / (chat_font * .47)))
        self._chat_max_lines = max(3, int((chat_text_top - chat_bottom - chat_pad / 2)
                                          / (chat_font * 1.22)))
        self.chat_scroll_hint = visual.TextStim(self.win, text="",
            pos=(-chat_left - chat_pad, chat_bottom + chat_pad * .6), height=13*ui_scale,
            font=menu_font, color="#FFE45C", bold=True, alignText="right",
            anchorHoriz="right", autoLog=False)
        self.chat_status = visual.TextStim(self.win, text="",
            pos=(-chat_left, height * .468), height=15*ui_scale, font=menu_font,
            color="#FFE45C", alignText="right", anchorHoriz="right", autoLog=False)
        # Shared stims (draft box, header, footer) move between two layouts: the
        # 40-key screen (modes 1-2, mode menu) and the ChatBox split screen (mode 3).
        # _apply_screen_layout switches them whenever a mode is entered.
        draft_h = height * .05
        draft_y = chat_bottom - height * .012 - draft_h / 2
        draft_text_h = min(26 * ui_scale, draft_h * .42)
        chatbox_layout = {
            "output_box": {"width": chat_w, "height": draft_h, "pos": (0, draft_y),
                           "fillColor": CHAT_SURFACE_DARK, "lineColor": "#FFE45C",
                           "lineWidth": max(1.5, 2 * ui_scale)},
            "output_text": {"pos": (0, draft_y - draft_h * .12), "height": draft_text_h,
                            "wrapWidth": chat_w - 2 * chat_pad, "color": CHAT_TEXT},
            "output_caption": {"text": "YOUR MESSAGE", "color": "#FFE45C", "height": 12 * ui_scale,
                               "pos": (chat_left + chat_pad * .8, draft_y + draft_h * .30)},
            # Draft box is autoDrawn (on top), so the page/state label lives in the top bar.
            "header": {"pos": (0, height * .468), "height": 15 * ui_scale,
                       "color": CHAT_TEXT, "bold": True},
            "footer": {"pos": (0, -height * .475), "height": 15 * ui_scale, "color": CHAT_MUTED},
            "window": {"color": CHAT_BG, "output_max_chars": max(
                12, int((chat_w - 2 * chat_pad) / (draft_text_h * .62)))},
        }
        # Snapshot exactly the attributes the ChatBox preset changes, from the 40-key build.
        keyboard_layout = {
            name: {attr: np.copy(value) if isinstance(value, np.ndarray) else value
                   for attr in values
                   for value in (getattr(getattr(self, name), attr),)}
            for name, values in chatbox_layout.items() if name != "window"}
        keyboard_layout["window"] = {"color": MODE_MENU_BG,
                                     "output_max_chars": self._output_max_chars}
        self._screen_layouts = {"keyboard": keyboard_layout, "chatbox": chatbox_layout}
        self._apply_screen_layout(self._screen_mode)
        self._build_chatbox_page_visuals(width, height, ui_scale, menu_font)
        self._layout_size = (width, height)
        return layout

    def _relayout_if_resized(self) -> None:
        """窗口尺寸稳定后按新尺寸重建视觉元素，并保留各元素当前的文字与自动绘制状态。"""
        size = tuple(_keyboard_coordinate_size(self.win))
        if (size == self._layout_size
                or size[0] < MIN_WINDOW_SIZE[0] or size[1] < MIN_WINDOW_SIZE[1]
                or time.monotonic() - getattr(self.win, "last_resize_at", 0.0) < RESIZE_SETTLE_S):
            return
        from psychopy.visual.basevisual import MinimalStim
        texts, auto_draw = {}, []
        for name, value in vars(self).items():
            if isinstance(value, MinimalStim):
                if hasattr(value, "text"):
                    texts[name] = value.text
                if value in self.win._toDraw:  # 未设置过的 autoDraw 属性本身为真值，不能直接判断
                    auto_draw.append(name)
        # 按原自动绘制顺序恢复；PsychoPy 按插入顺序比较深度，顺序打乱会报错。
        auto_draw.sort(key=lambda name: self.win._toDraw.index(getattr(self, name)))
        for name in auto_draw:
            getattr(self, name).autoDraw = False
        page_texts = {(page, index): item["label"].text
                      for page, items in self.chat_page_visuals.items()
                      for index, item in enumerate(items)}
        layout = self._build_visuals()
        for name, text in texts.items():
            stim = getattr(self, name, None)
            if hasattr(stim, "text") and stim.text != text:
                stim.text = text
        for name in auto_draw:
            getattr(self, name).autoDraw = True
        for (page, index), text in page_texts.items():
            label = self.chat_page_visuals[page][index]["label"]
            if label.text != text:
                label.text = text
        self._label_cue_target_id = None
        self._sync_output_text(force=True)
        if self._chat_history:
            self._format_chat_history()
        self.display_info.update(layout)
        self.display_info["client_size_reported"] = tuple(map(int, self.win.clientSize))

    def _finish_window_setup(self, layout: dict[str, Any]) -> None:
        """测量刷新率、生成闪烁帧表并完成静态渲染时序检查。"""
        cfg = self.cfg
        cue_width = self._cue_width
        for _ in range(30):
            self._check_abort()
            self._draw_startup_check()
            self.win.flip()
        refresh = self._measure_refresh_rate()
        if refresh is None or not np.isfinite(refresh) or refresh <= 2 * BENCHMARK_FREQUENCIES_HZ.max():
            raise RuntimeError("未测得足够稳定的刷新率；不以猜测的60Hz继续实验")
        self.refresh_hz = float(refresh)
        self.win.refreshThreshold = cfg.frame_long_factor / self.refresh_hz
        mode_frames = max(1, math.ceil(cfg.stimulus_s * self.refresh_hz) + 1)
        mode_t = np.arange(mode_frames)[:, None] / self.refresh_hz
        mode_freqs = np.asarray(MODE_FREQUENCIES_HZ, dtype=float)[None, :]
        mode_luminance = .5 * (1 + np.sin(2 * np.pi * mode_t * mode_freqs))
        self.mode_rgb_frames = np.repeat((2 * mode_luminance - 1)[:, :, None], 3, axis=2)
        self.chat_rgb_frames = {}
        for page in (1, 2):
            frequencies = self._chatbox_page_frequencies(page)
            frames = max(1, math.ceil(cfg.stimulus_s * self.refresh_hz) + 1)
            t = np.arange(frames)[:, None] / self.refresh_hz
            freqs = np.asarray(frequencies, dtype=float)[None, :]
            luminance = .5 * (1 + np.sin(2 * np.pi * t * freqs))
            self.chat_rgb_frames[page] = np.repeat((2 * luminance - 1)[:, :, None], 3, axis=2)
        durations = {"free": cfg.stimulus_s, "cued": cfg.cued_stimulus_s}
        self.luminance_by_mode = {
            mode: (np.empty((0, len(TARGETS))) if self.static_only else
                   make_luminance_table(self.refresh_hz, duration))
            for mode, duration in durations.items()
        }
        self.rgb_frames_by_mode = {
            mode: np.repeat((2 * luminance - 1)[:, :, None], 3, axis=2)
            for mode, luminance in self.luminance_by_mode.items()
        }
        flips = []
        for _ in range(121):
            self._check_abort()
            self._draw_startup_check()
            flips.append(self.win.flip())
        check = frame_diagnostics(flips, self.refresh_hz, cfg.frame_long_factor,
                                  cfg.frame_short_factor, cfg.max_warning_frame_anomalies,
                                  cfg.frame_rate_tolerance)
        if check['hard_invalid']:
            rate_detail = (f"整段翻转均速{check['observed_flip_rate_hz']:.2f}Hz，"
                           f"基准{self.refresh_hz:.2f}Hz，"
                           f"偏差{check['flip_rate_error_percent']:+.2f}%"
                           f"（限值{check['flip_rate_tolerance_percent']:.2f}%）" if
                           check['observed_flip_rate_hz'] is not None else "整段翻转速率不可计算")
            raise RuntimeError("完整键盘静态绘制时序异常："
                               f"严重间隔{check['severe_interval_count']}，"
                               f"最大{check['max_interval_ms']:.2f}ms，"
                               f"{rate_detail}；请先检查显示/负载")
        if check['warning']:
            print("[显示时序警告] 静态绘制出现一个异常帧间隔；继续前请检查负载。", flush=True)
        framebuffer = getattr(self.win, "frameBufferSize", self.win.size)
        content_scale = self.win.getContentScaleFactor() if hasattr(self.win, 'getContentScaleFactor') else None
        self.display_info = {**layout,
            "window_size_reported": tuple(map(int, self.win.size)),
            "client_size_reported": tuple(map(int, self.win.clientSize)),
            "full_screen": bool(self.win.fullscr), "screen_index": cfg.screen_index,
            "window_backend": self.win.winType,
            "framebuffer_size_reported": tuple(map(int, framebuffer)), "content_scale_factor": content_scale,
            "refresh_hz_measured": self.refresh_hz,
            "cued_guidance": {"timing": "automatic_within_block", "cue_s": cfg.cue_s,
                              "settle_s": cfg.cued_settle_s, "arrow_width_pix_units": cue_width,
                              "dim_others_during_preparation": True, "static_outline_during_stimulus": True},
            "target_viewing_distance_cm": cfg.viewing_distance_cm,
            "declared_screen_diagonal_inches": cfg.screen_diagonal_inches,
            "stimulus_frames": len(self.rgb_frames_by_mode["cued"]),
            "scheduled_stimulus_s": len(self.rgb_frames_by_mode["cued"])/self.refresh_hz,
            "stimulus_by_mode": {
                mode: {"frames": len(frames), "scheduled_s": len(frames) / self.refresh_hz}
                for mode, frames in self.rgb_frames_by_mode.items()
            },
            "mode_selection": {
                "layout": "left_to_right",
                "labels": list(MODE_NAMES),
                "frequencies_hz": list(MODE_FREQUENCIES_HZ),
                "frames": len(self.mode_rgb_frames),
                "scheduled_s": len(self.mode_rgb_frames) / self.refresh_hz,
                "control": "ssvep",
            },
            "chatbox_keyboard": {
                "pages": {
                    str(page): {"layout": "custom", "symbols": list(symbols),
                                "frequencies_hz": list(frequencies),
                                "frames": len(self.chat_rgb_frames[page]),
                                "scheduled_s": len(self.chat_rgb_frames[page]) / self.refresh_hz}
                    for page, symbols, frequencies in (
                        (page, self._chatbox_page_symbols(page), self._chatbox_page_frequencies(page))
                        for page in (1, 2))
                },
                "layout": "chat_top_ime_keyboard_bottom",
                "backspace_action": "delete_last_character",
                "page_switch_actions": {"123": "numbers", "ABC": "letters"},
                "page2_arrow_action": "return_to_mode_menu",
                "send_action": "submit_chat_draft",
            },
            "static_render_check": check, "physical_timing_measured": False, "luminance_gamma_calibrated": False}

    def _apply_screen_layout(self, mode: str) -> None:
        """切换共享文字/输入框的位置与配色：模式三为上下分屏，其它模式为40键界面。"""
        self._screen_mode = mode
        layout = self._screen_layouts["chatbox" if mode == "chatbox" else "keyboard"]
        for name, values in layout.items():
            if name == "window":
                continue
            stim = getattr(self, name)
            for attr, value in values.items():
                setattr(stim, attr, value)
        self.win.color = layout["window"]["color"]
        self._output_max_chars = layout["window"]["output_max_chars"]
        self._sync_output_text(force=True)

    def _build_chatbox_page_visuals(self, width: float, height: float,
                                    ui_scale: float, menu_font: str) -> None:
        """Build the IME-style ChatBox keyboard in the lower half of the window.

        Keys are placed from CHATBOX_PAGE*_GRID in key-pitch units. Gaps are
        deliberately wide (about 40% of the pitch) so neighbouring flickers stay
        visually separated; the decoder's class order still follows
        CHATBOX_PAGE*_SYMBOLS, independent of screen position.
        """
        visual = self.visual
        # Keys take ~70% of the pitch: large enough to read and fixate, while the
        # remaining ~30% gap plus the frequency permutation keeps neighbours apart.
        keyboard_top, keyboard_bottom = height * .05, -height * .445
        pitch_y = (keyboard_top - keyboard_bottom) / 4
        pitch_x = min(width * .92 / 10.4, pitch_y * 1.7)
        key_h = pitch_y * .72
        key_w = min(pitch_x * .70, key_h * 1.25)
        page_layouts: dict[int, list[tuple[str, tuple[float, float], tuple[float, float]]]] = {}
        for page, grid in ((1, CHATBOX_PAGE1_GRID), (2, CHATBOX_PAGE2_GRID)):
            symbols = self._chatbox_page_symbols(page)
            page_layouts[page] = []
            for symbol in symbols:
                column, row, span = grid[symbol]
                position = (column * pitch_x, keyboard_top - (row + .5) * pitch_y)
                page_layouts[page].append((symbol, position, (key_w * span, key_h)))

        wake_y = keyboard_top - 1.5 * pitch_y
        self.chat_wake_square = visual.Rect(self.win, width=key_w * 3.2, height=key_h * 1.5,
            pos=(0, wake_y), fillColor=CHAT_SURFACE, lineColor=CHAT_SURFACE,
            units="pix", colorSpace="rgb", autoLog=False)
        self.chat_wake_label = visual.TextStim(self.win, text="WAKE", pos=(0, wake_y),
            height=key_h * .55, font=menu_font, color=CHAT_KEY_TEXT, bold=True, autoLog=False)
        self.chat_rest_text = visual.TextStim(self.win, text="", pos=(0, wake_y - key_h * 1.6),
            height=18 * ui_scale, wrapWidth=width * .8, font=menu_font, color=CHAT_MUTED,
            autoLog=False)

        self.chat_page_visuals: dict[int, list[dict[str, Any]]] = {}
        for page, layout_items in page_layouts.items():
            visuals = []
            for symbol, position, size in layout_items:
                border = visual.Rect(self.win, width=size[0] + 5 * ui_scale,
                    height=size[1] + 5 * ui_scale, pos=position,
                    fillColor=CHAT_KEY_BORDER, lineColor=CHAT_KEY_BORDER,
                    units="pix", colorSpace="rgb", autoLog=False)
                square = visual.Rect(self.win, width=size[0], height=size[1], pos=position,
                    fillColor=CHAT_SURFACE, lineColor=CHAT_SURFACE,
                    units="pix", colorSpace="rgb", autoLog=False)
                if symbol in CHATBOX_SUGGESTION_SYMBOLS:
                    self._suggestion_label_height = min(size[1] * .40, 34 * ui_scale)
                text_length = len(_chatbox_display_label(symbol))
                # Bold capitals are ~0.75 em wide; keep the whole word inside the key.
                label_height = (min(size[1] * .56, size[0] * .62) if text_length == 1 else
                                min(size[1] * .36, size[0] * 1.25 / text_length))
                label = visual.TextStim(self.win, text=_chatbox_display_label(symbol), pos=position,
                    height=label_height, wrapWidth=size[0] * .90, font=menu_font,
                    color=CHAT_KEY_TEXT, bold=True, autoLog=False)
                visuals.append({"symbol": symbol, "border": border, "square": square,
                                "label": label, "position": position, "size": size})
            self.chat_page_visuals[page] = visuals

    def _measure_refresh_rate(self) -> Optional[float]:
        """在静态键盘上测量刷新率，避免 PsychoPy 绘制独立的等待画面。"""
        previous_recording = self.win.recordFrameIntervals
        self.win.frameIntervals = []
        self.win.recordFrameIntervals = True
        try:
            for _ in range(180):
                self._check_abort()
                self._draw_startup_check()
                self.win.flip()
                if len(self.win.frameIntervals) >= 30:
                    recent = np.asarray(self.win.frameIntervals[-30:], dtype=float)
                    if np.all(np.isfinite(recent)) and np.std(recent) < .0005:
                        return float(1.0 / np.mean(recent))
        finally:
            self.win.recordFrameIntervals = previous_recording
            self.win.frameIntervals = []
        return None

    def _cancel_decode(self) -> None:
        """通知当前解码任务取消，并尝试取消尚未开始的任务。"""
        if self._active_decode_cancel is not None:
            self._active_decode_cancel.set()
        if self._decode_future is not None:
            self._decode_future.cancel()

    def _check_abort(self) -> None:
        """处理退出或暂停按键，并同步检查 EEG 采集健康状态。"""
        key_list = ["escape", "space"] if self._pause_enabled else ["escape"]
        keys = self.event.getKeys(keyList=key_list)
        if "escape" in keys:
            self.cancel.set()
            self._cancel_decode()
            raise AbortSession("用户按ESC停止")
        if self.cancel.is_set():
            self._cancel_decode()
            raise AbortSession("已取消实验")
        self.collector.check_health()
        # 闪烁期间记录帧间隔；此时不重建，避免破坏刺激时序，待闪烁结束后再重新排布。
        if not self.win.recordFrameIntervals:
            self._relayout_if_resized()
        if self._pause_enabled and "space" in keys:
            self._cancel_decode()
            raise PauseSelection("用户按空格暂停；尚未提交的本轮不会写入字符")

    def _sync_output_text(self, force: bool = False) -> None:
        """仅在文本变化或强制刷新时更新顶部输出框。"""
        text = self.ledger.typed_text
        if force or text != self._output_cache:
            rendered = output_display_text(text, self._output_max_chars)
            self.output_text.text = rendered
            self._output_cache = text

    def _draw_status(self) -> None:
        """绘制当前页面通用的页眉和页脚。"""
        self.header.draw()
        self.footer.draw()

    def _draw_chatbox_keyboard(self, page: int, *, flicker_rgb: Optional[np.ndarray] = None,
                               selected_index: Optional[int] = None) -> None:
        """绘制模式三：上半部分聊天记录与草稿框，下半部分 SSVEP 键盘。"""
        self._sync_output_text()
        self._draw_chat_area()
        self._draw_status()
        for index, item in enumerate(self.chat_page_visuals[page]):
            if flicker_rgb is None:
                fill = "#FFE45C" if index == selected_index else CHAT_SURFACE
                item["square"].fillColor = fill
                item["square"].lineColor = fill
            else:
                item["square"].fillColor = tuple(map(float, flicker_rgb[index, :3]))
            item["border"].fillColor = "#FFE45C" if index == selected_index else CHAT_KEY_BORDER
            item["border"].lineColor = item["border"].fillColor
            item["border"].draw()
            item["square"].draw()
            item["label"].draw()

    def _chatbox_page_symbols(self, page: int) -> tuple[str, ...]:
        """返回指定 ChatBox 页面的动作顺序，与视觉对象和解码器一致。"""
        if page == 2:
            return CHATBOX_PAGE2_SYMBOLS
        if self.cfg.chatbox_predictions_enabled:
            return CHATBOX_PAGE1_SYMBOLS
        return tuple(s for s in CHATBOX_PAGE1_SYMBOLS if s not in CHATBOX_SUGGESTION_SYMBOLS)

    def _chatbox_page_frequencies(self, page: int) -> tuple[float, ...]:
        """该页每个目标的刺激频率，顺序与 _chatbox_page_symbols 一致。"""
        table = CHATBOX_FREQUENCY_BY_SYMBOL[page]
        return tuple(float(table[symbol]) for symbol in self._chatbox_page_symbols(page))

    def _chatbox_global_class(self, symbol: str) -> Optional[int]:
        """将普通字母/数字/空格映射到原有40键目标，供账本和记录复用。"""
        for target in TARGETS:
            if target.symbol == symbol:
                return target.class_id
        return None

    def _persist_chatbox_selection(self, selection_id: int, symbol: str,
                                   result: dict[str, Any],
                                   rejected_reason: Optional[str] = None) -> TrialRecord:
        """记录 ChatBox 选择；导航动作和被拒识的轮次落为 rejected，不写入正文。"""
        record = TrialRecord(selection_id, 1, None, mode="free", start=self.clock())
        global_class = self._chatbox_global_class(symbol)
        if rejected_reason is not None:
            record.result = dict(result)
            record.status = "rejected"
            record.reason = f"ChatBox rejected ({symbol}): {rejected_reason}"
        elif global_class is not None:
            record.result = dict(result)
            record.result.update(trial_id=selection_id, prediction=global_class,
                                 fbcca_prediction=global_class, cca_prediction=global_class)
            self.ledger.apply_prediction(global_class)
            record.text_applied = True
            record.status = "valid"
        else:
            record.status = "rejected"
            record.reason = f"ChatBox action: {symbol}"
        record.end = self.clock()
        self._phase(record, "COMPLETE")
        self.ledger.commit(record)
        self._persist_trial(record)
        self._release_free_eeg(record)
        self._sync_output_text(force=True)
        return record

    def _present_and_decode(self, selection_id: int, frames: np.ndarray, onset_event: EventStamp,
                            draw_flicker, draw_static) -> dict[str, Any]:
        """呈现一轮闪烁（逐帧调用 draw_flicker），之后静止等待 FBCCA 解码结果。"""
        offset_event = EventStamp(onset_event.kind.replace("onset", "offset"), selection_id)
        self.win.recordFrameIntervals = True
        self.win.frameIntervals = []
        try:
            for frame_index, frame in enumerate(frames):
                self._check_abort()
                draw_flicker(frame)
                if frame_index == 0:
                    self.win.callOnFlip(self.markers.mark, onset_event)
                self.win.flip()
            self.win.callOnFlip(self.markers.mark, offset_event)
            self.win.flip()
        finally:
            self.win.recordFrameIntervals = False
        if onset_event.timestamp is None or offset_event.timestamp is None:
            raise RuntimeFault("DISPLAY", "ChatBox stimulus event marker missing")
        if self.markers.error is not None:
            raise RuntimeFault("LSL_CONNECTION", f"LSL Marker send failed: {self.markers.error}")
        self.header.text = "Analyzing EEG..."
        self.footer.text = "Analyzing EEG  ·  keep looking  ·  ESC: exit"
        self._active_decode_cancel = threading.Event()
        self._decode_future = self.executor.submit(
            decode_trial, selection_id, onset_event.timestamp, self.collector,
            self.cfg, self._active_decode_cancel)
        while (not self._decode_future.done()
               or self.clock() < offset_event.timestamp + self.cfg.blank_min_s):
            self._check_abort()
            self._poll_chat_reply()
            draw_static()
            self.win.flip()
        try:
            return self._decode_future.result()
        finally:
            self._decode_future = None
            self._active_decode_cancel = None

    def _chatbox_gate(self, result: dict[str, Any]) -> tuple[bool, Optional[dict[str, Any]]]:
        """复用自由输入的拒识门控（分数/分差/alpha）；阈值未标定时不拦截。"""
        if not (self.cfg.chatbox_rejection_enabled and self.cfg.rejection_enabled
                and self.cfg.rejection_thresholds_configured):
            return True, None
        diagnostics = self._rejection_diagnostics(result)
        return bool(diagnostics["accepted"]), diagnostics

    def _run_rest_mode(self, selection_id: int) -> int:
        """键盘休息：只有 WAKE 键闪烁；连续 chatbox_wake_rounds 轮注视 WAKE 才恢复输入。

        返回用掉的选择编号。照护者按空格也可直接唤醒。
        """
        from .decoding import FbccaDecoder
        self._speak("Keyboard resting. Look at wake to type again.")
        self.markers.mark(EventStamp("chatbox_rest_start", selection_id))
        refresh = self.refresh_hz
        n_frames = len(self.chat_rgb_frames[1])
        t = np.arange(n_frames)[:, None] / refresh
        luminance = .5 * (1 + np.sin(2 * np.pi * t * CHATBOX_WAKE_FREQUENCY_HZ))
        frames = np.repeat((2 * luminance - 1)[:, :, None], 3, axis=2)
        candidates = (CHATBOX_WAKE_FREQUENCY_HZ, *CHATBOX_WAKE_DECOYS_HZ)
        self.collector.decoder = FbccaDecoder(self.cfg, calibration=None,
                                              target_frequencies_hz=candidates)
        wins = 0
        needed = self.cfg.chatbox_wake_rounds
        self.event.getKeys(keyList=["space"])

        def draw(fill=None) -> None:
            self._sync_output_text()
            self._draw_chat_area()
            self._draw_status()
            color = CHAT_SURFACE if fill is None else tuple(map(float, fill[0, :3]))
            self.chat_wake_square.fillColor = self.chat_wake_square.lineColor = color
            self.chat_wake_square.draw()
            self.chat_wake_label.draw()
            self.chat_rest_text.draw()

        while True:
            self._poll_chat_reply()
            if self._unread_reply:
                self._read_reply_aloud()
            if self.event.getKeys(keyList=["space"]):
                break
            selection_id += 1
            self.chat_rest_text.text = (
                f"Look at WAKE to turn the keyboard on  ({wins}/{needed})\n"
                "Nothing is typed while the keyboard rests.")
            self.header.text = "Keyboard resting"
            self.footer.text = "Look at WAKE  ·  caregiver SPACE: wake now  ·  ESC: exit"
            onset = EventStamp("chatbox_wake_onset", selection_id,
                               details={"wake_hz": CHATBOX_WAKE_FREQUENCY_HZ,
                                        "decoys_hz": list(CHATBOX_WAKE_DECOYS_HZ)})
            try:
                result = self._present_and_decode(selection_id, frames, onset, draw, draw)
            except InvalidTrial:
                wins = 0
                continue
            accepted, diagnostics = self._chatbox_gate(result)
            woke = int(result["prediction"]) == 1 and accepted
            wins = wins + 1 if woke else 0
            self.markers.mark(EventStamp("chatbox_wake_result", selection_id, details={
                "prediction_hz": candidates[int(result["prediction"]) - 1], "accepted": accepted,
                "wake_vote": woke, "consecutive": wins,
                "max_score": diagnostics["max_score"] if diagnostics else None}))
            if wins >= needed:
                break
        self._resting = False
        self._rejected_in_a_row = 0
        self.markers.mark(EventStamp("chatbox_rest_end", selection_id))
        self._speak("Keyboard on.")
        self.chat_status.text = "Keyboard on"
        return selection_id

    def _run_chatbox_selection(self, page: int, selection_id: int) -> tuple[str, dict[str, Any]]:
        """呈现一页 ChatBox SSVEP 刺激并返回页面内动作及解码结果。"""
        symbols = self._chatbox_page_symbols(page)
        frames = self.chat_rgb_frames[page]
        self.header.text = ("LETTERS  ·  look at a key" if page == 1
                            else "NUMBERS  ·  look at a key")
        self.footer.text = ("Top row: word suggestions  ·  123: numbers  ·  DEL: delete  ·  "
                            "SEND: send to DeepSeek  ·  ESC: exit"
                            if page == 1 else
                            "ABC: letters  ·  REST: keyboard off  ·  REPLAY: hear last reply  ·  "
                            "MENU: mode menu  ·  ESC: exit")
        onset_event = EventStamp("chatbox_stimulus_onset", selection_id,
                                 details={"page": page, "symbols": list(symbols),
                                          "frequencies_hz": list(
                                              self._chatbox_page_frequencies(page)),
                                          "suggestions": (list(self._suggestions) if page == 1
                                                          else [])})
        result = self._present_and_decode(
            selection_id, frames, onset_event,
            lambda frame: self._draw_chatbox_keyboard(page, flicker_rgb=frame),
            lambda: self._draw_chatbox_keyboard(page))
        prediction = int(result["prediction"])
        if not 1 <= prediction <= len(symbols):
            raise InvalidTrial(f"ChatBox prediction out of range: {prediction}")
        return symbols[prediction - 1], result

    def _draw_processing_status(self, record: TrialRecord) -> None:
        """绘制等待 EEG 解码结果的状态页。"""
        self._draw_status()
        if record.true_class is not None:
            self.processing_key.draw()
            self.processing_key_label.draw()
        self.processing_dot.draw()
        self.processing_title.draw()
        self.processing_subtitle.draw()

    def _draw_cued_result_feedback(self) -> None:
        """绘制提示测试的识别结果和可选警告。"""
        self._draw_status()
        self.result_verdict.draw()
        self.result_pair.draw()
        self.result_frequency.draw()
        self.result_divider.draw()
        self.result_details.draw()
        if self.result_warning.text:
            self.result_warning.draw()

    def _prepare_free_pause_display(self, message: str) -> None:
        """设置自由输入暂停页的文字内容。"""
        self.header.text = "Free spelling"
        self.footer.text = "SPACE or ENTER: resume   ·   ESC: exit"
        self.pause_subtitle.text = message

    def _draw_free_pause_display(self) -> None:
        """绘制自由输入暂停页。"""
        self._draw_status()
        self.pause_title.draw()
        self.pause_subtitle.draw()

    def _chat_layout(self, messages: Optional[Sequence[ChatMessage]] = None
                     ) -> tuple[list[str], list[tuple[int, int]]]:
        """折行后的对话行，以及最后一条消息每行的 (行号, 正文字符下标)。"""
        lines: list[str] = []
        spans: list[tuple[int, int]] = []
        for message in (self._chat_history if messages is None else messages)[-8:]:
            label = "YOU" if message.role == "user" else "DEEPSEEK"
            prefix = f"{label}:  "
            if lines:
                lines.append("")
            spans = []
            for text, start in wrap_with_offsets(prefix + plain_chat_text(message.content),
                                                 self._chat_wrap_chars):
                spans.append((len(lines), max(0, start - len(prefix))))
                lines.append(text)
        return lines, spans

    def _show_chat_lines(self, lines: list[str], offset: int) -> None:
        """显示从 offset 开始、能放进对话面板的若干行，并提示上方/下方是否还有内容。"""
        visible = lines[offset:offset + self._chat_max_lines]
        while visible and not visible[0]:
            visible = visible[1:]
        self.chat_history_text.text = "\n".join(visible)
        below = offset + self._chat_max_lines < len(lines)
        self.chat_scroll_hint.text = ("MORE BELOW  v" if below else
                                      "^  EARLIER ABOVE" if offset > 0 else "")

    def _format_chat_history(self) -> str:
        """显示对话末尾（最新内容）；只限制显示区域，不截断发送给 API 的完整历史。"""
        if not self._chat_history:
            self.chat_scroll_hint.text = ""
            return "No messages yet. Spell a message below and select SEND."
        lines, _ = self._chat_layout()
        self._show_chat_lines(lines, max(0, len(lines) - self._chat_max_lines))
        return self.chat_history_text.text

    def _speak(self, text: str) -> None:
        """语音提示；语音不可用时静默跳过。"""
        try:
            self._speaker.speak(text)
        except Exception as exc:  # noqa: BLE001 - voice must never stop the BCI session
            print(f"[语音] {exc}", file=sys.stderr, flush=True)
            self._speaker = Speaker()

    def _read_reply_aloud(self) -> None:
        """朗读最新回复，聊天区随朗读位置缓慢向下滚动；读完自动恢复 SSVEP 输入。

        朗读期间键盘保持静止（不闪烁），用户只需看和听，无需任何手部操作。
        照护者可按空格跳过，ESC 仍为退出。
        """
        self._unread_reply = False
        last = max((index for index, message in enumerate(self._chat_history)
                    if message.role == "assistant"), default=None)
        if last is None:
            self._speak("No reply yet.")
            return
        # REPLAY while a newer question is pending reads the latest reply that exists.
        spoken = plain_chat_text(self._chat_history[last].content)
        lines, spans = self._chat_layout(self._chat_history[:last + 1])
        if not spoken or not spans:
            return
        page_lines = self._chat_max_lines
        max_offset = max(0, len(lines) - page_lines)
        # Start with the end of the user's question visible above the reply.
        first_offset = min(max(0, spans[0][0] - 2), max_offset)
        self._speak(spoken)
        voice_progress = self._speaker.available
        self.header.text = "Reading reply aloud"
        self.footer.text = ("Typing resumes automatically after the reply  ·  "
                            "SPACE: skip  ·  ESC: exit")
        self.chat_status.text = "Reading reply..."
        self.chat_status.color = "#FFE45C"
        self.event.getKeys(keyList=["space"])
        started = self.clock()
        finished_at: Optional[float] = None
        offset = -1
        chars_per_s = self.cfg.chatbox_reading_chars_per_s
        try:
            while True:
                self._check_abort()
                now = self.clock()
                if self.event.getKeys(keyList=["space"]):
                    break
                position = self._speaker.progress() if voice_progress else None
                if position is None:
                    position = int((now - started) * chars_per_s)
                if finished_at is None:
                    done = (not self._speaker.is_speaking() if voice_progress
                            else position >= len(spoken))
                    # Guard against a stalled voice backend.
                    if done or now - started > len(spoken) / 4 + 15:
                        finished_at = now
                        position = len(spoken)
                else:
                    position = len(spoken)
                spoken_line = spans[0][0]
                for line_index, char_start in spans:
                    if char_start <= position:
                        spoken_line = line_index
                # Keep the line being read about 40% down the panel.
                target = min(max(spoken_line - int(page_lines * .4), first_offset), max_offset)
                if finished_at is not None:
                    target = max_offset
                if target != offset:
                    offset = target
                    self._show_chat_lines(lines, offset)
                self._draw_chatbox_keyboard(self._chat_page)
                self.win.flip()
                if finished_at is not None and now - finished_at >= self.cfg.chatbox_reply_hold_s:
                    break
        finally:
            self._speaker.stop()
            self.chat_history_text.text = self._format_chat_history()
            self.chat_status.text = "Reply read  ·  keep typing"
        # Tell the user the keyboard is about to flicker again.
        self._speak("Your turn.")

    def _wait_for_chat_reply(self) -> None:
        """发送后键盘保持静止（不闪烁），直到回复到达，避免等待期间误选字符。

        回复到达后由 _read_reply_aloud 朗读，读完再恢复闪烁输入。照护者可按空格提前恢复输入。
        """
        self.header.text = "Waiting for reply"
        self.footer.text = ("Keyboard paused until the reply has been read  ·  "
                            "SPACE: resume typing  ·  ESC: exit")
        self.event.getKeys(keyList=["space"])
        while self._chat_future is not None and not self._chat_future.done():
            self._check_abort()
            if self.event.getKeys(keyList=["space"]):
                self.chat_status.text = "Typing resumed  ·  reply will be read when it arrives"
                self._typing_during_reply = True
                return
            self._draw_chatbox_keyboard(self._chat_page)
            self.win.flip()

    def _draw_chat_area(self) -> None:
        """绘制上半屏的对话面板（不含闪烁）。"""
        self.chat_panel.draw()
        self.chat_title.draw()
        self.chat_status.draw()
        self.chat_history_caption.draw()
        self.chat_history_rule.draw()
        self.chat_history_text.draw()
        if self.chat_scroll_hint.text:
            self.chat_scroll_hint.draw()

    def _start_chat_request(self, text: str) -> bool:
        """在独立线程中把草稿发送给 DeepSeek；SSVEP 输入不等待回复，继续进行。"""
        text = text.strip()
        if not text:
            return False
        if self._chat_future is not None:
            self.chat_status.text = "Still waiting for the previous reply - draft kept"
            self.chat_status.color = CHAT_WARNING
            return False
        try:
            if self._chat_backend is None:
                self._chat_backend = DeepSeekChatBackend(
                    model=self.cfg.deepseek_model,
                    endpoint=self.cfg.deepseek_api_url,
                    timeout=self.cfg.deepseek_timeout_s,
                    api_key_env=self.cfg.deepseek_api_key_env,
                    api_key_file=self.cfg.deepseek_api_key_file)
        except (ChatBackendError, ValueError) as exc:
            self.chat_status.text = f"DeepSeek unavailable: {exc}"[:90]
            self.chat_status.color = CHAT_WARNING
            print(f"[DeepSeek] {exc}", file=sys.stderr, flush=True)
            return False
        request_history = list(self._chat_history)[-20:]
        if self.cfg.deepseek_system_prompt:
            request_history.insert(0, ChatMessage("system", self.cfg.deepseek_system_prompt))
        self._chat_history.append(ChatMessage("user", text))
        self._predictor.learn(text, weight=3)
        self.chat_history_text.text = self._format_chat_history()
        self.chat_status.text = "DeepSeek is replying...  (keyboard paused)"
        self.chat_status.color = "#FFE45C"
        self._speak("Message sent.")
        self._chat_future = self.chat_executor.submit(
            self._chat_backend.send, text, history=request_history)
        return True

    def _poll_chat_reply(self) -> None:
        """在两次刺激之间检查回复；不在闪烁帧中更新文本，避免掉帧。"""
        future = self._chat_future
        if future is None or not future.done():
            return
        self._chat_future = None
        self._typing_during_reply = False
        try:
            answer = future.result()
        except (ChatBackendError, ValueError) as exc:
            self.chat_status.text = f"DeepSeek request failed: {exc}"[:90]
            self.chat_status.color = CHAT_WARNING
            print(f"[DeepSeek] {exc}", file=sys.stderr, flush=True)
            self._speak("The reply failed. Your message was not answered.")
            return
        self._chat_history.append(ChatMessage("assistant", answer))
        self._predictor.learn(plain_chat_text(answer))
        self.chat_history_text.text = self._format_chat_history()
        self.chat_status.text = "Reply received"
        self.chat_status.color = "#FFE45C"
        # Read at the next gap between selections, never during a flicker round.
        self._unread_reply = True

    def _submit_chat_draft(self) -> None:
        """后台发送当前草稿；成功提交后清空输入框，键盘立即继续闪烁输入。"""
        if self._start_chat_request(self.ledger.typed_text):
            self.ledger.typed_text = ""
            self._sync_output_text(force=True)
            self.message.text = ""

    def _prepare_cued_rest_display(self, block_id: int, completed_in_block: int) -> None:
        """设置提示测试定时休息页的进度文字。"""
        self.header.text = (f"Block {block_id}/{self.cfg.blocks}  ·  "
                            f"{completed_in_block}/{len(TARGETS)} targets complete")
        self.footer.text = "No flicker  ·  ESC: stop"

    def _draw_cued_rest_display(self) -> None:
        """绘制提示测试定时休息页。"""
        self._draw_status()
        self.rest_title.draw()
        self.rest_timer.draw()
        self.rest_remaining_label.draw()
        self.rest_auto_label.draw()

    def _prepare_cued_result_feedback(self, record: TrialRecord, result: dict[str, Any],
                                      target: Target, mains: dict[str, Any]) -> None:
        """根据真实目标和解码结果设置提示测试反馈内容。"""
        pred = TARGETS[result['prediction'] - 1]
        fb_pred = TARGETS[result['fbcca_prediction'] - 1]
        cca_pred = TARGETS[result['cca_prediction'] - 1]
        correct = pred.class_id == target.class_id
        self.header.text = f"Trial {record.trial_id} of {self.cfg.blocks*len(TARGETS)}  ·  Result"
        self.footer.text = "Continuing automatically  ·  ESC: stop"
        self.result_verdict.text = "CORRECT" if correct else "WRONG"
        self.result_verdict.color = "#92E5B6" if correct else "#FF9A94"
        self.result_pair.text = f"Target {target.symbol}   →   Recognized {pred.symbol}"
        self.result_frequency.text = (f"{target.frequency_hz:.1f} Hz" if correct else
            f"Target {target.frequency_hz:.1f} Hz  ·  Recognized {pred.frequency_hz:.1f} Hz")
        self.result_details.text = (f"FBCCA  {fb_pred.symbol}  ({fb_pred.frequency_hz:.1f} Hz)\n"
                                    f"CCA  {cca_pred.symbol}  ({cca_pred.frequency_hz:.1f} Hz)")
        self.result_warning.text = ("High mains interference: check electrode contact" if
                                    mains.get("line_dominated") else "")

    def _show_cued_result_feedback(self, record: TrialRecord, result: dict[str, Any],
                                   target: Target, mains: dict[str, Any]) -> None:
        """在配置的反馈时长内持续显示提示测试结果。"""
        self._prepare_cued_result_feedback(record, result, target, mains)
        started = self.clock()
        self.win.recordFrameIntervals = False
        while True:
            self._check_abort()
            self._draw_cued_result_feedback()
            self.win.flip()
            if self.clock() - started >= self.cfg.feedback_s:
                return

    def _prepare_processing_status(self, record: TrialRecord) -> None:
        """根据自由输入或提示测试模式准备处理状态页。"""
        free = record.mode == "free"
        self.header.text = (f"Selection {record.trial_id}" if free else
                            f"Trial {record.trial_id} of {self.cfg.blocks*len(TARGETS)}")
        self.footer.text = "SPACE: pause  ·  ESC: stop" if free else "ESC: stop"
        target = None if free else TARGETS[record.true_class - 1]
        self.processing_key_label.text = ("" if target is None else
                                          target.display_label or "SPACE")
        self.processing_key_label.height = (self._processing_key_size*.25 if
                                            target is not None and target.symbol == "SPACE" else
                                            self._processing_key_size*.43)
        positions = (self._processing_free_positions if free else
                     self._processing_cued_positions)
        self.processing_dot.pos, self.processing_title.pos, self.processing_subtitle.pos = positions

    def _draw_startup_check(self) -> None:
        """绘制无闪烁的启动显示检查页。"""
        self._draw_keys()
        self.startup_dot.draw()
        self.startup_title.draw()
        self.startup_subtitle.draw()

    def _set_cue_label_target(self, target_id: Optional[int]) -> None:
        """仅在目标变化时更新键帽文字的提示配色。"""
        if target_id == self._label_cue_target_id:
            return
        for index, label in enumerate(self.labels, start=1):
            label.color = (self._static_label_color if target_id is None or index == target_id
                           else self._cue_label_color)
        self._label_cue_target_id = target_id

    def _draw_keys(self, target_id: Optional[int] = None, flicker_rgb: Optional[np.ndarray] = None,
                   show_outline: bool = False, dim_others: bool = False) -> None:
        """绘制静态或闪烁键盘，并可突出显示指定目标。"""
        if self.static_only and flicker_rgb is not None:
            raise RuntimeFault("DISPLAY", "静态自检禁止呈现闪烁帧")
        self._set_cue_label_target(target_id if dim_others and flicker_rgb is None else None)
        if flicker_rgb is not None:
            self.squares.colors = flicker_rgb
        elif dim_others and target_id is not None:
            self.squares.colors = self._cue_colors[target_id - 1]
        else:
            self.squares.colors = self._static_colors
        self.key_borders.draw()
        self.squares.draw()
        for label in self.labels:
            label.draw()
        if target_id is not None:
            pos = self.positions[target_id - 1]
            if self._show_cue_arrow:
                self.triangle.pos = (pos[0], pos[1] - self.key_size / 2 - self.cue_offset)
                self.triangle.draw()
            if show_outline:
                self.outline_shadow.pos = pos
                self.outline_shadow.draw()
                self.outline.pos = pos
                self.outline.draw()
        if target_id is not None:
            self.cue_title.draw()
        else:
            self.header.draw()
        self.footer.draw()

    def _prepare_cued_trial(self, record: TrialRecord, cue_event: EventStamp) -> None:
        """静态定位 → 定时稳定注视 → 闪烁；全程允许ESC和流健康检查。"""
        target = TARGETS[record.true_class - 1]
        progress = f"Block {record.block_id}/{self.cfg.blocks} | Trial {record.trial_id}/{self.cfg.blocks*len(TARGETS)}"
        self.cue_title.text = f"TARGET: {target.symbol}    |    ROW {target.row+1}   COL {target.col+1}"
        self.footer.text = f"{progress} | Find the highlighted key ({self.cfg.cue_s:g} s) | ESC: stop"
        self.event.getKeys(keyList=["space"])
        self._phase(record, "CUE")
        self._check_abort()
        self._draw_keys(target.class_id, show_outline=True, dim_others=True)
        self.win.callOnFlip(self.markers.mark, cue_event)
        self.win.flip()
        record.cue_onset = cue_event.timestamp
        while self.clock() - record.cue_onset < self.cfg.cue_s:
            self._check_abort()
            self.event.getKeys(keyList=["space"])
            self._draw_keys(target.class_id, show_outline=True, dim_others=True)
            self.win.flip()
        self._phase(record, "FIXATE")
        self.footer.text = f"{progress} | Keep looking at the key center ({self.cfg.cued_settle_s:g} s) | ESC: stop"
        fixation = EventStamp("fixation_onset", record.trial_id, target.class_id,
                              {"settle_s": self.cfg.cued_settle_s,
                               "trigger": "automatic_after_cue"})
        self._check_abort()
        self._draw_keys(target.class_id, show_outline=True, dim_others=True)
        self.win.callOnFlip(self.markers.mark, fixation)
        self.win.flip()
        record.fixation_onset = fixation.timestamp
        while self.clock() - record.fixation_onset < self.cfg.cued_settle_s:
            self._check_abort()
            self.event.getKeys(keyList=["space"])
            self._draw_keys(target.class_id, show_outline=True, dim_others=True)
            self.win.flip()
        self._set_cue_label_target(None)
        self.footer.text = f"{progress} | Keep looking at the key center | ESC: stop"

    def show_message(self, text: str, min_duration_s: float = 0.0, wait_space: bool = False) -> None:
        """显示消息至少指定时长，并可等待用户按空格继续。"""
        self.message.text = text
        self._sync_output_text()
        if wait_space:
            self.event.getKeys(keyList=["space"])
        started = self.clock()
        self.win.recordFrameIntervals = False
        while True:
            self._check_abort()
            self._draw_status()
            self.message.draw()
            self.win.flip()
            if self.clock() - started >= min_duration_s:
                if not wait_space or "space" in self.event.getKeys(keyList=["space"]):
                    return

    def _phase(self, record: TrialRecord, name: str) -> None:
        """将当前 LSL 时钟时间记录为试次阶段切换点。"""
        record.phase_times.append((name, self.clock()))

    def _persist_trial(self, record: TrialRecord) -> None:
        """在启用记录器时立即保存已结束试次。"""
        if self.recorder is None:
            return
        try:
            self.recorder.write_trial(record)
        except Exception as exc:
            raise RuntimeFault("SAVE", f"逐试次保存失败；保留 {self.recorder.session_dir}：{exc}") from exc

    def _note_fault(self, exc: BaseException, *, fatal: bool, default: str = "DISPLAY") -> None:
        """登记并打印试次故障，必要时标记为会话致命错误。"""
        fault = fault_record(exc, default)
        self.trial_faults.append(fault)
        if fatal:
            self.fatal_fault = fault
        print_fault(fault)

    @staticmethod
    def _attach_failed_epoch(record: TrialRecord, exc: BaseException) -> None:
        """将异常携带的部分事件窗附加到试次，以便失败后诊断。"""
        epoch = getattr(exc, "epoch", None)
        if epoch is not None and record.result is None:
            record.result = {"trial_id": record.trial_id, "epoch": epoch,
                             "quality": epoch.diagnostics.get("quality")}

    def _rejection_diagnostics(self, result: dict[str, Any]) -> dict[str, Any]:
        """评估自由输入证据门控，但不把分类分数解释为概率。"""
        cfg = self.cfg
        min_score = (None if cfg.rejection_min_score is None
                     else float(cfg.rejection_min_score))
        min_margin = (None if cfg.rejection_min_margin is None
                      else float(cfg.rejection_min_margin))
        max_alpha_ratio = (None if cfg.rejection_max_alpha_ratio is None
                           else float(cfg.rejection_max_alpha_ratio))
        configured = cfg.rejection_thresholds_configured
        max_score: Optional[float] = None
        score_margin: Optional[float] = None
        alpha_ratio: Optional[float] = None
        reason_codes: list[str] = []
        reasons: list[str] = []

        if cfg.rejection_enabled and result.get("calibration_fallback"):
            reason_codes.append("calibration_fallback")
            reasons.append("个人模板回退，当前分数不适用个人模型的拒识阈值")

        if cfg.rejection_enabled and not configured:
            reason_codes.append("thresholds_unconfigured")
            reasons.append(
                "拒识阈值未完整配置；请先用明确注视目标和不打算输入的真实EEG数据标定，"
                "再用独立数据验证")

        try:
            scores = np.asarray(result.get("scores", []), dtype=float).ravel()
        except (TypeError, ValueError):
            scores = np.empty(0, dtype=float)
        if scores.size == 0 or not np.isfinite(scores).all():
            reason_codes.append("scores_missing_or_nonfinite")
            reasons.append("分类分数缺失或非有限")
        else:
            ordered = np.sort(scores)
            max_score = float(ordered[-1])
            if ordered.size < 2:
                reason_codes.append("insufficient_scores")
                reasons.append("候选分数少于两个，无法计算第一/第二名分差")
            else:
                score_margin = float(ordered[-1] - ordered[-2])
                if configured and max_score < float(min_score):
                    reason_codes.append("score_below_threshold")
                    reasons.append(
                        f"最高分 {max_score:.3f} < 阈值 {float(min_score):.3f}")
                if configured and score_margin < float(min_margin):
                    reason_codes.append("margin_below_threshold")
                    reasons.append(
                        f"第一/第二名分差 {score_margin:.3f} < 阈值 {float(min_margin):.3f}")
            if cfg.rejection_enabled:
                epoch = result.get("epoch")
                eeg = getattr(epoch, "data", None)
                fs = getattr(epoch, "fs", None)
                if eeg is None or fs is None:
                    reason_codes.append("intent_features_missing")
                    reasons.append("缺少意图门控所需的原始 EEG 窗")
                else:
                    try:
                        alpha_ratio = alpha_power_ratio(eeg, float(fs))
                    except (ValueError, DegenerateSignalError) as exc:
                        reason_codes.append("intent_features_invalid")
                        reasons.append(f"意图门控特征无效：{exc}")
                    else:
                        if configured and alpha_ratio > float(max_alpha_ratio):
                            reason_codes.append("alpha_ratio_above_threshold")
                            reasons.append(
                                f"alpha 相对能量 {alpha_ratio:.3f} > 阈值 {float(max_alpha_ratio):.3f}")

        intent_pass = bool(
            not cfg.rejection_enabled
            or (alpha_ratio is not None and max_alpha_ratio is not None
                and alpha_ratio <= float(max_alpha_ratio)))

        accepted = bool(
            max_score is not None
            and score_margin is not None
            and (not cfg.rejection_enabled or (
                configured
                and max_score >= float(min_score)
                and score_margin >= float(min_margin)
                and intent_pass))
            and not reason_codes
        )
        if accepted:
            reason_codes = []
            reason = None
            reason_code = None
        else:
            reason_code = "+".join(reason_codes) or "rejected"
            reason = "拒识： " + "; ".join(reasons)
        return {
            "enabled": cfg.rejection_enabled,
            "configured": configured,
            "accepted": accepted,
            "max_score": max_score,
            "score_margin": score_margin,
            "alpha_ratio": alpha_ratio,
            "min_score": min_score,
            "min_margin": min_margin,
            "max_alpha_ratio": max_alpha_ratio,
            "reason_code": reason_code,
            "reason_codes": reason_codes,
            "reason": reason,
        }

    def _rejection_diagnostics_for_mode(self, mode: str,
                                        result: dict[str, Any]) -> Optional[dict[str, Any]]:
        """仅为自由输入模式计算拒识诊断，并按配置决定是否应用阈值。"""
        return self._rejection_diagnostics(result) if mode == "free" else None

    @staticmethod
    def _rejection_calibration_notice() -> str:
        """返回自由输入拒识阈值尚未标定时的安全提示。"""
        return (
            "Rejection thresholds are not configured.\n\n"
            "Before FREE INPUT can add a character, space, or delete action, "
            "calibrate score, margin, and alpha-ratio thresholds with real EEG while looking at an intended "
            "target and while not intending to enter text, then verify them on "
            "independent data.\n\n"
            "FBCCA scores are not probabilities. This gate is not validated "
            "automatic idle detection; no text will be entered until it is configured."
        )

    def _blank_until(self, when: float) -> None:
        """保持静态状态页，直到指定的 LSL 时钟时间。"""
        while self.clock() < when:
            self._check_abort()
            self._draw_status()
            self.win.flip()

    def _present_trial_stimulus(self, record: TrialRecord, cue_event: EventStamp,
                                onset_event: EventStamp, offset_event: EventStamp) -> None:
        """呈现准备和闪烁阶段，并记录事件、逐帧时间及显示诊断。"""
        cfg = self.cfg
        free = record.mode == "free"
        target_id = record.true_class
        if free:
            self._phase(record, "READY")
            for frame in range(max(1, round(cfg.free_prepare_s * self.refresh_hz))):
                self._check_abort()
                self._draw_keys()
                if frame == 0:
                    self.win.callOnFlip(self.markers.mark, cue_event)
                self.win.flip()
            record.cue_onset = cue_event.timestamp
            self.header.text = f"FREE | Selection {record.trial_id} | Keep looking at your chosen character"
        else:
            self._prepare_cued_trial(record, cue_event)
        self._phase(record, "STIMULATE")
        self.win.frameIntervals = []
        self.win.recordFrameIntervals = True
        try:
            stimulus_frames = self.rgb_frames_by_mode["free" if free else "cued"]
            for frame, rgb in enumerate(stimulus_frames):
                self._check_abort()
                self._draw_keys(target_id, rgb, show_outline=not free)
                if frame == 0:
                    self.win.callOnFlip(self.markers.mark, onset_event)
                self.win.callOnFlip(lambda: record.frame_lsl_times.append(self.clock()))
                flip_time = self.win.flip()
                record.frame_flip_times.append(float(flip_time))
                if frame == 0:
                    record.stimulus_onset = onset_event.timestamp
            self.win.callOnFlip(self.markers.mark, offset_event)
            self.win.callOnFlip(lambda: record.frame_lsl_times.append(self.clock()))
            offset_flip = self.win.flip()
            record.frame_flip_times.append(float(offset_flip))
        finally:
            self.win.recordFrameIntervals = False
        record.stimulus_offset = offset_event.timestamp
        if onset_event.timestamp is None or offset_event.timestamp is None:
            raise InvalidTrial("刺激事件标记缺失")
        record.actual_stimulus_s = offset_event.timestamp - onset_event.timestamp
        record.requested_window_start = onset_event.timestamp + cfg.response_delay_s
        record.requested_window_end = record.requested_window_start + cfg.window_s
        record.frame_diagnostics = frame_diagnostics(
            record.frame_flip_times, self.refresh_hz, cfg.frame_long_factor,
            cfg.frame_short_factor, cfg.max_warning_frame_anomalies,
            cfg.frame_rate_tolerance)
        record.frame_diagnostics['psychopy_frame_intervals_s'] = np.asarray(self.win.frameIntervals).copy()

    def _await_trial_result(self, record: TrialRecord, onset_event: EventStamp,
                            offset_event: EventStamp) -> dict[str, Any]:
        """异步等待分类及最短空白期结束，并校验返回结果归属。"""
        cfg = self.cfg
        self._decode_future = self.executor.submit(decode_trial, record.trial_id, onset_event.timestamp,
                                                   self.collector, cfg, self._active_decode_cancel)
        while not self._decode_future.done() or self.clock() < offset_event.timestamp + cfg.blank_min_s:
            self._check_abort()
            if self.clock() < offset_event.timestamp + cfg.blank_min_s:
                self._draw_status()
            else:
                self._draw_processing_status(record)
            self.win.flip()
        self._check_abort()
        result = self._decode_future.result()
        if result['trial_id'] != record.trial_id:
            raise InvalidTrial("收到属于其他试次的分类结果，拒绝提交")
        if self.markers.error is not None:
            raise RuntimeFault("LSL_CONNECTION", f"LSL Marker发送失败：{self.markers.error}")
        if (not 1 <= result['prediction'] <= len(TARGETS)
                or not 1 <= result['fbcca_prediction'] <= len(TARGETS)
                or not 1 <= result['cca_prediction'] <= len(TARGETS)):
            raise InvalidTrial("预测类别超出40目标范围")
        return result

    def _apply_trial_result(self, record: TrialRecord, result: dict[str, Any],
                            target: Optional[Target]) -> None:
        """执行拒识门控、写入字符并显示与模式匹配的反馈。"""
        free = record.mode == "free"
        record.result = result
        mains = result.get("quality", {}).get("mains_noise", {})
        if mains.get("line_dominated"):
            print(f"[试次{record.trial_id} 工频提示] " + "; ".join(mains["warnings"]), flush=True)
        rejection_diagnostics = self._rejection_diagnostics_for_mode(record.mode, result)
        if rejection_diagnostics is not None:
            result["rejection_diagnostics"] = rejection_diagnostics
        if rejection_diagnostics is not None and not rejection_diagnostics["accepted"]:
            rejection_reason = rejection_diagnostics["reason"]
            record.status = "rejected"
            record.reason = rejection_reason
            self._phase(record, "FEEDBACK")
            if free:
                # 拒识是后台门控结果；自由输入保持键盘页，不切到失败提示页。
                # 试次仍会落盘为 rejected，且不会写入任何字符。
                self.message.text = ""
                self.header.text = "FREE | Keep looking at any character"
                self.footer.text = "SPACE: pause  ·  ESC: stop"
                return
            return
        if self.wheelchair is not None:
            direction = DEFAULT_DIRECTION_BY_CLASS.get(int(result['prediction']))
            if direction is not None:
                try:
                    self.wheelchair.move(direction)
                    print(f"[轮椅] class={result['prediction']} -> {direction}", flush=True)
                except Exception as exc:
                    # 轮椅网络故障不能悄悄变成继续运动；发送STOP后结束本试次。
                    try:
                        self.wheelchair.stop(disarm=True)
                    except Exception:
                        pass
                    raise RuntimeError(f"轮椅动作发送失败：{exc}") from exc
        self.ledger.apply_prediction(result['prediction'])
        record.text_applied = True
        record.status = "valid"
        self._sync_output_text()
        self._phase(record, "FEEDBACK")
        pred = TARGETS[result['prediction'] - 1]
        if free:
            self.header.text = f"FREE | Selection {record.trial_id} | Recognized: {pred.symbol}"
            feedback = (f"Recognized: {pred.symbol} ({pred.frequency_hz:.1f} Hz)\n\n"
                        "Choose your next character in the next round.\n"
                        "Keyboard SPACE: pause   ESC: exit")
        else:
            self._show_cued_result_feedback(record, result, target, mains)
            return
        if mains.get("line_dominated"):
            feedback += "\nHigh mains interference: check electrode / reference / BIAS contact."
        self.show_message(feedback, self.cfg.feedback_s)

    def _run_trial(self, record: TrialRecord) -> None:
        """运行单个试次的准备、刺激、时序质控、解码和反馈流程。"""
        cfg = self.cfg
        self.collector.check_health()
        self._graphics_gc.collect(self.win)
        free = record.mode == "free"
        if free and record.true_class is not None:
            raise ValueError("自由输入不应有预设目标")
        target = None if free else TARGETS[record.true_class - 1]
        target_id = None if target is None else target.class_id
        if self._decode_future is not None and not self._decode_future.done():
            raise RuntimeError("前一次分类尚未退出；请暂停后重试")
        self._decode_future = None
        self._active_decode_cancel = threading.Event()
        self._phase(record, "PREPARE")
        if free:
            self.header.text = f"FREE | Selection {record.trial_id} | Choose your next character"
            self.footer.text = "Keyboard SPACE: pause   ·   ESC: exit   ·   <-: delete"
        else:
            self.header.text = f"Block {record.block_id}/{cfg.blocks} | Trial {record.trial_id}/{cfg.blocks*40} | Look at {target.symbol}"
            self.footer.text = "Follow the highlighted key; trials advance automatically | ESC: stop"
        self._sync_output_text()
        details = {"mode": record.mode, "window_s": cfg.window_s,
                   "response_delay_s": cfg.response_delay_s,
                   "minimum_stimulus_s": (cfg.stimulus_s if free else cfg.cued_stimulus_s),
                   "continuous_eeg_saved_from_stimulus_onset": True}
        if not free:
            details["offline_max_delay_s"] = cfg.cued_offline_max_delay_s
        if target is not None:
            details["frequency_hz"] = target.frequency_hz
        cue_event = EventStamp("ready_onset" if free else "cue_onset", record.trial_id, target_id,
            {} if free else {"cue_s": cfg.cue_s, "progression": "automatic",
                             "settle_s": cfg.cued_settle_s})
        onset_event = EventStamp("stimulus_onset", record.trial_id, target_id, details)
        offset_event = EventStamp("stimulus_offset", record.trial_id, target_id)
        self._present_trial_stimulus(record, cue_event, onset_event, offset_event)
        self._phase(record, "BLANK")
        self._prepare_processing_status(record)
        if record.requested_window_end > record.stimulus_offset + 1e-6:
            self._blank_until(offset_event.timestamp + cfg.blank_min_s)
            raise InvalidTrial(
                "EEG分析窗超出实际闪烁区间："
                f"window_end={record.requested_window_end:.6f}, "
                f"stimulus_offset={record.stimulus_offset:.6f}；本试次不分类")
        if record.frame_diagnostics['hard_invalid']:
            self._blank_until(offset_event.timestamp + cfg.blank_min_s)
            observed_rate = record.frame_diagnostics['observed_flip_rate_hz']
            rate_detail = (f"整段翻转均速{observed_rate:.2f}Hz，"
                           f"基准{self.refresh_hz:.2f}Hz，"
                           f"偏差{record.frame_diagnostics['flip_rate_error_percent']:+.2f}%"
                           f"（限值{record.frame_diagnostics['flip_rate_tolerance_percent']:.2f}%）" if
                           observed_rate is not None else "整段翻转速率不可计算")
            raise InvalidTrial(f"刺激时序异常：长间隔{record.frame_diagnostics['long_intervals']}，"
                               f"短间隔{record.frame_diagnostics['short_intervals']}，"
                               f"严重间隔{record.frame_diagnostics['severe_interval_count']}"
                               "（单间隔至少2个刷新周期或时间戳非递增），"
                               f"最大{record.frame_diagnostics['max_interval_ms']:.2f}ms，"
                               f"{rate_detail}", code="DISPLAY")
        if record.frame_diagnostics['warning']:
            print(f"[显示时序警告] trial {record.trial_id} 存在一个异常帧间隔："
                  f"长间隔{record.frame_diagnostics['long_intervals']}，"
                  f"短间隔{record.frame_diagnostics['short_intervals']}；继续分类。", flush=True)
        if self.markers.error is not None:
            raise RuntimeFault("LSL_CONNECTION", f"LSL Marker发送失败：{self.markers.error}")
        self._phase(record, "WAIT_DATA_AND_CLASSIFY")
        result = self._await_trial_result(record, onset_event, offset_event)
        self._apply_trial_result(record, result, target)

    def _draw_mode_menu(self, selected: Optional[int] = None,
                        flicker_rgb: Optional[np.ndarray] = None) -> None:
        """绘制从左到右的三个模式方块，可按帧更新 SSVEP 亮度。"""
        if isinstance(selected, str):
            selected = {"free": 0, "cued": 1, "chatbox": 2}.get(selected)
        self.mode_title.draw()
        if flicker_rgb is None:
            self.mode_squares.colors = self.mode_static_colors
        else:
            self.mode_squares.colors = flicker_rgb
        self.mode_borders.draw()
        self.mode_squares.draw()
        for label in self.mode_labels:
            label.draw()
        if selected is not None:
            self.mode_outline.pos = self.mode_positions[selected]
            self.mode_outline.draw()
        self.mode_status.draw()
        self.mode_hint.draw()
        self.mode_warning.draw()

    def _select_mode(self) -> str:
        """通过三个横向 SSVEP 方块选择自由输入、提示测试或 ChatBox。"""
        self._pause_enabled = False
        self.output_box.autoDraw = False
        self.output_text.autoDraw = False
        self.output_caption.autoDraw = False
        self._apply_screen_layout("menu")
        mode_names = ("free", "cued", "chatbox")
        previous_decoder = getattr(self.collector, "decoder", None)
        from .decoding import FbccaDecoder
        self.collector.decoder = FbccaDecoder(
            self.cfg, calibration=None,
            target_frequencies_hz=MODE_FREQUENCIES_HZ)
        trial_id = 0
        try:
            self.event.getKeys(keyList=["escape"])
            while True:
                trial_id += 1
                self._check_abort()
                self.mode_status.text = "Look at one square. Presenting SSVEP stimulus..."
                onset_event = EventStamp(
                    "mode_stimulus_onset", trial_id, details={
                        "frequencies_hz": list(MODE_FREQUENCIES_HZ),
                        "labels": list(MODE_NAMES), "control": "ssvep"})
                offset_event = EventStamp("mode_stimulus_offset", trial_id)
                frame_flip_times: list[float] = []
                self.win.recordFrameIntervals = True
                self.win.frameIntervals = []
                try:
                    for frame, rgb in enumerate(self.mode_rgb_frames):
                        self._check_abort()
                        self._draw_mode_menu(flicker_rgb=rgb)
                        if frame == 0:
                            self.win.callOnFlip(self.markers.mark, onset_event)
                        flip_time = self.win.flip()
                        frame_flip_times.append(float(flip_time))
                    self.win.callOnFlip(self.markers.mark, offset_event)
                    frame_flip_times.append(float(self.win.flip()))
                finally:
                    self.win.recordFrameIntervals = False
                if onset_event.timestamp is None or offset_event.timestamp is None:
                    raise RuntimeFault("DISPLAY", "模式选择刺激事件标记缺失")
                if self.markers.error is not None:
                    raise RuntimeFault("LSL_CONNECTION", f"LSL Marker发送失败：{self.markers.error}")

                self.mode_status.text = "Analyzing EEG. Keep looking at the square..."
                cancel = threading.Event()
                self._active_decode_cancel = cancel
                self._decode_future = self.executor.submit(
                    decode_trial, trial_id, onset_event.timestamp, self.collector,
                    self.cfg, cancel)
                while (not self._decode_future.done()
                       or self.clock() < offset_event.timestamp + self.cfg.blank_min_s):
                    self._check_abort()
                    self._draw_mode_menu()
                    self.win.flip()
                try:
                    result = self._decode_future.result()
                except InvalidTrial as exc:
                    self._decode_future = None
                    self._active_decode_cancel = None
                    self.mode_status.text = f"Selection failed: {exc}. Retrying..."
                    retry_started = self.clock()
                    while self.clock() - retry_started < self.cfg.feedback_s:
                        self._check_abort()
                        self._draw_mode_menu()
                        self.win.flip()
                    continue
                self._decode_future = None
                self._active_decode_cancel = None
                prediction = int(result["prediction"])
                if not 1 <= prediction <= len(mode_names):
                    raise InvalidTrial(f"模式选择预测类别超出范围：{prediction}")
                selected = prediction - 1
                self.mode_status.text = f"Selected {MODE_NAMES[selected]}. Starting..."
                selected_started = self.clock()
                while self.clock() - selected_started < self.cfg.feedback_s:
                    self._check_abort()
                    self._draw_mode_menu(selected=selected)
                    self.win.flip()
                self.cfg.session_mode = mode_names[selected]
                self._apply_screen_layout(self.cfg.session_mode)
                print(f"\nSelected mode by SSVEP: {MODE_NAMES[selected]} ({MODE_FREQUENCIES_HZ[selected]:.1f} Hz)",
                      flush=True)
                return mode_names[selected]
        finally:
            self._cancel_decode()
            self.collector.decoder = previous_decoder
            self.output_box.autoDraw = True
            self.output_text.autoDraw = True
            self.output_caption.autoDraw = self.cfg.session_mode == "chatbox"

    def run(self) -> str:
        """进入模式选择并运行所选会话，返回结束原因。"""
        if self.static_only:
            raise RuntimeFault("DISPLAY", "静态自检不能启动实验")
        while True:
            mode = self._select_mode()
            if mode == "chatbox":
                reason = self._run_chatbox()
                if reason == "RETURN_TO_MODE_MENU":
                    continue
                return reason
            return self._run_free() if mode == "free" else self._run_cued()

    def _wait_free_pause(self, message: str) -> None:
        """暂停自由输入，等待解码任务结束并由用户恢复。"""
        self._pause_enabled = False
        self._cancel_decode()
        self.event.getKeys(keyList=["space", "return", "num_enter"])
        started = self.clock()
        self.markers.mark(EventStamp("pause", 0, details={"typed_text": self.ledger.typed_text}))
        self._prepare_free_pause_display(message)
        self._sync_output_text()
        self.win.recordFrameIntervals = False
        while True:
            self._check_abort()
            keys = self.event.getKeys(keyList=["space", "return", "num_enter"])
            ready = self._decode_future is None or self._decode_future.done()
            self._draw_free_pause_display()
            self.win.flip()
            if ready and self.clock() - started >= .35 and keys:
                self._decode_future = None
                self._active_decode_cancel = None
                self._pause_enabled = True
                self.markers.mark(EventStamp("resume", 0))
                return

    @staticmethod
    def _release_free_eeg(record: TrialRecord) -> None:
        """在试次已持久化后释放自由输入记录中的大型数组。"""
        if record.result is not None:
            record.result.pop("epoch", None)
            record.result.pop("history_notched_data", None)
        record.frame_flip_times.clear()
        record.frame_lsl_times.clear()
        record.frame_diagnostics.pop("intervals_s", None)
        record.frame_diagnostics.pop("psychopy_frame_intervals_s", None)

    @staticmethod
    def _advance_free_invalid_streak(current: int, record: TrialRecord,
                                     *, pause_requested: bool,
                                     exit_requested: bool) -> int:
        """仅累计真实失败试次；正常拒识不视为故障。"""
        if (pause_requested or exit_requested or record.text_applied
                or record.status == "rejected"):
            return 0
        return current + 1

    def _run_free(self) -> str:
        """循环运行自由输入选择，处理暂停、拒识和连续无效保护。"""
        self._pause_enabled = False
        selection_id = 0
        consecutive_invalid = 0
        try:
            if self.cfg.rejection_enabled and not self.cfg.rejection_thresholds_configured:
                notice = self._rejection_calibration_notice()
                print("[自由输入保护] " + notice.replace("\n", " "), flush=True)
                self.show_message(notice, max(self.cfg.feedback_s, 2.0))
            self._pause_enabled = True
            while True:
                selection_id += 1
                record = TrialRecord(selection_id, 1, None, mode="free", start=self.clock())
                pause_requested = False
                exit_requested = False
                fatal = False
                try:
                    self._run_trial(record)
                except PauseSelection as exc:
                    pause_requested = True
                    record.status = "valid" if record.text_applied else "aborted"
                    record.reason = str(exc)
                except (AbortSession, KeyboardInterrupt) as exc:
                    exit_requested = True
                    record.status = "valid" if record.text_applied else "aborted"
                    record.reason = str(exc) or "KeyboardInterrupt"
                    self.cancel.set()
                except Exception as exc:
                    self._attach_failed_epoch(record, exc)
                    record.status = "valid" if record.text_applied else "invalid"
                    record.reason = f"{type(exc).__name__}: {exc}"
                    fatal = (isinstance(exc, RuntimeFault) or not isinstance(exc, InvalidTrial)
                             or self.collector.error is not None or self.markers.error is not None)
                    self._note_fault(exc, fatal=fatal)
                finally:
                    self._cancel_decode()
                    try:
                        self.win.recordFrameIntervals = False
                        self._draw_status()
                        self.win.flip()
                    except Exception as exc:
                        fatal = True
                        record.reason += f"; display-close error: {exc}"
                        self._note_fault(exc, fatal=True, default="DISPLAY")
                        if not record.text_applied:
                            record.status = "invalid"
                    record.end = self.clock()
                    self._phase(record, "COMPLETE")
                    try:
                        self.markers.mark(EventStamp("selection_end", record.trial_id, None,
                            {"mode": "free", "status": record.status, "reason": record.reason,
                             "prediction": (record.result or {}).get("prediction"),
                             "rejection_diagnostics": (
                                 record.result.get("rejection_diagnostics")
                                 if record.result else None),
                             "text_applied": record.text_applied}))
                    except Exception as exc:
                        record.reason += f"; marker error: {exc}"
                        fatal = True
                        self._note_fault(exc, fatal=True, default="LSL_CONNECTION")
                    self.ledger.commit(record)
                    self._persist_trial(record)
                    self._release_free_eeg(record)
                if record.text_applied:
                    consecutive_invalid = 0
                elif record.status == "rejected":
                    print(f"FREE {selection_id:04d}: rejected | {_brief_reason(record.reason)}", flush=True)
                else:
                    print(f"FREE {selection_id:04d}: no character added | {_brief_reason(record.reason)}", flush=True)
                consecutive_invalid = self._advance_free_invalid_streak(
                    consecutive_invalid, record,
                    pause_requested=pause_requested, exit_requested=exit_requested)
                if exit_requested or fatal:
                    return record.reason or "Free spelling stopped"
                if pause_requested:
                    self._wait_free_pause("Your text is kept. Unfinished selection discarded.")
                    consecutive_invalid = 0
                elif consecutive_invalid >= self.cfg.max_consecutive_invalid:
                    self._wait_free_pause("Several selections could not be decoded.\n"
                                          "No character was inserted; check the EEG stream before resuming.")
                    consecutive_invalid = 0
        finally:
            self._pause_enabled = False
            self._cancel_decode()

    def _run_chatbox(self) -> str:
        """模式三：使用两页 SSVEP ChatBox 键盘，SEND 发送正文。"""
        self._pause_enabled = False
        selection_id = 0
        page = self._chat_page = 1
        self.chat_history_text.text = self._format_chat_history()
        if not self._speaker.available:
            self._speaker = make_speaker(self.cfg.chatbox_voice_enabled, self.cfg.chatbox_voice_rate)
        self._speak("Chat ready. Look at a key to type.")
        previous_decoder = getattr(self.collector, "decoder", None)
        try:
            while True:
                selection_id += 1
                self._chat_page = page
                if self._resting:
                    selection_id = self._run_rest_mode(selection_id)
                    page = self._chat_page = 1
                    continue
                if self._chat_future is not None and not self._typing_during_reply:
                    self._wait_for_chat_reply()
                self._poll_chat_reply()
                if self._unread_reply:
                    self._read_reply_aloud()
                if page == 1:
                    self._refresh_suggestions()
                frequencies = self._chatbox_page_frequencies(page)
                from .decoding import FbccaDecoder
                self.collector.decoder = FbccaDecoder(self.cfg, calibration=None,
                                                      target_frequencies_hz=frequencies)
                try:
                    symbol, result = self._run_chatbox_selection(page, selection_id)
                except InvalidTrial as exc:
                    self.mode_status.text = f"ChatBox selection failed: {exc}. Retrying..."
                    self.header.text = "Not recognised  ·  retrying"
                    if self.cfg.chatbox_speak_selections:
                        self._speak("Not recognised. Try again.")
                    retry_started = self.clock()
                    while self.clock() - retry_started < self.cfg.feedback_s:
                        self._check_abort()
                        self._draw_chatbox_keyboard(page)
                        self.win.flip()
                    continue
                accepted, gate = self._chatbox_gate(result)
                if not accepted:
                    # Probably not looking at the keyboard: type nothing, stay quiet.
                    self._persist_chatbox_selection(selection_id, symbol, result,
                                                    rejected_reason=gate["reason"])
                    self.markers.mark(EventStamp("selection_end", selection_id, None, {
                        "mode": "chatbox", "page": page, "action": symbol, "status": "rejected",
                        "max_score": gate["max_score"], "alpha_ratio": gate["alpha_ratio"]}))
                    self._rejected_in_a_row += 1
                    self.header.text = "No key selected"
                    self.chat_status.text = "Not sure you were looking - nothing typed"
                    auto_rest = self.cfg.chatbox_auto_rest_after
                    if auto_rest and self._rejected_in_a_row >= auto_rest:
                        self._resting = True
                    continue
                self._rejected_in_a_row = 0
                record = self._persist_chatbox_selection(selection_id, symbol, result)
                try:
                    self.markers.mark(EventStamp("selection_end", selection_id, None,
                        {"mode": "chatbox", "page": page, "action": symbol,
                         "suggestion": self._suggestion_word(symbol) or None,
                         "status": record.status, "text_applied": record.text_applied}))
                except Exception as exc:
                    self._note_fault(exc, fatal=True, default="LSL_CONNECTION")
                    return f"ChatBox marker error: {exc}"
                suggestion = self._suggestion_word(symbol)
                if symbol == "BACKSPACE":
                    self.ledger.typed_text = self.ledger.typed_text[:-1]
                    self._sync_output_text(force=True)
                elif suggestion:
                    self.ledger.typed_text = apply_suggestion(self.ledger.typed_text, suggestion)
                    self._sync_output_text(force=True)
                # Ask DeepSeek for the next words now, so it works during the highlight.
                self._request_llm_suggestions()
                self._confirm_chatbox_selection(page, symbol)
                if symbol == "REPLAY":
                    self._read_reply_aloud()
                    continue
                if symbol == "REST":
                    self._resting = True
                    continue
                if symbol == "BACK_MENU":
                    return "RETURN_TO_MODE_MENU"
                if symbol == "123":
                    page = 2
                    continue
                if symbol == "ABC":
                    page = 1
                    continue
                if symbol == "SEND":
                    if self.ledger.typed_text.strip():
                        self._submit_chat_draft()
                    else:
                        self.chat_status.text = "Nothing to send"
                        self.chat_status.color = CHAT_MUTED
                    page = 1
                    continue
        finally:
            self._pause_enabled = False
            self._cancel_decode()
            self.collector.decoder = previous_decoder
            self._speaker.stop()
            if self._chat_future is not None:
                self._chat_future.cancel()

    def _suggestion_word(self, symbol: str) -> str:
        """候选键当前显示的单词；非候选键或空槽返回空字符串。"""
        if symbol not in CHATBOX_SUGGESTION_SYMBOLS:
            return ""
        return self._suggestions[CHATBOX_SUGGESTION_SYMBOLS.index(symbol)]

    def _request_llm_suggestions(self) -> None:
        """为当前草稿后台请求 DeepSeek 预测；同一草稿和上下文只请求一次。"""
        if not (self.cfg.chatbox_predictions_enabled and self.cfg.chatbox_llm_predictions):
            return
        key = (self.ledger.typed_text, len(self._chat_history))
        if key == self._prediction_key:
            return
        if self._prediction_backend is None:
            try:
                self._prediction_backend = DeepSeekChatBackend(
                    model=self.cfg.deepseek_model, endpoint=self.cfg.deepseek_api_url,
                    timeout=max(1.0, self.cfg.chatbox_prediction_wait_s + 3.0),
                    api_key_env=self.cfg.deepseek_api_key_env,
                    api_key_file=self.cfg.deepseek_api_key_file)
            except (ChatBackendError, ValueError) as exc:
                print(f"[预测] DeepSeek 预测不可用，仅使用本地词表：{exc}", file=sys.stderr, flush=True)
                self.cfg.chatbox_llm_predictions = False
                return
        self._prediction_key = key
        self._prediction_started = self.clock()
        self._prediction_future = self.prediction_executor.submit(
            llm_suggestions, self._prediction_backend, self.ledger.typed_text,
            list(self._chat_history), len(CHATBOX_SUGGESTION_SYMBOLS))

    def _refresh_suggestions(self) -> None:
        """在两次刺激之间更新候选词：本地结果立即可用，DeepSeek 结果最多等待
        chatbox_prediction_wait_s（键盘保持静止）。闪烁期间候选词不会改变。"""
        if not self.cfg.chatbox_predictions_enabled:
            return
        draft = self.ledger.typed_text
        self._request_llm_suggestions()
        remote: list[str] = []
        future = self._prediction_future
        if future is not None and self._prediction_key == (draft, len(self._chat_history)):
            deadline = self._prediction_started + self.cfg.chatbox_prediction_wait_s
            while not future.done() and self.clock() < deadline:
                self._check_abort()
                self._draw_chatbox_keyboard(1)
                self.win.flip()
            if future.done():
                try:
                    remote = future.result()
                except Exception as exc:  # noqa: BLE001 - prediction is best effort
                    print(f"[预测] {exc}", file=sys.stderr, flush=True)
        count = len(CHATBOX_SUGGESTION_SYMBOLS)
        words = merge_suggestions(remote, self._predictor.suggest(draft, count * 2), draft, count)
        self._suggestions = words + [""] * (count - len(words))
        for item in self.chat_page_visuals[1]:
            if item["symbol"] in CHATBOX_SUGGESTION_SYMBOLS:
                word = self._suggestion_word(item["symbol"])
                if item["label"].text != (word or " "):
                    item["label"].text = word or " "
                    item["label"].height = self._suggestion_label_height * min(
                        1.0, 7.0 / max(7, len(word)))

    def _confirm_chatbox_selection(self, page: int, symbol: str) -> None:
        """识别后短暂高亮所选键并语音确认，让用户无需移动视线即可知道输入了什么。"""
        word = self._suggestion_word(symbol)
        if symbol in CHATBOX_SUGGESTION_SYMBOLS and not word:
            word = "nothing"  # empty slot: no text was changed
        if self.cfg.chatbox_speak_selections and symbol not in ("SEND", "REPLAY"):
            self._speak(word or CHATBOX_SPOKEN_ACTIONS.get(symbol, symbol))
        index = self._chatbox_page_symbols(page).index(symbol)
        self.header.text = f"Selected  {word.upper() if word else _chatbox_display_label(symbol)}"
        started = self.clock()
        while self.clock() - started < self.cfg.chatbox_feedback_s:
            self._check_abort()
            self._draw_chatbox_keyboard(page, selected_index=index)
            self.win.flip()

    def _run_cued(self) -> str:
        """按随机计划运行提示测试，并处理重试、分段休息和终止。"""
        cfg = self.cfg
        self._pause_enabled = False
        try:
            from psychopy import sound
            self._rest_beep = sound.Sound(880, secs=0.15, volume=0.5, autoLog=False)
        except Exception as exc:
            raise RuntimeFault("AUDIO", f"模式2休息提示音初始化失败：{exc}") from exc
        schedule = deque(make_schedule(cfg.blocks, cfg.random_seed))
        consecutive_invalid = 0
        completed_in_block = 0
        previous_block = 1
        stop_reason = "Completed planned trials"
        trial_id = 0
        while schedule:
            block_id, target_id = schedule.popleft()
            trial_id += 1
            if block_id != previous_block:
                print(f"[模式2轮间休息] 第{previous_block}轮完成，休息{cfg.block_rest_s:g}秒。",
                      flush=True)
                self._wait_cued_rest(previous_block, completed_in_block,
                                      duration_s=cfg.block_rest_s)
                previous_block = block_id
                completed_in_block = 0
            record = TrialRecord(trial_id, block_id, target_id, start=self.clock())
            fatal = False
            try:
                self._run_trial(record)
            except (AbortSession, KeyboardInterrupt) as exc:
                record.status = "valid" if record.text_applied else "aborted"
                record.reason = str(exc) or "KeyboardInterrupt"
                self.cancel.set()
                fatal = True
                stop_reason = record.reason
            except Exception as exc:
                self._attach_failed_epoch(record, exc)
                record.status = "valid" if record.text_applied else "invalid"
                record.reason = f"{type(exc).__name__}: {exc}"
                fatal = (isinstance(exc, RuntimeFault) or not isinstance(exc, InvalidTrial)
                         or self.collector.error is not None or self.markers.error is not None)
                self._note_fault(exc, fatal=fatal)
                stop_reason = record.reason
            finally:
                try:
                    self.win.recordFrameIntervals = False
                    self.win.flip()
                except Exception as clear_error:
                    if not record.text_applied and record.status != "aborted":
                        record.status = "invalid"
                    record.reason = (record.reason + f"; display-close error: {clear_error}").strip("; ")
                    stop_reason, fatal = record.reason, True
                    self._note_fault(clear_error, fatal=True, default="DISPLAY")
                record.end = self.clock()
                self._phase(record, "COMPLETE")
                try:
                    self.markers.mark(EventStamp("trial_end", record.trial_id, record.true_class,
                        {"status": record.status, "reason": record.reason,
                         "prediction": (record.result or {}).get("prediction")}))
                except Exception as marker_error:
                    if not record.text_applied and record.status != "aborted":
                        record.status = "invalid"
                    record.reason = (record.reason + f"; marker error: {marker_error}").strip("; ")
                    stop_reason, fatal = record.reason, True
                    self._note_fault(marker_error, fatal=True, default="LSL_CONNECTION")
                self.ledger.commit(record)
                self._persist_trial(record)
            if record.status == 'valid':
                consecutive_invalid = 0
                completed_in_block += 1
            else:
                consecutive_invalid += 1
                print(f"Trial {trial_id:03d}: {record.status.upper()} - {_brief_reason(record.reason)}", flush=True)
                if not fatal:
                    schedule.appendleft((block_id, target_id))
            if fatal:
                break
            if consecutive_invalid >= cfg.max_consecutive_invalid:
                print(f"[采集暂停] 连续{consecutive_invalid}次无效：{record.reason}", flush=True)
                self.show_message(
                    "Several attempts were invalid. The current target will be retried.\n"
                    "Check the EEG stream, then press SPACE to continue; ESC to stop.",
                    cfg.feedback_s, wait_space=True)
                consecutive_invalid = 0
            if (record.status == 'valid' and completed_in_block % cfg.cued_rest_every == 0
                    and schedule and schedule[0][0] == block_id):
                print(f"[模式2休息] 已完成{completed_in_block}个目标，休息{cfg.cued_rest_s:g}秒。",
                      flush=True)
                self._wait_cued_rest(block_id, completed_in_block)
        else:
            stop_reason = "Completed planned trials"
        return stop_reason

    def _wait_cued_rest(self, block_id: int, completed_in_block: int,
                        *, duration_s: Optional[float] = None) -> None:
        """静态休息期间持续显示剩余时间；最后3、2、1秒各响一次。"""
        duration = self.cfg.cued_rest_s if duration_s is None else duration_s
        if not math.isfinite(duration) or duration < 0:
            raise ValueError("休息时长必须非负且有限")
        self._prepare_cued_rest_display(block_id, completed_in_block)
        self._sync_output_text()
        self.win.recordFrameIntervals = False
        started = self.clock()
        last_beep_second = None
        try:
            while True:
                self._check_abort()
                remaining = duration - (self.clock() - started)
                if remaining <= 0:
                    return
                seconds_left = math.ceil(remaining)
                minutes, seconds = divmod(seconds_left, 60)
                timer_text = f"{minutes:02d}:{seconds:02d}"
                if self.rest_timer.text != timer_text:
                    self.rest_timer.text = timer_text
                if duration >= 3 and seconds_left in (3, 2, 1):
                    if seconds_left != last_beep_second:
                        try:
                            self._rest_beep.play()
                        except Exception as exc:
                            raise RuntimeFault("AUDIO", f"模式2休息提示音播放失败：{exc}") from exc
                        last_beep_second = seconds_left
                self._draw_cued_rest_display()
                self.win.flip()
        finally:
            if self._rest_beep is not None:
                self._rest_beep.stop()

    def _release_visual_stimuli(self) -> None:
        """先释放刺激及其循环引用，再销毁它们所属的 GL 上下文。"""
        from psychopy.visual.basevisual import MinimalStim

        for page_items in getattr(self, "chat_page_visuals", {}).values():
            for item in page_items:
                for key in ("border", "square", "label"):
                    value = item.get(key)
                    if isinstance(value, MinimalStim):
                        value.autoDraw = False
                        item[key] = None

        for name, value in list(vars(self).items()):
            if isinstance(value, MinimalStim):
                value.autoDraw = False
                setattr(self, name, None)
            elif isinstance(value, (list, dict)):
                items = list(value.values()) if isinstance(value, dict) else value
                if items and all(isinstance(item, MinimalStim) for item in items):
                    for item in items:
                        item.autoDraw = False
                    value.clear()

    def close(self) -> None:
        """取消解码、关闭执行器，并在主线程释放视觉资源和窗口。"""
        MainThreadGraphicsGC._check_thread()
        if self._closed:
            return
        self.cancel.set()
        self._cancel_decode()
        try:
            self.executor.shutdown(wait=True, cancel_futures=True)
            self.chat_executor.shutdown(wait=False, cancel_futures=True)
            self.prediction_executor.shutdown(wait=False, cancel_futures=True)
            self._speaker.close()
            if self.win is not None:
                try:
                    self.win.recordFrameIntervals = False
                    self.win._setCurrent()
                    self._release_visual_stimuli()
                    self._graphics_gc.collect(self.win)
                finally:
                    self.win.close()
                    self.win = None
        finally:
            self._closed = True
            if self.wheelchair is not None:
                self.wheelchair.close()
            self._graphics_gc.close()



__all__ = [name for name in globals() if not name.startswith('__')]

