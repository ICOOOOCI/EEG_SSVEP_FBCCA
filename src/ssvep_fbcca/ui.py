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
from .wheelchair import WheelchairPort
from .selection import ChairStateMachine, SelectionVote
from .predict import WordPredictor, apply_suggestion, llm_suggestions, merge_suggestions



LAYOUT_REFERENCE_SIZE = (1470.0, 960.0)
# 窗口模式重新排布所需的最小客户区（更小时保持原布局），以及尺寸停止变化多久后才重新排布。
MIN_WINDOW_SIZE = (800, 520)
RESIZE_SETTLE_S = 0.25

# Each menu tile is decoded against frequencies that are not shown (spread over the
# alpha band): looking elsewhere then tends to win a decoy instead of opening a mode.
MODE_FREQUENCIES_HZ = (8.0, 15.0, 12.0)
MODE_DECOYS_HZ = (9.6, 10.6, 13.6)
MODE_NAMES = ("WHEELCHAIR", "ROBOT ARM", "AI CHAT")
# Mode 2 is a placeholder page: its only key returns to the menu.
ARM_MENU_HZ = 12.0
ARM_DECOYS_HZ = (8.6, 10.2, 13.8, 15.4)
# Mode menu follows the supplied reference: a nearly black blue-gray canvas,
# pale lavender selection tiles, and high-contrast black labels.
MODE_MENU_BG = (-.88, -.84, -.78)
MODE_TILE = (.90, .84, .96)
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
CHAT_MUTED = (.64, .62, .66)
CHAT_WARNING = (1.0, .76, -.30)
CHAT_ACCENT = "#B8A9FF"
CHAT_SELECT = "#FFD84D"               # recognised key; must differ from every key face
CHAT_BUBBLE_USER = "#5A4BB0"
CHAT_BUBBLE_REPLY = "#2A3040"
# Resting key faces tell key types apart. They apply only while the keyboard is
# still: during a flicker round every key uses the same grey luminance frames,
# so these colours cannot change the stimulus contrast of any key.
CHAT_KEY_FUNCTION = "#9DA9CC"
CHAT_KEY_DELETE = "#F2A097"
CHAT_KEY_SEND = "#9E8BFF"
CHAT_KEY_SUGGESTION = "#BDE6DF"

# ChatBox two-page keyboard; TARGETS only maps typed characters to class ids.
# SUGGEST1-3 are the word-prediction keys (candidate bar above Q-P). They are
# appended so the letter keys keep their class order, and are left out entirely
# when chatbox_predictions_enabled is false (the page then has 31 targets).
CHATBOX_SUGGESTION_SYMBOLS = ("SUGGEST1", "SUGGEST2", "SUGGEST3")
CHATBOX_PAGE1_SYMBOLS = tuple("QWERTYUIOPASDFGHJKLZXCVBNM") + (
    "123", "SPACE", "SEND", "BACKSPACE", "BACK_MENU") + CHATBOX_SUGGESTION_SYMBOLS
CHATBOX_PAGE2_SYMBOLS = tuple("1234567890") + (
    ",", ".", "BACK_MENU", "BACKSPACE", "ABC", "SPACE", "SEND")

# IME-style geometry in key-pitch units: (x column, row, width in key widths).
# Row 0 is the top keyboard row; the page switch key sits bottom-left.
CHATBOX_PAGE1_GRID = {
    **{symbol: (-4.5 + index, 0, 1.0) for index, symbol in enumerate("QWERTYUIOP")},
    **{symbol: (-4.0 + index, 1, 1.0) for index, symbol in enumerate("ASDFGHJKL")},
    **{symbol: (-3.0 + index, 2, 1.0) for index, symbol in enumerate("ZXCVBNM")},
    "BACK_MENU": (-4.5, 2, 1.4), "BACKSPACE": (4.5, 2, 1.4),
    "123": (-4.5, 3, 1.4), "SPACE": (0.0, 3, 3.6), "SEND": (4.5, 3, 1.4),
    # Candidate bar: three wide keys, well apart, above the top letter row.
    "SUGGEST1": (-3.3, -1.1, 2.6), "SUGGEST2": (0.0, -1.1, 2.6), "SUGGEST3": (3.3, -1.1, 2.6),
}
# Number page: an even grid that fills the same area as the letter page.
CHATBOX_PAGE2_GRID = {
    **{symbol: ((-4.5, -2.25, 0.0, 2.25, 4.5)[index], -1.1, 1.4) for index, symbol in enumerate("12345")},
    **{symbol: ((-4.5, -2.25, 0.0, 2.25, 4.5)[index], 0.27, 1.4) for index, symbol in enumerate("67890")},
    **{symbol: ((-4.5, -1.5, 1.5, 4.5)[index], 1.63, 1.4)
       for index, symbol in enumerate(("BACK_MENU", ",", ".", "BACKSPACE"))},
    "ABC": (-4.5, 3, 1.4), "SPACE": (0.0, 3, 3.6), "SEND": (4.5, 3, 1.4),
}
# Frequencies are permuted from the 8.0-15.8 Hz / 0.2 Hz benchmark grid by an
# offline search that penalises similar frequencies on nearby keys. On the
# letter page, touching keys differ by >=1.4 Hz and any two keys within 0.4 Hz
# have at least one key between them. The number page has its own grid and assignment below.
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
    # MENU uses the best free grid frequency here: 1.2 Hz from A and Z, slightly under
    # the 1.4 Hz rule above, because moving a letter would change its recorded frequency.
    "BACK_MENU": 13.6,
    # Suggestion keys use unused grid frequencies; every adjacent key differs by >=1.0 Hz.
    "SUGGEST1": 12.2, "SUGGEST2": 15.0, "SUGGEST3": 9.0,
}
# Assigned by an offline search for this grid: neighbouring keys differ by >=1.6 Hz.
CHATBOX_PAGE2_FREQUENCY_BY_SYMBOL = {
    "1": 10.2, "2": 14.6, "3": 10.6, "4": 15.8, "5": 10.4, "6": 8.0, "7": 13.0, "8": 8.6, "9": 13.6, "0": 8.2,
    ",": 11.0, ".": 15.6, "BACK_MENU": 15.2, "BACKSPACE": 11.6, "ABC": 13.2, "SPACE": 9.0, "SEND": 9.4,
}
CHATBOX_FREQUENCY_BY_SYMBOL = {1: CHATBOX_PAGE1_FREQUENCY_BY_SYMBOL,
                               2: CHATBOX_PAGE2_FREQUENCY_BY_SYMBOL}

# Mode 1 wheelchair keys. STOP sits in the alpha band on purpose, so idle alpha activity
# favours stopping; keys that touch differ by >=1.8 Hz.
CHAIR_SYMBOLS = ("UP", "DOWN", "LEFT", "RIGHT", "STOP", "MENU")
CHAIR_FREQUENCY_BY_SYMBOL = {"UP": 12.0, "DOWN": 15.0, "LEFT": 8.4, "RIGHT": 13.8,
                             "STOP": 10.2, "MENU": 15.6}
CHAIR_KEY_STOP = "#F0635C"
CHAIR_STATE_STOPPED = "#8E2F2B"
CHAIR_STATE_MOVING = "#1F7A55"
CHAIR_STATE_CONFIRM = "#8A6A12"
CHAIR_KEY_PAUSED = "#4A5068"
# EEG arrives slightly after the clock reading; windows end this far in the past.
DATA_LAG_S = 0.05

def _chatbox_display_label(symbol: str) -> str:
    """将 ChatBox 页面内部动作名称转换为屏幕标签。"""
    return {"BACK_MENU": "MENU", "BACKSPACE": "DEL", "ABC": "ABC",
            "SPACE": "SPACE", "SEND": "SEND", "123": "123",
            "SUGGEST1": " ", "SUGGEST2": " ",
            "SUGGEST3": " "}.get(symbol, symbol)

# Spoken confirmation after each recognised key, so users who cannot move can
# hear what was selected without looking away from the keyboard.
CHATBOX_SPOKEN_ACTIONS = {"SPACE": "space", "BACKSPACE": "delete", "123": "numbers",
                          "ABC": "letters", "BACK_MENU": "menu", ",": "comma",
                          ".": "period"}

def _chatbox_key_face(symbol: str) -> Any:
    """按键类型返回静止时的键面颜色；字母、数字和标点使用默认键面。"""
    if symbol in CHATBOX_SUGGESTION_SYMBOLS:
        return CHAT_KEY_SUGGESTION
    if symbol == "SEND":
        return CHAT_KEY_SEND
    if symbol == "BACKSPACE":
        return CHAT_KEY_DELETE
    if symbol in ("123", "ABC", "SPACE", "BACK_MENU"):
        return CHAT_KEY_FUNCTION
    return CHAT_SURFACE

def _rounded_rect_vertices(width: float, height: float, radius: float,
                           segments: int = 6) -> list[tuple[float, float]]:
    """返回以原点为中心的圆角矩形顶点。"""
    r = min(radius, width / 2, height / 2)
    x, y = width / 2 - r, height / 2 - r
    points = []
    for center_x, center_y, start in ((x, y, 0), (-x, y, 90), (-x, -y, 180), (x, -y, 270)):
        for step in range(segments + 1):
            angle = math.radians(start + 90 * step / segments)
            points.append((center_x + r * math.cos(angle), center_y + r * math.sin(angle)))
    return points

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
                 *, static_only: bool = False, wheelchair: Optional[WheelchairPort] = None):
        """绑定会话组件并在主线程创建 PsychoPy 窗口和视觉刺激。"""
        MainThreadGraphicsGC._check_thread()
        from psychopy import visual, event, logging
        self.cfg, self.collector, self.markers, self.ledger = cfg, collector, markers, ledger
        self.recorder = recorder
        self.static_only = static_only
        self.wheelchair = wheelchair if wheelchair is not None else WheelchairPort()
        self._chair_state = "STOP"
        self._chair_stop_pressed = False
        self._chair_active: Optional[tuple[str, ...]] = None
        self._live_candidate: Optional[int] = None
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
        self._active_decode_cancel: Optional[threading.Event] = None
        self._decode_future: Any = None
        self._chat_future: Any = None
        self._chat_history: list[ChatMessage] = []
        self._chat_page = 1
        self._unread_reply = False
        self._typing_during_reply = False  # caregiver pressed SPACE while a reply was pending
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
        layout = self._build_visuals()
        self._finish_window_setup(layout)

    def _rounded_rect(self, width: float, height: float, radius: float, pos: Sequence[float],
                      fill: Any, line: Any = None, line_width: float = 1.5) -> Any:
        """创建圆角矩形；键面颜色可与 Rect 一样逐帧设置。"""
        return self.visual.ShapeStim(self.win, units="pix", colorSpace="rgb",
            vertices=_rounded_rect_vertices(width, height, radius), pos=pos,
            fillColor=fill, lineColor=line, lineWidth=line_width, autoLog=False)

    def _build_visuals(self) -> dict[str, Any]:
        """按当前客户区尺寸创建全部视觉元素；窗口尺寸变化后可再次调用以重新排布。"""
        visual, cfg = self.visual, self.cfg
        width, height = _keyboard_coordinate_size(self.win)
        ui_scale = _keyboard_ui_scale(width, height)
        layout = {"coordinate_space": "psychopy_pix_client", "ui_scale_from_reference": ui_scale,
                  "display_size_pix_units": list(self.win.keyboard_display_size)}
        menu_font = "Arial"
        self.mode_title = visual.TextStim(self.win, text="Select a mode",
            pos=(0, height*.30), height=42*ui_scale, font=menu_font,
            color="white", bold=True, autoLog=False)
        mode_square_size = min(300*ui_scale, width*.17, height*.28)
        mode_x = mode_square_size * 1.5
        self.mode_positions = np.asarray(((-mode_x, 0.0), (0.0, 0.0), (mode_x, 0.0)), dtype=float)
        mode_radius = mode_square_size * .09
        self.mode_squares = [self._rounded_rect(mode_square_size, mode_square_size, mode_radius,
                                                position, MODE_TILE)
                             for position in self.mode_positions]
        self.mode_outline = self._rounded_rect(
            mode_square_size + 18*ui_scale, mode_square_size + 18*ui_scale,
            mode_radius + 9*ui_scale, (0, 0), None, CHAT_SELECT, max(1, 3*ui_scale))
        self.mode_labels = [
            visual.TextStim(self.win, text=label_text, pos=position,
                height=mode_square_size*.13, wrapWidth=mode_square_size*.9,
                font=menu_font, color=MODE_LABEL, bold=True, autoLog=False)
            for label_text, position in zip(("MODE 1\nWHEELCHAIR", "MODE 2\nROBOT ARM", "MODE 3\nAI CHAT"),
                                            self.mode_positions)]
        self.mode_status = visual.TextStim(self.win, text="",
            pos=(0, -height/2 + 125*ui_scale), height=20*ui_scale,
            font=menu_font, color="#BDBDC6", autoLog=False)
        # The chat splits the window: chat history and the draft box fill the top
        # half, and the SSVEP keyboard fills the bottom half like a phone IME.
        chat_w = width * .92
        chat_left = -chat_w / 2
        chat_top, chat_bottom = height * .47, height * .257
        chat_pad = 18 * ui_scale
        self.chat_panel = self._rounded_rect(chat_w, chat_top - chat_bottom, 14*ui_scale,
            (0, (chat_top + chat_bottom) / 2), CHAT_SURFACE_DARK)
        chat_font = 18 * ui_scale
        self._chat_font = chat_font
        self._chat_left, self._chat_right = chat_left + chat_pad * .6, -chat_left - chat_pad * .6
        self._chat_bubble_pad = (13 * ui_scale, 6 * ui_scale)
        self._chat_bubble_radius = 11 * ui_scale
        bubble_text_max = chat_w * .80 - 2 * self._chat_bubble_pad[0]
        # Mixed-case Arial averages about half an em per glyph; capitals about 0.7 em.
        self._chat_wrap_reply = max(20, int(bubble_text_max / (chat_font * .47)))
        self._chat_wrap_user = max(12, int(bubble_text_max / (chat_font * .70)))
        probe = visual.TextStim(self.win, text="Ag\nAg\nAg", height=chat_font, font=menu_font,
                                units="pix", autoLog=False)
        one = visual.TextStim(self.win, text="Ag", height=chat_font, font=menu_font,
                              units="pix", autoLog=False)
        self._chat_line_h = max(chat_font, (probe.boundingBox[1] - one.boundingBox[1]) / 2)
        # Blank separator rows give the gap between bubbles, so scrolling stays line-based.
        self._chat_text_top = chat_top - height * .012 - self._chat_bubble_pad[1]
        self._chat_max_lines = max(3, int((self._chat_text_top - chat_bottom - height * .012
                                           - self._chat_bubble_pad[1]) / self._chat_line_h))
        self.chat_bubbles = [
            {"face": self._rounded_rect(10, 10, 1, (0, 0), CHAT_BUBBLE_REPLY),
             "text": visual.TextStim(self.win, text=" ", height=chat_font, font=menu_font,
                 wrapWidth=chat_w, color=CHAT_TEXT, alignText="left", anchorHoriz="left",
                 anchorVert="top", units="pix", autoLog=False),
             "visible": False}
            for _ in range((self._chat_max_lines + 1) // 2 + 1)]
        self.chat_empty_text = visual.TextStim(self.win,
            text="No messages yet  ·  spell a message below, then select SEND",
            pos=(0, (chat_top + chat_bottom) / 2), height=chat_font, font=menu_font,
            wrapWidth=chat_w * .9, color=CHAT_MUTED, autoLog=False)
        self._chat_empty = True
        # Below the keyboard, so it never covers a bubble.
        self.chat_scroll_hint = visual.TextStim(self.win, text="", pos=(0, -height * .475),
            height=14*ui_scale, font=menu_font, color=CHAT_ACCENT, autoLog=False)
        # Only problems are shown here; routine states stay wordless.
        self.chat_status = visual.TextStim(self.win, text="", pos=(0, -height * .475),
            height=18*ui_scale, wrapWidth=width*.9, font=menu_font, color=CHAT_WARNING, autoLog=False)
        draft_h = height * .05
        draft_y = chat_bottom - height * .012 - draft_h / 2
        draft_text_h = min(26 * ui_scale, draft_h * .50)
        self.chat_draft_box = self._rounded_rect(chat_w, draft_h, draft_h * .32, (0, draft_y),
            CHAT_SURFACE_DARK, CHAT_ACCENT, max(1.5, 2 * ui_scale))
        draft_label_w = 92 * ui_scale
        self.output_text = visual.TextStim(self.win, text="",
            pos=(chat_left + chat_pad + draft_label_w, draft_y), height=draft_text_h, font=menu_font,
            wrapWidth=chat_w - 2 * chat_pad - draft_label_w, color=CHAT_TEXT,
            alignText="left", anchorHoriz="left", autoLog=False)
        self.output_caption = visual.TextStim(self.win, text="MESSAGE",
            pos=(chat_left + chat_pad, draft_y), height=16 * ui_scale, font=menu_font,
            color=CHAT_ACCENT, bold=True, alignText="left", anchorHoriz="left", autoLog=False)
        self.output_text.autoDraw = False
        self.output_caption.autoDraw = False
        self._output_max_chars = max(
            12, int((chat_w - 2 * chat_pad - draft_label_w) / (draft_text_h * .72)))
        self._sync_output_text(force=True)
        self._build_chatbox_page_visuals(width, height, ui_scale, menu_font)
        self._build_chair_visuals(width, height, ui_scale, menu_font)
        self.arm_title = visual.TextStim(self.win, text="ROBOT ARM", pos=(0, height * .22),
            height=height * .07, font=menu_font, color=CHAT_TEXT, bold=True, autoLog=False)
        self.arm_menu_key = self._rounded_rect(height * .30, height * .205, height * .03,
                                               (0, -height * .08), CHAT_KEY_FUNCTION)
        self.arm_menu_label = visual.TextStim(self.win, text="MENU", pos=(0, -height * .08),
            height=height * .05, font=menu_font, color=CHAT_KEY_TEXT, bold=True, autoLog=False)
        self.arm_menu_outline = self._rounded_rect(height * .30 + 18 * ui_scale, height * .205 + 18 * ui_scale,
            height * .03 + 9 * ui_scale, (0, -height * .08), None, CHAT_SELECT, max(1, 3 * ui_scale))
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
        self._sync_output_text(force=True)
        self._format_chat_history()
        self.display_info.update(layout)
        self.display_info["client_size_reported"] = tuple(map(int, self.win.clientSize))

    def _finish_window_setup(self, layout: dict[str, Any]) -> None:
        """测量刷新率、生成闪烁帧表并完成静态渲染时序检查。"""
        cfg = self.cfg
        for _ in range(30):
            self._check_abort()
            self._draw_startup_check()
            self.win.flip()
        refresh = self._measure_refresh_rate()
        if refresh is None or not np.isfinite(refresh) or refresh <= 2 * BENCHMARK_FREQUENCIES_HZ.max():
            raise RuntimeError("未测得足够稳定的刷新率；不以猜测的60Hz继续实验")
        self.refresh_hz = float(refresh)
        self.win.refreshThreshold = cfg.frame_long_factor / self.refresh_hz
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
            "target_viewing_distance_cm": cfg.viewing_distance_cm,
            "declared_screen_diagonal_inches": cfg.screen_diagonal_inches,
            "mode_selection": {
                "layout": "left_to_right",
                "labels": list(MODE_NAMES),
                "frequencies_hz": list(MODE_FREQUENCIES_HZ),
                "decoy_frequencies_hz": list(MODE_DECOYS_HZ),
                "control": "ssvep",
            },
            "wheelchair_keyboard": {"symbols": list(CHAIR_SYMBOLS),
                                    "frequencies_hz": [CHAIR_FREQUENCY_BY_SYMBOL[s] for s in CHAIR_SYMBOLS]},
            "chatbox_keyboard": {
                "pages": {
                    str(page): {"layout": "custom", "symbols": list(symbols),
                                "frequencies_hz": list(frequencies)}
                    for page, symbols, frequencies in (
                        (page, self._chatbox_page_symbols(page), self._chatbox_page_frequencies(page))
                        for page in (1, 2))
                },
                "layout": "chat_top_ime_keyboard_bottom",
                "backspace_action": "delete_last_character",
                "page_switch_actions": {"123": "numbers", "ABC": "letters"},
                "menu_action": "return_to_menu",
                "send_action": "submit_chat_draft",
            },
            "selection": {"window_s": cfg.window_s, "chat_window_s": cfg.chat_window_s,
                          "decode_step_s": cfg.decode_step_s, "score_ema": cfg.score_ema,
                          "chat_score_ema": cfg.chat_score_ema, "hold_windows": cfg.selection_hold_windows,
                          "chat_hold_windows": cfg.chat_hold_windows},
            "static_render_check": check, "physical_timing_measured": False, "luminance_gamma_calibrated": False}

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

        self.chat_page_visuals: dict[int, list[dict[str, Any]]] = {}
        for page, layout_items in page_layouts.items():
            visuals = []
            for symbol, position, size in layout_items:
                radius = key_h * .14
                # Same selection ring as the mode menu tiles.
                border = self._rounded_rect(size[0] + 18 * ui_scale, size[1] + 18 * ui_scale,
                                            radius + 9 * ui_scale, position, None, CHAT_SELECT,
                                            max(1, 3 * ui_scale))
                square = self._rounded_rect(size[0], size[1], radius, position,
                                            _chatbox_key_face(symbol))
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
                                "label": label, "position": position, "size": size,
                                "face": _chatbox_key_face(symbol)})
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
        """处理退出按键，并同步检查 EEG 采集健康状态。"""
        keys = self.event.getKeys(keyList=["escape"])
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

    def _sync_output_text(self, force: bool = False) -> None:
        """仅在文本变化或强制刷新时更新顶部输出框。"""
        text = self.ledger.typed_text
        if force or text != self._output_cache:
            self.output_text.text = output_display_text(text, self._output_max_chars)
            self._output_cache = text

    def _draw_chatbox_keyboard(self, page: int, *, flicker_rgb: Optional[np.ndarray] = None,
                               selected_index: Optional[int] = None) -> None:
        """绘制模式三：上半部分聊天记录与草稿框，下半部分 SSVEP 键盘。"""
        self._sync_output_text()
        self._draw_chat_area()
        for index, item in enumerate(self.chat_page_visuals[page]):
            if flicker_rgb is None:
                item["square"].fillColor = item["face"]
            else:
                item["square"].fillColor = tuple(map(float, flicker_rgb[index, :3]))
            item["square"].draw()
            item["label"].draw()
            if index == selected_index:
                item["border"].draw()

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
        """将普通字母/数字/空格映射到字符类别，供账本和记录复用。"""
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

    def _flicker_rgb(self, frequencies: np.ndarray, frame: int) -> np.ndarray:
        """返回本帧各方块的灰度亮度；持续闪烁，不依赖预先生成的帧表。"""
        level = np.sin(2 * np.pi * frequencies * frame / self.refresh_hz)
        return np.repeat(level[:, None], 3, axis=1)

    def _window_candidate(self, vote: SelectionVote, result: dict[str, Any], gate: str,
                          visible: int, profile: Optional[Config] = None) -> Optional[int]:
        """平滑分数并判断本窗口是否有效；返回候选序号，隐藏对照频率胜出时返回 None。

        gate 为 required 时阈值缺失即无效，optional 时阈值缺失不拦截，off 时不做门控。
        """
        profile = self.cfg if profile is None else profile
        smoothed = vote.smooth(result["scores"])
        configured = profile.rejection_thresholds_configured
        if gate == "required" and not configured:
            return None
        if gate != "off" and configured:
            if not self._rejection_diagnostics(dict(result, scores=smoothed), profile)["accepted"]:
                return None
        best = int(np.argmax(smoothed))
        return best if best < visible else None

    def _flicker_until(self, kind: str, trial_id: int, decode_frequencies: Sequence[float],
                       shown_frequencies: Sequence[float], draw: Any, on_window: Any,
                       on_frame: Any = None, window_floor: Any = None,
                       profile: Optional[Config] = None) -> Any:
        """持续闪烁并按滑动窗口识别，直到回调返回非 None 的结果。

        EEG 只取本次闪烁开始之后的数据；解码在工作线程进行，来不及时跳过窗口，
        不会累积延迟。无效窗口以 ``on_window(None)`` 通知。
        """
        from .decoding import FbccaDecoder
        cfg = self.cfg if profile is None else profile
        shown = np.asarray(shown_frequencies, dtype=float)
        previous_decoder = getattr(self.collector, "decoder", None)
        self.collector.decoder = FbccaDecoder(cfg, target_frequencies_hz=tuple(decode_frequencies))
        onset = EventStamp(kind, trial_id, details={"decode_frequencies_hz": list(decode_frequencies),
                                                    "window_s": cfg.window_s, "step_s": cfg.decode_step_s})
        cancel = self._active_decode_cancel = threading.Event()
        future = None
        frame = 0
        window_id = 0
        next_decode = 0.0
        self._live_candidate = None
        self.win.frameIntervals = []
        self.win.recordFrameIntervals = True
        try:
            while True:
                self._check_abort()
                if on_frame is not None:
                    value = on_frame()
                    if value is not None:
                        return value
                draw(self._flicker_rgb(shown, frame))
                if frame == 0:
                    self.win.callOnFlip(self.markers.mark, onset)
                self.win.flip()
                frame += 1
                if onset.timestamp is None:
                    raise RuntimeFault("DISPLAY", "闪烁开始事件标记缺失")
                if self.markers.error is not None:
                    raise RuntimeFault("LSL_CONNECTION", f"LSL Marker发送失败：{self.markers.error}")
                if future is not None and future.done():
                    finished, future = future, None
                    self._decode_future = None
                    try:
                        result = finished.result()
                    except InvalidTrial as exc:
                        print(f"[窗口无效] {exc}", flush=True)
                        result = None
                    value = on_window(result)
                    if value is not None:
                        return value
                now = self.clock()
                if future is None and now >= next_decode:
                    start = now - DATA_LAG_S - cfg.window_s
                    floor = onset.timestamp + cfg.response_delay_s
                    if window_floor is not None:
                        floor = max(floor, window_floor())
                    if start >= floor:
                        window_id += 1
                        next_decode = now + cfg.decode_step_s
                        future = self._decode_future = self.executor.submit(
                            decode_trial, window_id, start - cfg.response_delay_s,
                            self.collector, cfg, cancel)
        finally:
            self.win.recordFrameIntervals = False
            self._live_candidate = None
            cancel.set()
            if future is not None:
                try:
                    future.result(timeout=2.0)
                except Exception:
                    pass
            self._decode_future = None
            self._active_decode_cancel = None
            self.collector.decoder = previous_decoder

    def _select_by_windows(self, kind: str, trial_id: int, decode_frequencies: Sequence[float],
                           shown_frequencies: Sequence[float], draw: Any, gate: str,
                           on_frame: Any = None, profile: Optional[Config] = None) -> Any:
        """闪烁直到连续有效窗口选中同一方块，返回 (序号, 结果)，或 on_frame 的返回值。"""
        cfg = self.cfg if profile is None else profile
        vote = SelectionVote(cfg.selection_hold_windows, cfg.score_ema)
        visible = len(shown_frequencies)

        def on_window(result: Optional[dict[str, Any]]) -> Optional[tuple[int, dict[str, Any]]]:
            candidate = None if result is None else self._window_candidate(vote, result, gate, visible, cfg)
            self._live_candidate = candidate
            stable = vote.record(candidate)
            return None if stable is None else (stable, result)

        return self._flicker_until(kind, trial_id, decode_frequencies, shown_frequencies,
                                   draw, on_window, on_frame, profile=cfg)

    def _run_chatbox_selection(self, page: int, selection_id: int) -> Optional[tuple[str, dict[str, Any]]]:
        """持续闪烁一页键盘直到选中；打字期间回复到达时返回 None，先处理回复。"""
        symbols = self._chatbox_page_symbols(page)
        frequencies = self._chatbox_page_frequencies(page)
        gate = "optional" if self.cfg.chatbox_rejection_enabled else "off"

        def reply_ready() -> Optional[str]:
            return "REPLY" if self._chat_future is not None and self._chat_future.done() else None

        outcome = self._select_by_windows(
            "chatbox_stimulus_onset", selection_id, frequencies, frequencies,
            lambda frame: self._draw_chatbox_keyboard(page, flicker_rgb=frame,
                                                      selected_index=self._live_candidate),
            gate, reply_ready, self.cfg.chat_profile)
        if outcome == "REPLY":
            return None
        return symbols[outcome[0]], outcome[1]

    def _chat_layout(self, messages: Optional[Sequence[ChatMessage]] = None
                     ) -> tuple[list[tuple[str, Optional[str]]], list[tuple[int, int]]]:
        """折行后的 (文本, 角色) 行，以及最后一条消息每行的 (行号, 正文字符下标)。"""
        lines: list[tuple[str, Optional[str]]] = []
        spans: list[tuple[int, int]] = []
        for message in (self._chat_history if messages is None else messages)[-8:]:
            wrap = self._chat_wrap_user if message.role == "user" else self._chat_wrap_reply
            if lines:
                lines.append(("", None))
            spans = []
            for text, start in wrap_with_offsets(plain_chat_text(message.content), wrap):
                spans.append((len(lines), start))
                lines.append((text, message.role))
        return lines, spans

    def _show_chat_lines(self, lines: list[tuple[str, Optional[str]]], offset: int) -> None:
        """把 offset 起能放进面板的行排成气泡，并提示上方/下方是否还有内容。"""
        blocks: list[tuple[int, str, list[str]]] = []
        for row, (text, role) in enumerate(lines[offset:offset + self._chat_max_lines]):
            if role is None:
                continue
            previous = lines[offset + row - 1][1] if row else None
            if blocks and previous is not None:
                blocks[-1][2].append(text)
            else:
                blocks.append((row, role, [text]))
        pad_x, pad_y = self._chat_bubble_pad
        for index, bubble in enumerate(self.chat_bubbles):
            bubble["visible"] = index < len(blocks)
            if not bubble["visible"]:
                continue
            row, role, block_lines = blocks[index]
            stim = bubble["text"]
            stim.text = "\n".join(block_lines)
            bubble_w = stim.boundingBox[0] + 2 * pad_x
            bubble_h = len(block_lines) * self._chat_line_h + 2 * pad_y
            top = self._chat_text_top - row * self._chat_line_h + pad_y
            left = self._chat_right - bubble_w if role == "user" else self._chat_left
            stim.pos = (left + pad_x, top - pad_y)
            bubble["face"].vertices = _rounded_rect_vertices(bubble_w, bubble_h,
                                                             self._chat_bubble_radius)
            bubble["face"].pos = (left + bubble_w / 2, top - bubble_h / 2)
            bubble["face"].fillColor = CHAT_BUBBLE_USER if role == "user" else CHAT_BUBBLE_REPLY
        below = offset + self._chat_max_lines < len(lines)
        self.chat_scroll_hint.text = ("More below  v" if below else
                                      "^  Earlier above" if offset > 0 else "")

    def _format_chat_history(self) -> None:
        """显示对话末尾（最新内容）；只限制显示区域，不截断发送给 API 的完整历史。"""
        self._chat_empty = not self._chat_history
        lines, _ = self._chat_layout()
        self._show_chat_lines(lines, max(0, len(lines) - self._chat_max_lines))

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
            self._format_chat_history()
        # Tell the user the keyboard is about to flicker again.
        self._speak("Your turn.")

    def _wait_for_chat_reply(self) -> None:
        """发送后键盘保持静止（不闪烁），直到回复到达，避免等待期间误选字符。

        回复到达后由 _read_reply_aloud 朗读，读完再恢复闪烁输入。照护者可按空格提前恢复输入。
        """
        self.event.getKeys(keyList=["space"])
        while self._chat_future is not None and not self._chat_future.done():
            self._check_abort()
            if self.event.getKeys(keyList=["space"]):
                self._typing_during_reply = True
                return
            self._draw_chatbox_keyboard(self._chat_page)
            self.win.flip()

    def _draw_chat_area(self) -> None:
        """绘制上半屏的对话面板（不含闪烁）。"""
        self.chat_panel.draw()
        self.chat_status.draw()
        if self._chat_empty:
            self.chat_empty_text.draw()
        for bubble in self.chat_bubbles:
            if bubble["visible"]:
                bubble["face"].draw()
                bubble["text"].draw()
        if self.chat_scroll_hint.text and not self.chat_status.text:
            self.chat_scroll_hint.draw()
        self.chat_draft_box.draw()

    def _start_chat_request(self, text: str) -> bool:
        """在独立线程中把草稿发送给 DeepSeek；SSVEP 输入不等待回复，继续进行。"""
        text = text.strip()
        if not text:
            return False
        if self._chat_future is not None:
            self.chat_status.text = "Still waiting for the previous reply - draft kept"
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
            print(f"[DeepSeek] {exc}", file=sys.stderr, flush=True)
            return False
        request_history = list(self._chat_history)[-20:]
        if self.cfg.deepseek_system_prompt:
            request_history.insert(0, ChatMessage("system", self.cfg.deepseek_system_prompt))
        self._chat_history.append(ChatMessage("user", text))
        self._predictor.learn(text, weight=3)
        self._format_chat_history()
        self.chat_status.text = ""
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
            print(f"[DeepSeek] {exc}", file=sys.stderr, flush=True)
            self._speak("The reply failed. Your message was not answered.")
            return
        self._chat_history.append(ChatMessage("assistant", answer))
        self._predictor.learn(plain_chat_text(answer))
        self._format_chat_history()
        # Read at the next gap between selections, never during a flicker round.
        self._unread_reply = True

    def _submit_chat_draft(self) -> None:
        """后台发送当前草稿；成功提交后清空输入框，键盘立即继续闪烁输入。"""
        if self._start_chat_request(self.ledger.typed_text):
            self.ledger.typed_text = ""
            self._sync_output_text(force=True)

    def _draw_startup_check(self) -> None:
        """启动时显示主菜单；键盘仍在后台绘制后清除，让时序检查覆盖实际绘制负载。"""
        self._draw_chatbox_keyboard(1)
        self.win.clearBuffer()
        self._draw_mode_menu()

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

    def _rejection_diagnostics(self, result: dict[str, Any],
                               profile: Optional[Config] = None) -> dict[str, Any]:
        """评估窗口证据门控，但不把分类分数解释为概率。"""
        cfg = self.cfg if profile is None else profile
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

    def _draw_mode_menu(self, selected: Optional[int] = None,
                        flicker_rgb: Optional[np.ndarray] = None) -> None:
        """绘制菜单方块，可按帧更新 SSVEP 亮度。"""
        self.mode_title.draw()
        for index, square in enumerate(self.mode_squares):
            square.fillColor = (MODE_TILE if flicker_rgb is None
                                else tuple(map(float, flicker_rgb[index, :3])))
            square.draw()
        for label in self.mode_labels:
            label.draw()
        if selected is not None:
            self.mode_outline.pos = self.mode_positions[selected]
            self.mode_outline.draw()
        self.mode_status.draw()

    def _select_mode(self) -> str:
        """注视方块进入对应模式；隐藏的对照频率胜出时继续等待。"""
        self.output_text.autoDraw = False
        self.output_caption.autoDraw = False
        self.win.color = MODE_MENU_BG
        self.mode_status.text = ""
        self.event.getKeys(keyList=["escape"])
        selected, _ = self._select_by_windows(
            "mode_stimulus_onset", 0, (*MODE_FREQUENCIES_HZ, *MODE_DECOYS_HZ), MODE_FREQUENCIES_HZ,
            lambda frame: self._draw_mode_menu(selected=self._live_candidate, flicker_rgb=frame),
            "optional")
        started = self.clock()
        while self.clock() - started < self.cfg.feedback_s:
            self._check_abort()
            self._draw_mode_menu(selected=selected)
            self.win.flip()
        print(f"\nSelected by SSVEP: {MODE_NAMES[selected]} ({MODE_FREQUENCIES_HZ[selected]:.1f} Hz)",
              flush=True)
        return ("wheelchair", "arm", "chatbox")[selected]

    def run(self) -> str:
        """显示菜单并运行所选模式，返回结束原因。"""
        if self.static_only:
            raise RuntimeFault("DISPLAY", "静态自检不能启动实验")
        while True:
            mode = self._select_mode()
            runners = {"wheelchair": self._run_wheelchair, "arm": self._run_arm,
                       "chatbox": self._run_chatbox}
            reason = runners[mode]()
            if reason != "RETURN_TO_MODE_MENU":
                return reason

    @staticmethod
    def _release_free_eeg(record: TrialRecord) -> None:
        """在试次已持久化后释放自由输入记录中的大型数组。"""
        if record.result is not None:
            record.result.pop("epoch", None)
        record.frame_flip_times.clear()
        record.frame_lsl_times.clear()
        record.frame_diagnostics.pop("intervals_s", None)
        record.frame_diagnostics.pop("psychopy_frame_intervals_s", None)

    def _run_chatbox(self) -> str:
        """使用两页 SSVEP ChatBox 键盘，SEND 发送正文。"""
        # The menu can re-enter this mode; ids must stay unique within the session.
        selection_id = max(self.ledger.ids, default=0)
        page = self._chat_page = 1
        self.win.color = CHAT_BG
        self.output_text.autoDraw = True
        self.output_caption.autoDraw = True
        self._format_chat_history()
        if not self._speaker.available:
            self._speaker = make_speaker(self.cfg.chatbox_voice_enabled, self.cfg.chatbox_voice_rate)
        self._speak("Chat ready. Look at a key to type.")
        previous_decoder = getattr(self.collector, "decoder", None)
        try:
            while True:
                selection_id += 1
                self._chat_page = page
                self._graphics_gc.collect(self.win)
                if self._chat_future is not None and not self._typing_during_reply:
                    self._wait_for_chat_reply()
                self._poll_chat_reply()
                if self._unread_reply:
                    self._read_reply_aloud()
                if page == 1:
                    self._refresh_suggestions()
                outcome = self._run_chatbox_selection(page, selection_id)
                if outcome is None:
                    continue
                symbol, result = outcome
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
                    page = 1
                    continue
        finally:
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
        if self.cfg.chatbox_speak_selections and symbol != "SEND":
            self._speak(word or CHATBOX_SPOKEN_ACTIONS.get(symbol, symbol))
        index = self._chatbox_page_symbols(page).index(symbol)
        started = self.clock()
        while self.clock() - started < self.cfg.chatbox_feedback_s:
            self._check_abort()
            self._draw_chatbox_keyboard(page, selected_index=index)
            self.win.flip()

    def _draw_arm(self, flicker_rgb: Optional[np.ndarray] = None) -> None:
        """绘制模式2占位页：标题和返回菜单键。"""
        self.arm_title.draw()
        self.arm_menu_key.fillColor = (CHAT_KEY_FUNCTION if flicker_rgb is None
                                       else tuple(map(float, flicker_rgb[0, :3])))
        self.arm_menu_key.draw()
        self.arm_menu_label.draw()
        if self._live_candidate == 0:
            self.arm_menu_outline.draw()

    def _run_arm(self) -> str:
        """模式2（机械臂）尚未实现：只能注视 MENU 返回菜单。"""
        self.win.color = CHAT_BG
        self._select_by_windows("arm_stimulus_onset", 0, (ARM_MENU_HZ, *ARM_DECOYS_HZ), (ARM_MENU_HZ,),
                                lambda frame: self._draw_arm(frame), "optional")
        return "RETURN_TO_MODE_MENU"

    def _build_chair_visuals(self, width: float, height: float,
                             ui_scale: float, menu_font: str) -> None:
        """Build the mode 1 direction pad: STOP in the centre, arrows around it."""
        key_h = height * .205
        key_w = key_h * 1.45
        pitch_x, pitch_y = key_w + height * .07, height * .27
        center_y = -height * .075
        grid = {"UP": (0, 1), "DOWN": (0, -1), "LEFT": (-1, 0), "RIGHT": (1, 0),
                "STOP": (0, 0), "MENU": (-1, 1)}
        arrow = key_h * .17
        arrows = {"UP": ((-arrow, -arrow * .6), (arrow, -arrow * .6), (0, arrow * .6)),
                  "DOWN": ((-arrow, arrow * .6), (arrow, arrow * .6), (0, -arrow * .6)),
                  "LEFT": ((arrow * .6, -arrow), (arrow * .6, arrow), (-arrow * .6, 0)),
                  "RIGHT": ((-arrow * .6, -arrow), (-arrow * .6, arrow), (arrow * .6, 0))}
        faces = {"STOP": CHAIR_KEY_STOP, "MENU": CHAT_KEY_FUNCTION}
        self.chair_keys = []
        for symbol in CHAIR_SYMBOLS:
            column, row = grid[symbol]
            position = (column * pitch_x, center_y + row * pitch_y)
            directional = symbol in arrows
            label_y = position[1] - (key_h * .22 if directional else 0)
            self.chair_keys.append({
                "symbol": symbol, "face": faces.get(symbol, CHAT_SURFACE),
                "border": self._rounded_rect(key_w + 18 * ui_scale, key_h + 18 * ui_scale,
                                             key_h * .14 + 9 * ui_scale, position, None,
                                             CHAT_SELECT, max(1, 3 * ui_scale)),
                "square": self._rounded_rect(key_w, key_h, key_h * .14, position,
                                             faces.get(symbol, CHAT_SURFACE)),
                "arrow": (self.visual.ShapeStim(self.win, units="pix", vertices=arrows[symbol],
                              pos=(position[0], position[1] + key_h * .14), fillColor=CHAT_KEY_TEXT,
                              lineColor=None, colorSpace="rgb", autoLog=False)
                          if directional else None),
                "label": self.visual.TextStim(self.win, text=symbol, pos=(position[0], label_y),
                              height=key_h * (.17 if directional else .24), font=menu_font,
                              color=CHAT_KEY_TEXT, bold=True, autoLog=False),
            })
        state_y = height * .435
        self.chair_state_box = self._rounded_rect(width * .34, height * .085, height * .03,
                                                  (0, state_y), CHAIR_STATE_STOPPED)
        self.chair_state_text = self.visual.TextStim(self.win, text="STOPPED", pos=(0, state_y),
            height=height * .045, font=menu_font, color=CHAT_TEXT, bold=True, autoLog=False)
        self._show_chair_state()

    def _show_chair_state(self, confirming: Optional[str] = None) -> None:
        """更新顶部状态条；只在状态变化时调用，避免闪烁期间反复生成文字纹理。"""
        moving = self._chair_state != "STOP"
        if confirming is not None:
            self.chair_state_box.fillColor = CHAIR_STATE_CONFIRM
            self.chair_state_text.text = f"CONFIRM {confirming}"
            return
        self.chair_state_box.fillColor = CHAIR_STATE_MOVING if moving else CHAIR_STATE_STOPPED
        self.chair_state_text.text = f"MOVING {self._chair_state}" if moving else "STOPPED"

    def _chair_send(self, command: str, selection_id: int = 0) -> None:
        """发送命令并记录事件；发送失败时立即尝试停止，然后结束本次运行。"""
        try:
            self.wheelchair.send(command)
        except Exception as exc:
            self._chair_state = "STOP"
            self._show_chair_state()
            try:
                self.wheelchair.send("STOP")
            except Exception as stop_exc:
                print(f"[轮椅] STOP发送失败：{stop_exc}", file=sys.stderr, flush=True)
            raise RuntimeFault("INTERNAL", f"轮椅命令发送失败（{command}）：{exc}") from exc
        changed = command != self._chair_state
        self._chair_state = command
        if changed:
            self._show_chair_state()
        # Direction repeats every window as a heartbeat; only log changes and stops.
        if self.markers is not None and (changed or command == "STOP"):
            self.markers.mark(EventStamp("wheelchair_command", selection_id, details={"command": command}))

    def _draw_chair(self, flicker_rgb: Optional[np.ndarray] = None,
                    selected: Optional[str] = None) -> None:
        """绘制方向键；每帧检查键盘空格，按下后不等本轮结束就立即停止。"""
        if self.event.getKeys(keyList=["space"]):
            self._chair_stop_pressed = True
            self._chair_send("STOP")
        self.chair_state_box.draw()
        self.chair_state_text.draw()
        for index, key in enumerate(self.chair_keys):
            if self._chair_active is not None and key["symbol"] not in self._chair_active:
                key["square"].fillColor = CHAIR_KEY_PAUSED
            elif flicker_rgb is None:
                key["square"].fillColor = key["face"]
            else:
                key["square"].fillColor = tuple(map(float, flicker_rgb[index, :3]))
            key["square"].draw()
            if key["arrow"] is not None:
                key["arrow"].draw()
            key["label"].draw()
            if key["symbol"] == selected:
                key["border"].draw()

    def _persist_chair_selection(self, selection_id: int, symbol: str, command: str,
                                 result: dict[str, Any],
                                 gate: Optional[dict[str, Any]]) -> None:
        """保存本轮 EEG 和决定；轮椅命令不写入文字，因此记录状态为 rejected。"""
        # A cycle can log several actions; ids stay unique within the session.
        selection_id = max(selection_id, max(self.ledger.ids, default=0) + 1)
        record = TrialRecord(selection_id, 1, None, mode="free", start=self.clock())
        record.result = dict(result, rejection_diagnostics=gate)
        record.status = "rejected"
        record.reason = f"Wheelchair: target {symbol}, action {command}"
        record.end = self.clock()
        self._phase(record, "COMPLETE")
        self.ledger.commit(record)
        self._persist_trial(record)
        self._release_free_eeg(record)

    def _chair_cycle(self, selection_id: int) -> str:
        """运行一次“选择、确认、行驶”；返回 STOP、STOPPED、CANCELLED 或 MENU。"""
        cfg = self.cfg
        vote = SelectionVote(cfg.selection_hold_windows, cfg.score_ema)
        machine = ChairStateMachine(vote, settle_s=cfg.chair_confirm_settle_s,
                                    confirm_timeout_s=cfg.chair_confirm_timeout_s,
                                    watchdog_s=cfg.chair_watchdog_s)
        frequencies = tuple(CHAIR_FREQUENCY_BY_SYMBOL[s] for s in CHAIR_SYMBOLS)
        self._chair_stop_pressed = False

        def sync_display(before_phase: str) -> None:
            """阶段变化时更新闪烁的键和状态条。"""
            if machine.phase == before_phase:
                return
            self._chair_active = machine.active_keys()
            self._show_chair_state(machine.target if machine.phase == machine.CONFIRM else None)

        def apply(command: Optional[str], outcome: Optional[str], result: Optional[dict[str, Any]],
                  before_phase: str) -> Optional[str]:
            if command is not None:
                self._chair_send(command, selection_id)
            sync_display(before_phase)
            if result is not None and (outcome is not None or machine.phase != before_phase):
                self._persist_chair_selection(selection_id, machine.target or "", command or machine.phase,
                                              result, None)
            return outcome

        def on_window(result: Optional[dict[str, Any]]) -> Optional[str]:
            before = machine.phase
            candidate = None
            if result is not None:
                index = self._window_candidate(vote, result, "required", len(CHAIR_SYMBOLS))
                candidate = None if index is None else CHAIR_SYMBOLS[index]
                self._live_candidate = index
            else:
                self._live_candidate = None
            command, outcome = machine.step(candidate, self.clock())
            return apply(command, outcome, result, before)

        def on_frame() -> Optional[str]:
            before = machine.phase
            if self._chair_stop_pressed:
                self._chair_stop_pressed = False
                command, outcome = machine.stop_pressed()
            else:
                command, outcome = machine.tick(self.clock())
            return apply(command, outcome, None, before)

        try:
            return self._flicker_until(
                "wheelchair_stimulus_onset", selection_id, frequencies, frequencies,
                lambda frame: self._draw_chair(flicker_rgb=frame,
                    selected=None if self._live_candidate is None else CHAIR_SYMBOLS[self._live_candidate]),
                on_window, on_frame, lambda: machine.window_floor)
        finally:
            self._chair_active = None
            self._show_chair_state()

    def _run_wheelchair(self) -> str:
        """模式1：滑动窗口识别，方向键二次确认；不确定、出错或离开时都发送 STOP。"""
        selection_id = max(self.ledger.ids, default=0)
        self.win.color = CHAT_BG
        self.event.getKeys(keyList=["space"])
        try:
            self._chair_send("STOP", selection_id)
            while True:
                selection_id = max(selection_id, max(self.ledger.ids, default=0)) + 1
                self._graphics_gc.collect(self.win)
                outcome = self._chair_cycle(selection_id)
                self._chair_send("STOP", selection_id)
                highlight = {"STOP": "STOP", "STOPPED": "STOP", "MENU": "MENU"}.get(outcome)
                started = self.clock()
                while self.clock() - started < self.cfg.chatbox_feedback_s:
                    self._check_abort()
                    self._draw_chair(selected=highlight)
                    self.win.flip()
                if outcome == "MENU":
                    return "RETURN_TO_MODE_MENU"
        finally:
            self._cancel_decode()
            self._chair_active = None
            try:
                self._chair_send("STOP", selection_id)
            except Exception as exc:
                print(f"[轮椅] 退出时 STOP 发送失败：{exc}", file=sys.stderr, flush=True)

    def _release_visual_stimuli(self) -> None:
        """先释放刺激及其循环引用，再销毁它们所属的 GL 上下文。"""
        from psychopy.visual.basevisual import MinimalStim

        self.chair_keys = []
        self.chat_bubbles = []
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
            self._graphics_gc.close()



__all__ = [name for name in globals() if not name.startswith('__')]

