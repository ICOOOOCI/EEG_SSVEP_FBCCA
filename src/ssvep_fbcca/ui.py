from __future__ import annotations

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



KEY_BASE_SIZE = 140.0

KEY_BASE_GAP = 30.0

GRID_WIDTH_FRACTION = .96

GRID_HEIGHT_FRACTION = .58

LAYOUT_REFERENCE_SIZE = (1470.0, 960.0)

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
    """Windows/macOS 使用目标屏幕尺寸全屏启动，布局使用 PsychoPy pix 坐标。"""
    size = cfg.window_size
    import pyglet
    screens = pyglet.canvas.get_display().get_screens()
    if not 0 <= cfg.screen_index < len(screens):
        raise ValueError(f"显示器编号 {cfg.screen_index} 不可用；检测到 {len(screens)} 个显示器")
    screen = screens[cfg.screen_index]
    if cfg.full_screen:
        size = (screen.width, screen.height)
    win = visual.Window(size=size, fullscr=cfg.full_screen, screen=cfg.screen_index,
        winType="pyglet", useRetina=True,
        units="pix", color=(-.78, -.77, -.73), colorSpace="rgb", waitBlanking=True,
        allowGUI=not cfg.full_screen, checkTiming=False, autoLog=False)
    win.keyboard_display_size = np.asarray((screen.width, screen.height), dtype=float)
    return win

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
                 *, static_only: bool = False):
        """绑定会话组件并在主线程创建 PsychoPy 窗口和视觉刺激。"""
        MainThreadGraphicsGC._check_thread()
        from psychopy import visual, event, logging
        self.cfg, self.collector, self.markers, self.ledger = cfg, collector, markers, ledger
        self.recorder = recorder
        self.static_only = static_only
        self.fatal_fault: Optional[dict] = None
        self.trial_faults: list[dict] = []
        self.visual, self.event = visual, event
        self.clock = collector.clock
        self.cancel = threading.Event()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="FBCCA-decoder")
        self.win: Any = None
        self.display_info: dict = {}
        self._pause_enabled = False
        self._rest_beep: Any = None
        self._active_decode_cancel: Optional[threading.Event] = None
        self._decode_future: Any = None
        self._chat_future: Any = None
        self._chat_history: list[ChatMessage] = []
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
        width, height = _keyboard_coordinate_size(self.win)
        layout = keyboard_layout_info(cfg, (width, height), self.win.keyboard_display_size)
        self.positions = np.asarray(layout["key_positions_pix_units"])
        key, gap_y = layout["key_size_pix_units"], layout["row_gap_pix_units"]
        scale = key / KEY_BASE_SIZE
        ui_scale = layout["ui_scale_from_reference"]
        self.key_size = key
        cue_width = min(36.0 * scale, max(0.0, gap_y - 12 * scale) * 9 / 5)
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
        self.output_box.autoDraw = False
        self.output_text.autoDraw = False
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
        self.mode_title = visual.TextStim(self.win, text="Select mode", pos=(0, 165*ui_scale),
            height=34*ui_scale, font=menu_font, color="white", autoLog=False)
        self.mode_positions = {"free": (0, 82*ui_scale), "cued": (0, 0),
                               "chatbox": (0, -82*ui_scale)}
        self.mode_outline = visual.Rect(self.win, width=min(460*ui_scale, width*.8),
            height=76*ui_scale, fillColor=None, lineColor="#FFE45C",
            lineWidth=max(1, 2*ui_scale), units="pix", autoLog=False)
        self.mode_labels = {
            "free": visual.TextStim(self.win, text="1  Free spelling", pos=self.mode_positions["free"],
                height=28*ui_scale, font=menu_font, color="white", autoLog=False),
            "cued": visual.TextStim(self.win, text="2  Cued test", pos=self.mode_positions["cued"],
                height=28*ui_scale, font=menu_font, color="white", autoLog=False),
            "chatbox": visual.TextStim(self.win, text="3  ChatBox  (DeepSeek)", pos=self.mode_positions["chatbox"],
                height=28*ui_scale, font=menu_font, color="white", autoLog=False),
        }
        self.mode_hint = visual.TextStim(self.win, text="1/2/3: select  ·  Enter / Space: start  ·  Esc: exit",
            pos=(0, -height/2 + 82*ui_scale), height=20*ui_scale,
            font=menu_font, color="white", autoLog=False)
        self.mode_warning = visual.TextStim(self.win, text="Warning: starting will show 8–15.8 Hz visual flicker · Esc: stop",
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
        # 模式三的聊天页复用同一个 PsychoPy 窗口，但隐藏40键，只显示消息历史。
        self.chat_title = visual.TextStim(self.win, text="DeepSeek ChatBox",
            pos=(0, height*.41), height=34*ui_scale, font=menu_font,
            color="#FFE45C", bold=True, autoLog=False)
        self.chat_panel = visual.Rect(self.win, width=width*.86, height=height*.68,
            pos=(0, height*.0), fillColor=(-.88, -.88, -.86), lineColor="#64646D",
            lineWidth=max(1, 1.5*ui_scale), units="pix", autoLog=False)
        self.chat_history_text = visual.TextStim(self.win, text="", pos=(-width*.39, height*.29),
            height=18*ui_scale, wrapWidth=width*.76, font=menu_font, color="white",
            alignText="left", anchorHoriz="left", anchorVert="top", autoLog=False)
        self.chat_status = visual.TextStim(self.win, text="", pos=(0, -height*.39),
            height=18*ui_scale, wrapWidth=width*.86, font=menu_font, color="#BDBDC6",
            autoLog=False)
        self.chat_hint = visual.TextStim(self.win, text="Press Enter to return to the 40-key keyboard  ·  ESC: exit",
            pos=(0, -height*.46), height=16*ui_scale, font=menu_font,
            color="white", autoLog=False)
        for _ in range(30):
            self._check_abort()
            self._draw_startup_check()
            self.win.flip()
        refresh = self._measure_refresh_rate()
        if refresh is None or not np.isfinite(refresh) or refresh <= 2 * BENCHMARK_FREQUENCIES_HZ.max():
            raise RuntimeError("未测得足够稳定的刷新率；不以猜测的60Hz继续实验")
        self.refresh_hz = float(refresh)
        self.win.refreshThreshold = cfg.frame_long_factor / self.refresh_hz
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
            "static_render_check": check, "physical_timing_measured": False, "luminance_gamma_calibrated": False}

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
        if self._pause_enabled and "space" in keys:
            self._cancel_decode()
            raise PauseSelection("用户按空格暂停；尚未提交的本轮不会写入字符")

    def _sync_output_text(self, force: bool = False) -> None:
        """仅在文本变化或强制刷新时更新顶部输出框。"""
        text = self.ledger.typed_text
        if force or text != self._output_cache:
            self.output_text.text = output_display_text(text, self._output_max_chars)
            self._output_cache = text

    def _draw_status(self) -> None:
        """绘制当前页面通用的页眉和页脚。"""
        self.header.draw()
        self.footer.draw()

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

    def _format_chat_history(self) -> str:
        """将本地对话历史转换成可读的聊天页文本。"""
        if not self._chat_history:
            return "No messages yet."
        lines = []
        for message in self._chat_history[-6:]:
            label = "YOU" if message.role == "user" else "DEEPSEEK"
            content = " ".join(message.content.split())
            if len(content) > 650:
                content = content[:650].rstrip() + "..."
            lines.append(f"{label}\n{content}")
        rendered = "\n\n".join(lines)
        # 只限制显示区域，不截断发送给 API 的完整历史。
        return rendered[-1800:]

    def _draw_chat_display(self) -> None:
        """绘制模式三聊天窗口，不呈现40键闪烁。"""
        self.chat_panel.draw()
        self.chat_title.draw()
        self.chat_history_text.draw()
        self.chat_status.draw()
        self.chat_hint.draw()

    def _wait_chat_return(self) -> None:
        """在聊天页等待第二次回车，并避免提交键残留导致立即返回。"""
        self.event.getKeys(keyList=["return", "num_enter"])
        armed_at = self.clock()
        while True:
            self._check_abort()
            keys = self.event.getKeys(keyList=["return", "num_enter"])
            self._draw_chat_display()
            self.win.flip()
            if self.clock() - armed_at >= .35 and any(k in keys for k in ("return", "num_enter")):
                return

    def _run_chat_turn(self, text: str) -> None:
        """把一条已由40键输入的文本发送到 DeepSeek 并展示回复。"""
        text = text.strip()
        if not text:
            return
        self.output_box.autoDraw = False
        self.output_text.autoDraw = False
        self._chat_history.append(ChatMessage("user", text))
        self.chat_history_text.text = self._format_chat_history()
        self.chat_status.text = "DeepSeek is replying..."
        try:
            if self._chat_backend is None:
                self._chat_backend = DeepSeekChatBackend(
                    model=self.cfg.deepseek_model,
                    endpoint=self.cfg.deepseek_api_url,
                    timeout=self.cfg.deepseek_timeout_s,
                    api_key_env=self.cfg.deepseek_api_key_env)
            request_history = list(self._chat_history[:-1])[-20:]
            if self.cfg.deepseek_system_prompt:
                request_history.insert(0, ChatMessage("system", self.cfg.deepseek_system_prompt))
            self._chat_future = self.executor.submit(
                self._chat_backend.send, text, history=request_history)
            while not self._chat_future.done():
                self._check_abort()
                self._draw_chat_display()
                self.win.flip()
            answer = self._chat_future.result()
            self._chat_history.append(ChatMessage("assistant", answer))
            self.chat_history_text.text = self._format_chat_history()
            self.chat_status.text = "Reply complete"
        except (ChatBackendError, ValueError) as exc:
            self.chat_status.text = f"DeepSeek request failed: {exc}"
            print(f"[DeepSeek] {exc}", file=sys.stderr, flush=True)
        finally:
            self._chat_future = None
        try:
            self._wait_chat_return()
        finally:
            self.output_box.autoDraw = True
            self.output_text.autoDraw = True

    def _chat_submit_requested(self) -> bool:
        """消费一次物理回车请求；没有草稿时回车不会打断连续输入。"""
        keys = self.event.getKeys(keyList=["return", "num_enter"])
        return bool(self.ledger.typed_text.strip()) and any(
            key in keys for key in ("return", "num_enter"))

    def _submit_chat_draft(self) -> None:
        """提交当前完整草稿，并在聊天页返回后继续下一条脑机输入。"""
        prompt = self.ledger.typed_text
        self.ledger.typed_text = ""
        self._sync_output_text(force=True)
        self.message.text = ""
        self._run_chat_turn(prompt)

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
        self.header.text = ((f"CHATBOX | Selection {record.trial_id}" if self.cfg.session_mode == "chatbox"
                             else f"Selection {record.trial_id}") if free else
                            f"Trial {record.trial_id} of {self.cfg.blocks*len(TARGETS)}")
        self.footer.text = (("Waiting for EEG result · ESC: stop" if self.cfg.session_mode == "chatbox"
                             else "SPACE: pause  ·  ESC: stop") if free else "ESC: stop")
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
            if self.cfg.session_mode == "chatbox":
                self.header.text = f"CHATBOX | Selection {record.trial_id} | Recognized: {pred.symbol}"
                feedback = (f"Recognized: {pred.symbol} ({pred.frequency_hz:.1f} Hz)\n\n"
                            "Continue with the 40-key keyboard; press Enter when finished.\n"
                            "ESC: exit")
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
            self.header.text = (f"CHATBOX | Selection {record.trial_id} | Choose your next character"
                                if cfg.session_mode == "chatbox" else
                                f"FREE | Selection {record.trial_id} | Choose your next character")
            self.footer.text = ("Enter: send when finished  ·  ESC: exit" if cfg.session_mode == "chatbox"
                                else "Keyboard SPACE: pause   ·   ESC: exit   ·   <-: delete")
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

    def _draw_mode_menu(self, selected: str) -> None:
        """绘制模式选择菜单并标出当前选项。"""
        self.mode_title.draw()
        self.mode_outline.pos = self.mode_positions[selected]
        self.mode_outline.draw()
        for label in self.mode_labels.values():
            label.draw()
        self.mode_hint.draw()
        self.mode_warning.draw()

    def _select_mode(self) -> str:
        """等待用户选择自由输入、提示测试或 DeepSeek ChatBox。"""
        self._pause_enabled = False
        selected = self.cfg.session_mode
        mode_keys = ["1", "2", "3", "num_1", "num_2", "num_3",
                     "space", "return", "num_enter"]
        self.event.getKeys(keyList=mode_keys)
        self.output_box.autoDraw = False
        self.output_text.autoDraw = False
        try:
            while True:
                self._check_abort()
                keys = self.event.getKeys(keyList=mode_keys)
                if "1" in keys or "num_1" in keys:
                    selected = "free"
                if "2" in keys or "num_2" in keys:
                    selected = "cued"
                if "3" in keys or "num_3" in keys:
                    selected = "chatbox"
                self._draw_mode_menu(selected)
                self.win.flip()
                if any(k in keys for k in ("space", "return", "num_enter")):
                    self.cfg.session_mode = selected
                    labels = {"free": "Free spelling", "cued": "Cued test",
                              "chatbox": "DeepSeek ChatBox"}
                    print("\nSelected mode: " + labels[selected], flush=True)
                    return selected
        finally:
            self.output_box.autoDraw = True
            self.output_text.autoDraw = True

    def run(self) -> str:
        """进入模式选择并运行所选会话，返回结束原因。"""
        if self.static_only:
            raise RuntimeFault("DISPLAY", "静态自检不能启动实验")
        mode = self._select_mode()
        if mode == "chatbox":
            return self._run_chatbox()
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
        """模式三：持续使用40键输入，回车发送后进入聊天页，再回车返回。"""
        self._pause_enabled = False
        selection_id = 0
        consecutive_invalid = 0
        try:
            while True:
                # Enter 是提交键，不是“每字符一次”的确认键；没有 Enter 时马上
                # 开始下一轮 EEG 选择，因此可以连续输入整句话。
                if self._chat_submit_requested():
                    self._submit_chat_draft()
                    continue
                selection_id += 1
                record = TrialRecord(selection_id, 1, None, mode="free", start=self.clock())
                exit_requested = False
                fatal = False
                try:
                    self._run_trial(record)
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
                            {"mode": "chatbox", "status": record.status,
                             "prediction": (record.result or {}).get("prediction"),
                             "text_applied": record.text_applied}))
                    except Exception as exc:
                        record.reason += f"; marker error: {exc}"
                        fatal = True
                        self._note_fault(exc, fatal=True, default="LSL_CONNECTION")
                    self.ledger.commit(record)
                    self._persist_trial(record)
                    self._release_free_eeg(record)
                if exit_requested or fatal:
                    return record.reason or "ChatBox stopped"
                if record.text_applied:
                    consecutive_invalid = 0
                elif record.status == "rejected":
                    consecutive_invalid = 0
                else:
                    consecutive_invalid += 1
                if consecutive_invalid >= self.cfg.max_consecutive_invalid:
                    self.show_message("Several selections were not recognized. Check the EEG stream, then press ESC to exit.",
                                      self.cfg.feedback_s)
                    consecutive_invalid = 0
                if self._chat_submit_requested():
                    self._submit_chat_draft()
        finally:
            self._pause_enabled = False
            self._cancel_decode()
            if self._chat_future is not None:
                self._chat_future.cancel()

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

