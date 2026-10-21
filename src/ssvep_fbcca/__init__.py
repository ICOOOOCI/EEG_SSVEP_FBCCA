"""40 目标 SSVEP 键盘应用的稳定公共 API。

实现按配置、信号、采集、记录、分析和 UI 分层；这里仅导出可供外部调用的
稳定符号，避免把第三方依赖和内部辅助函数泄漏成隐式 API。
"""

from .common import (
    AbortSession, DegenerateSignalError, InvalidTrial, PauseSelection,
    RuntimeFault, check_environment, configure_console, fault_record,
    print_fault, probe_save_directory, write_diagnostic,
)
from .config import (
    BENCHMARK_FREQUENCIES_HZ, CONFIG, DEFAULT_WINDOW_S, FREQUENCIES_HZ_BY_ROW,
    KEY_ROWS, LINE_ROBUST_BANDS, M3_5_BANDS, M3_BANDS, N_HARMONICS, TARGETS,
    TARGET_FS, Config, Target, configure_fbcca_profile,
)
from .interfaces import Decoder, EegSource, EventSink, TrialSink
from .signal import (
    NotchContext, apply_history_notch, assess_signal_quality, cca_correlation,
    classify_eeg_window, fbcca_fusion, generate_reference_signals,
    mains_noise_diagnostics, preprocess_lsl_window, sample_count,
)
from .decoding import FbccaDecoder
from .calibration import PersonalCalibration, calibration_settings
from .acquisition import ContinuousLSL, Epoch, MNEStreamWindow, decode_trial
from .events import EventMarkers, EventStamp, TrialLedger, TrialRecord
from .quality import spectral_quality
from .intent import alpha_power_ratio
from .recording import (
    ContinuousRecording, SessionRecorder, export_cued_sidecars,
    package_cued_session, package_recording_files,
)
from .analysis import (
    frame_diagnostics, itr_bits_per_minute, make_luminance_table, make_schedule,
    output_display_text, preflight_transport, print_free_summary, print_summary,
    summarize_free_trials, summarize_trials,
)
from .ui import MainThreadGraphicsGC, PsychoPyKeyboard, calculate_keyboard_layout, keyboard_layout_info
from .llm import ChatBackendError, ChatMessage, DeepSeekChatBackend
from .app import cli, main, observe_lsl_timing, static_ui_self_test

__all__ = [
    "AbortSession", "DegenerateSignalError", "InvalidTrial", "PauseSelection",
    "RuntimeFault", "check_environment", "configure_console", "fault_record",
    "print_fault", "probe_save_directory", "write_diagnostic", "Config", "CONFIG",
    "Target", "TARGETS", "KEY_ROWS", "FREQUENCIES_HZ_BY_ROW", "BENCHMARK_FREQUENCIES_HZ",
    "DEFAULT_WINDOW_S", "TARGET_FS", "N_HARMONICS", "M3_BANDS", "M3_5_BANDS",
    "LINE_ROBUST_BANDS", "configure_fbcca_profile", "Decoder", "EegSource", "EventSink",
    "TrialSink", "NotchContext", "apply_history_notch", "assess_signal_quality",
    "cca_correlation", "classify_eeg_window", "fbcca_fusion", "generate_reference_signals",
    "mains_noise_diagnostics", "preprocess_lsl_window", "sample_count", "FbccaDecoder",
    "PersonalCalibration", "calibration_settings", "ContinuousLSL", "Epoch", "MNEStreamWindow",
    "decode_trial", "EventMarkers", "EventStamp", "TrialLedger", "TrialRecord", "spectral_quality",
    "alpha_power_ratio",
    "ContinuousRecording", "SessionRecorder", "export_cued_sidecars", "package_cued_session",
    "package_recording_files", "frame_diagnostics", "itr_bits_per_minute", "make_luminance_table",
    "make_schedule", "output_display_text", "preflight_transport", "print_free_summary",
    "print_summary", "summarize_free_trials", "summarize_trials", "MainThreadGraphicsGC",
    "PsychoPyKeyboard", "calculate_keyboard_layout", "keyboard_layout_info", "cli", "main",
    "observe_lsl_timing", "static_ui_self_test", "ChatBackendError", "ChatMessage",
    "DeepSeekChatBackend",
    "DEFAULT_DIRECTION_BY_CLASS", "WheelchairBridgeClient", "WheelchairBridgeError",
]

from .wheelchair import DEFAULT_DIRECTION_BY_CLASS, WheelchairBridgeClient, WheelchairBridgeError
