"""40 目标 SSVEP 键盘应用的稳定公共 API。

实现按配置、信号、采集、记录、分析和 UI 分层；这里仅导出可供外部调用的
稳定符号，避免把第三方依赖和内部辅助函数泄漏成隐式 API。
"""

import os as _os

# liblsl 在首次加载时读取 LSLAPICFG；必须早于任何 mne_lsl 导入。默认只显示警告/错误，
# 用户已自行设置 LSLAPICFG 时不覆盖。
_os.environ.setdefault("LSLAPICFG", _os.path.join(_os.path.dirname(__file__), "data", "lsl_api.cfg"))

from .common import (
    AbortSession, DegenerateSignalError, InvalidTrial,
    RuntimeFault, check_environment, configure_console, fault_record,
    print_fault, probe_save_directory, write_diagnostic,
)
from .config import (
    BENCHMARK_FREQUENCIES_HZ, CONFIG, DEFAULT_WINDOW_S, FREQUENCIES_HZ_BY_ROW,
    KEY_ROWS, M3_5_BANDS, N_HARMONICS, TARGETS,
    TARGET_FS, Config, Target,
)
from .interfaces import Decoder, EegSource, EventSink, TrialSink
from .signal import (
    assess_signal_quality, cca_correlation,
    classify_eeg_window, fbcca_fusion, generate_reference_signals,
    mains_noise_diagnostics, preprocess_lsl_window, sample_count,
)
from .decoding import FbccaDecoder
from .acquisition import ContinuousLSL, Epoch, MNEStreamWindow, decode_trial
from .events import EventMarkers, EventStamp, TrialLedger, TrialRecord
from .quality import spectral_quality
from .intent import alpha_power_ratio
from .recording import (
    ContinuousRecording, SessionRecorder, package_recording_files, tidy_session_archive, write_tidy_archive,
)
from .analysis import (
    frame_diagnostics, output_display_text, preflight_transport, print_free_summary,
    summarize_free_trials,
)
from .ui import MainThreadGraphicsGC, PsychoPyKeyboard
from .wheelchair import WHEELCHAIR_COMMANDS, WheelchairPort
from .llm import ChatBackendError, ChatMessage, DeepSeekChatBackend
from .voice import Speaker, make_speaker
from .app import cli, main, observe_lsl_timing, static_ui_self_test

__all__ = [
    "AbortSession", "DegenerateSignalError", "InvalidTrial",
    "RuntimeFault", "check_environment", "configure_console", "fault_record",
    "print_fault", "probe_save_directory", "write_diagnostic", "Config", "CONFIG",
    "Target", "TARGETS", "KEY_ROWS", "FREQUENCIES_HZ_BY_ROW", "BENCHMARK_FREQUENCIES_HZ",
    "DEFAULT_WINDOW_S", "TARGET_FS", "N_HARMONICS", "M3_5_BANDS",
    "Decoder", "EegSource", "EventSink",
    "TrialSink", "assess_signal_quality",
    "cca_correlation", "classify_eeg_window", "fbcca_fusion", "generate_reference_signals",
    "mains_noise_diagnostics", "preprocess_lsl_window", "sample_count", "FbccaDecoder",
    "ContinuousLSL", "Epoch", "MNEStreamWindow",
    "decode_trial", "EventMarkers", "EventStamp", "TrialLedger", "TrialRecord", "spectral_quality",
    "alpha_power_ratio",
    "ContinuousRecording", "SessionRecorder",
    "package_recording_files", "tidy_session_archive", "write_tidy_archive", "frame_diagnostics",
    "output_display_text", "preflight_transport", "print_free_summary",
    "summarize_free_trials", "MainThreadGraphicsGC",
    "PsychoPyKeyboard", "cli", "main",
    "observe_lsl_timing", "static_ui_self_test", "ChatBackendError", "ChatMessage",
    "DeepSeekChatBackend", "Speaker", "make_speaker", "WHEELCHAIR_COMMANDS", "WheelchairPort",
]
