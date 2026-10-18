from __future__ import annotations

from .common import *
from .config import *
from .signal import *
from .calibration import *
from .acquisition import *
from .events import *
from .quality import *
from .recording import *
from .analysis import *
from .ui import *
from .decoding import FbccaDecoder



LAST_SESSION: Optional[dict[str, Any]] = None

def static_ui_self_test(cfg: Config, *, seconds: float = 5.0) -> dict:
    """渲染正式键盘、固定提示和输出框，但不启动 LSL 或解码。"""
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("自检时长必须为有限正数")
    app = None
    result = {"status": "cancelled", "exit_code": 130,
              "eeg_connected": False, "flicker_presented": False,
              "scope": "静态布局、字体、窗口与软件刷新时序；不验证EEG、识别准确率或物理显示延迟"}
    try:
        app = PsychoPyKeyboard(replace(cfg), ContinuousLSL(cfg), None, TrialLedger(), static_only=True)
        app.output_box.autoDraw = True
        app.output_text.autoDraw = True
        app.ledger.typed_text = "STATIC UI CHECK  ABC 123"
        app._sync_output_text(force=True)
        app.cue_title.text = "STATIC UI CHECK - no EEG / no flicker"
        app.footer.text = "40 keys + output + fixed cue | ESC: cancel | Closes automatically"
        started = time.monotonic()
        while time.monotonic() - started < seconds:
            app._check_abort()
            app._draw_keys(1, show_outline=True)
            app.win.flip()
        result.update(status="passed", exit_code=0, display_info=app.display_info)
    except AbortSession:
        print("静态界面自检已取消，未记为通过。", flush=True)
    except Exception as exc:
        raise RuntimeFault("DISPLAY", str(exc)) from exc
    finally:
        if app is not None:
            try:
                app.close()
            except Exception as exc:
                fault = fault_record(exc, "DISPLAY")
                result.setdefault("errors", []).append(fault)
                result.update(status="failed", exit_code=40)
                print_fault(fault)
    if result["status"] == "passed":
        print("静态界面自检通过：40键、文字、输出框、固定提示与静态绘制时序已检查。", flush=True)
    return result

def main(*, cfg: Optional[Config] = None) -> dict:
    """唯一运行入口；不要求CLI参数；返回全部内存记录。"""
    global LAST_SESSION
    cfg = replace(cfg if cfg is not None else CONFIG)
    cfg.validate()
    print(f"[算法] {cfg.filter_bank_profile}；{len(cfg.filter_bands)}个子带，"
          f"上限{max(high for _, high in cfg.filter_bands):g}Hz，{cfg.n_harmonics}次谐波；"
          f"权重 a={cfg.weight_a:g}, b={cfg.weight_b:g}；分析窗{cfg.window_s:g}秒。\n"
          f"当前{len(cfg.channel_indices)}路设备/键盘时序适配不等同论文完整实验复现；"
          "LSL声明采样率不能独立证明硬件有效带宽。", flush=True)
    if cfg.notch_mode == "history":
        print(f"[预处理] history：{cfg.notch_history_s:g}秒历史辅助{cfg.notch_hz:g}Hz陷波，"
              f"分类仍使用{cfg.window_s:g}秒；上下文不足即拒绝，不能加载旧个人模板。"
              "此模式需新的在线数据验证准确率。", flush=True)
    else:
        print("[预处理] epoch：沿用当前分析窗独立陷波。", flush=True)
    calibration = None
    if cfg.calibration_file:
        model_path = Path(cfg.calibration_file)
        if not model_path.is_absolute():
            model_path = ROOT / model_path
        calibration = PersonalCalibration.load(model_path, cfg)
        print(f"个人模型已加载：受试者 {cfg.participant_id}，"
              f"每类 {calibration.counts.min()}–{calibration.counts.max()} 次校准，"
              f"模型 {calibration.model_id[:12]}", flush=True)
    print("[1/3] 运行环境由启动入口核验；准备保存目录检查…", flush=True)
    print(f"安全提示：此程序呈现{BENCHMARK_FREQUENCIES_HZ.min():g}–{BENCHMARK_FREQUENCIES_HZ.max():g}Hz视觉闪烁。"
          "对闪光敏感、有光敏性癫痫史者不要自行测试；"
          "出现眼部不适、头痛、眩晕等应立即按ESC停止。先在OpenBCI GUI检查电极与接线，程序不会直接开始闪烁。")
    collector = ContinuousLSL(cfg)
    collector.calibration = calibration
    collector.decoder = FbccaDecoder(cfg, calibration=calibration)
    ledger = TrialLedger()
    markers: Optional[EventMarkers] = None
    app: Optional[PsychoPyKeyboard] = None
    recorder: Optional[SessionRecorder] = None
    review: dict = {}
    reason = "Not started"
    errors: list[dict] = []
    stage = "SAVE"

    def add_error(exc: BaseException, default: str) -> None:
        """将异常登记为标准故障并立即输出。"""
        fault = fault_record(exc, default)
        errors.append(fault)
        print_fault(fault)

    try:
        probe_save_directory(cfg.record_root)
        stage = "SAVE"
        recorder = SessionRecorder(cfg, context={"mode_at_start": cfg.session_mode,
                                                  "trained_model": ({
                                                      "model_id": calibration.model_id,
                                                      **calibration.metadata,
                                                  } if calibration is not None else None)})
        collector.recorder = recorder
        print("\n[2/3] 连接EEG LSL并检查采样时间轴（请保持采集软件的LSL输出开启）…", flush=True)
        stage = "LSL_CONNECTION"
        collector.start()
        stage = "ACQUISITION"
        review = preflight_transport(cfg, collector)
        if calibration is not None:
            calibration.validate_config(cfg)
        stage = "LSL_CONNECTION"
        markers = EventMarkers(cfg, collector.clock)
        markers.mark(EventStamp("session_start", 0, details={"mode_at_start": cfg.session_mode}))
        stage = "INTERNAL"
        t = np.arange(sample_count(cfg.window_s, cfg.target_fs)) / cfg.target_fs
        warm = np.tile(np.sin(2*np.pi*10*t), (len(cfg.channel_indices), 1))
        warm_context = None
        if cfg.notch_mode == "history":
            count = round(cfg.notch_history_s * cfg.target_fs)
            history_t = np.arange(count + len(t)) / cfg.target_fs
            history_x = np.tile(np.sin(2*np.pi*10*history_t), (len(cfg.channel_indices), 1))
            warm = history_x[:, count:].copy()
            warm_context = NotchContext(
                data=history_x, local_timestamps=history_t,
                uniform_timestamps=history_t[count:].copy(), source_offset=count,
                epoch_end=count/cfg.target_fs + cfg.window_s,
                requested_history_samples=count)
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):
            # 启动预热也经过 Decoder 端口，替换解码器时不会遗漏启动路径。
            collector.decoder.decode(warm, cfg.target_fs, notch_context=warm_context)
        print("[3/3] 创建键盘窗口；在菜单按1自由输入、2提示测试，再按SPACE开始…", flush=True)
        stage = "DISPLAY"
        app = PsychoPyKeyboard(cfg, collector, markers, ledger, recorder)
        reason = app.run()
        if app.fatal_fault is not None:
            errors.append(app.fatal_fault)
    except (AbortSession, KeyboardInterrupt) as exc:
        reason = str(exc) or "用户中止"
    except Exception as exc:
        add_error(exc, stage)
        reason = f"{type(exc).__name__}: {exc}"
    finally:
        for resource in (app, collector):
            if resource is not None:
                try:
                    resource.close()
                except Exception as exc:
                    add_error(exc, "DISPLAY" if resource is app else "ACQUISITION")
                    reason += f"; {type(resource).__name__}.close: {type(exc).__name__}: {exc}"
        if markers is not None:
            try:
                markers.mark(EventStamp("session_end", 0, details={"stop_reason": reason}))
            except Exception as exc:
                add_error(exc, "LSL_CONNECTION")
                reason += f"; session_end marker: {type(exc).__name__}: {exc}"
            try:
                markers.close()
            except Exception as exc:
                add_error(exc, "LSL_CONNECTION")
                reason += f"; markers.close: {type(exc).__name__}: {exc}"
        report = (summarize_free_trials(ledger.records, ledger) if cfg.session_mode == "free"
                  else summarize_trials(ledger.records, cfg.blocks * len(TARGETS)))
        session_artifact = None
        if recorder is not None:
            try:
                session_artifact = recorder.finalize(
                    summary=report, events=markers.events if markers else [],
                    acquisition_metadata=collector.metadata, acquisition_review=review,
                    clock_updates=collector.clock_updates,
                    display_info=app.display_info if app else {},
                    typed_text=ledger.typed_text, stop_reason=reason)
            except Exception as exc:
                add_error(exc, "SAVE")
                reason += f"; 保存失败：{exc}"
                if recorder.archive_path.exists():
                    session_artifact = str(recorder.archive_path)
                    print(f"会话ZIP已保存，但临时文件清理未完成：{type(exc).__name__}: {exc}", flush=True)
                elif recorder.session_path.exists():
                    session_artifact = str(recorder.session_path)
                    print(f"会话NPZ已保存，但配套文件封装未完成：{type(exc).__name__}: {exc}", flush=True)
                else:
                    print(f"实验记录导出失败，已有逐试次文件保留：{type(exc).__name__}: {exc}", flush=True)
        if collector.error is not None and not any(f["code"] == "ACQUISITION" for f in errors):
            add_error(collector.error, "ACQUISITION")
        if (recorder is not None and recorder.continuous is not None
                and recorder.continuous.error is not None
                and not any(f["code"] == "SAVE" for f in errors)):
            add_error(recorder.continuous.error, "SAVE")
        if markers is not None and markers.error is not None and not any(f["code"] == "LSL_CONNECTION" for f in errors):
            add_error(markers.error, "LSL_CONNECTION")
        LAST_SESSION = {"mode": cfg.session_mode, "config": cfg, "targets": TARGETS, "records": ledger.records,
            "events": markers.events if markers else [], "acquisition_metadata": collector.metadata,
            "acquisition_review": review, "clock_updates": collector.clock_updates,
            "display_info": app.display_info if app else {}, "summary": report,
            "typed_text": ledger.typed_text, "stop_reason": reason,
            "exit_code": (50 if any(f["code"] == "SAVE" for f in errors)
                          else errors[0]["exit_code"] if errors else 0),
            "errors": errors, "trial_faults": app.trial_faults if app else [],
            "session_artifact": session_artifact,
            "export_artifacts": dict(recorder.export_artifacts) if recorder else {},
            "marker_error": str(markers.error) if markers and markers.error else None,
            "validation_scope": "software events only; physical screen/device delays unmeasured"}
        if cfg.session_mode == "free":
            print_free_summary(report)
        else:
            print_summary(report, ledger)
        saved_path = session_artifact
        if saved_path is None and recorder is not None and recorder.session_dir.exists():
            saved_path = str(recorder.session_dir)
        print(f"结束原因：{reason}")
        print(f"保存路径：{saved_path or '未生成'}")
        if recorder is not None:
            for path in recorder.export_artifacts.values():
                if path != session_artifact:
                    print(f"导出文件：{path}")
        if markers is not None and markers.error is not None:
            print(f"外部Marker发送存在错误：{markers.error}；请勿把本次外部Marker流视为完整。")
    return LAST_SESSION

def observe_lsl_timing(cfg: Config, collector: ContinuousLSL, seconds: float,
                       *, report: Optional[dict] = None) -> dict:
    """预热后每秒检查最近2秒；保留中途异常，不能只用末窗宣称全程通过。

    只读接收缓冲，不开启闪烁，不产生准确率。相邻窗口重叠；此诊断
    不能代替设备计数器或物理延迟测量。工频占比仅作描述，不判失败。
    """
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("观察时长必须为有限正数")
    if collector.buffer is None:
        raise RuntimeFault("ACQUISITION", "LSL缓冲区尚未建立")
    result = report if report is not None else {}
    result.update(requested_observation_s=float(seconds), window_s=2.0,
                  check_interval_s=1.0, completed=False, passed=False,
                  observations=[], invalid_window_count=0,
                  scope="每秒检查最近2秒重叠窗；首窗包含预热末段；不测物理延迟或识别准确率")
    started = time.monotonic()
    deadline = started + seconds
    next_check = started
    last_end = -math.inf
    while True:
        collector.check_health()
        now = time.monotonic()
        if now >= next_check or now >= deadline:
            end = collector.buffer.latest
            if end > last_end:
                item = {"elapsed_s": float(now - started),
                        "requested_window_start": float(end - 2.0),
                        "requested_window_end": float(end)}
                try:
                    epoch = collector.buffer.epoch(
                        end - 2.0, 2.0, collector.fs, cfg.max_gap_factor,
                        cfg.rate_tolerance, cfg.hard_gap_factor,
                        cfg.hard_rate_tolerance, cfg.max_warning_gaps,
                        **({"notch_history_s": cfg.notch_history_s} if cfg.notch_mode == "history" else {}))
                    item.update(valid=True, diagnostics=epoch.diagnostics,
                                mains_noise=mains_noise_diagnostics(
                                    epoch.data, epoch.fs, cfg.notch_hz))
                except InvalidTrial as exc:
                    item.update(valid=False, reason=str(exc),
                                diagnostics=exc.epoch.diagnostics if exc.epoch is not None else {})
                    result["invalid_window_count"] += 1
                    print(f"[LSL持续检查 {now - started:.1f}s] 窗口异常：{exc}", flush=True)
                result["observations"].append(item)
                last_end = end
            next_check = now + 1.0
        if now >= deadline:
            break
        collector.stop_event.wait(max(0.0, min(.05, deadline - now, next_check - now)))
    result.update(completed=True, elapsed_s=float(time.monotonic() - started),
                  checked_window_count=len(result["observations"]))
    result["passed"] = bool(result["observations"] and result["invalid_window_count"] == 0)
    fractions = [item["mains_noise"]["median_line_fraction"]
                 for item in result["observations"]
                 if item["valid"] and item["mains_noise"]["available"]]
    result["median_line_fraction_across_windows"] = float(np.median(fractions)) if fractions else None
    print(f"[LSL持续检查] {result['checked_window_count']}个重叠窗，"
          f"{result['invalid_window_count']}个时间轴异常窗；"
          "工频逐通道占比和每窗诊断写入检查报告。", flush=True)
    return result

def positive_seconds(value: str) -> float:
    """将命令行文本解析为有限正秒数。"""
    seconds = float(value)
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError("seconds 必须是有限正数")
    return seconds

def nonnegative_seconds(value: str) -> float:
    """将命令行文本解析为允许零的有限非负秒数。"""
    seconds = float(value)
    if not math.isfinite(seconds) or seconds < 0:
        raise argparse.ArgumentTypeError("seconds 必须是有限非负数")
    return seconds

def cli(argv=None) -> int:
    """Windows 与 macOS 共用的单文件命令行入口。"""
    configure_console()
    parser = argparse.ArgumentParser(description="40目标 EEG 键盘")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--diagnose", action="store_true", help="检查依赖、liblsl 和保存目录")
    modes.add_argument("--check-lsl-timing", action="store_true",
                       help="不显示闪烁，连接EEG并检查LSL时间轴及滤波前工频占比")
    modes.add_argument("--self-test-ui", action="store_true", help="运行静态键盘自检")
    modes.add_argument("--offline-session", type=Path, metavar="ZIP",
                       help="离线分析已保存的会话ZIP；不连接LSL或启动显示界面")
    parser.add_argument("--offline-output", type=Path, metavar="DIR",
                        help="离线分析输出目录，仅配合--offline-session；默认offline_results")
    parser.add_argument("--config", type=Path, metavar="TOML",
                        help="从标准 TOML 配置文件加载会话、采集、解码和显示参数")
    parser.add_argument("--seconds", type=positive_seconds, default=5.0,
                        help="静态自检展示或LSL预热后持续检查的秒数")
    parser.add_argument("--warmup-seconds", type=positive_seconds,
                        help="EEG预热秒数；普通启动默认10秒，时间轴诊断默认30秒；"
                             "history模式至少收齐4秒历史及2.1秒检查数据")
    parser.add_argument("--windowed", action="store_true", help="使用1280×800窗口")
    parser.add_argument("--screen", type=int, help="显示器编号，从0开始")
    parser.add_argument("--record-root", help="会话保存目录")
    for signal in SYNC_CHANNELS:
        parser.add_argument("--" + signal.replace("_", "-") + "-channel", type=int,
            help=f"{signal}在同一EEG LSL流内的列号（从0开始）；未提供时记为未获取")
    parser.add_argument("--timing-calibration", help="随会话保存的实测时间校准JSON；不自动应用补偿")
    parser.add_argument("--fbcca-profile", choices=("m3", "m3_5band", "line_robust"),
                        help="m3=论文7子带/90Hz/5谐波，需原始采样率>180Hz；"
                             "m3_5band=5子带/90Hz/3谐波；"
                             "line_robust=48Hz/3子带/3谐波适配方案")
    parser.add_argument("--notch-mode", choices=("epoch", "history"),
                        help="epoch=原短窗陷波；history=使用前4秒真实源数据辅助陷波，"
                             "分析窗不变，仅支持无个人模板的FBCCA，需独立实测")
    parser.add_argument("--line-regression-hz", type=positive_seconds, metavar="HZ",
                        help="在每个分析窗回归去除指定工频的正弦/余弦；仅支持epoch，"
                             "频率须为正且低于Nyquist，默认关闭")
    parser.add_argument("--response-delay", type=nonnegative_seconds, metavar="SECONDS",
                        help="从刺激软件事件到分析窗起点的延迟秒数，允许0；"
                             "此值不是实测设备延迟，默认0.23")
    parser.add_argument("--protocol", choices=("standard_2s", "standard_5s", "paper_window_1p25s"),
                        help="分析窗长：2秒、5秒或论文1.25秒；仅改变窗长，不复刻论文完整时序")
    parser.add_argument("--seed", type=int, help="提示目标随机种子；不同复测可使用不同非负整数")
    parser.add_argument("--participant", help="匿名受试者编号；加载个人模型时必填")
    parser.add_argument("--calibration", help="个人校准模型NPZ、完整模式2会话PKL或ZIP路径（受试者与配置须匹配）")
    args = parser.parse_args(argv)
    if args.screen is not None and args.screen < 0:
        parser.error("screen 必须非负")
    if args.seed is not None and args.seed < 0:
        parser.error("seed 必须非负")
    if args.offline_output is not None and args.offline_session is None:
        parser.error("--offline-output 必须配合 --offline-session")
    if args.offline_session is not None:
        try:
            from .offline import run as run_offline_analysis

            output = (args.offline_output or Path("offline_results")).expanduser()
            print(f"[离线分析] {args.offline_session.expanduser()}", flush=True)
            run_offline_analysis(args.offline_session.expanduser(), output)
            print(f"[离线分析完成] 输出目录：{output.resolve()}", flush=True)
            return 0
        except KeyboardInterrupt:
            print("[离线分析] 已取消", file=sys.stderr, flush=True)
            return 130
        except Exception as exc:
            print(f"[离线分析失败] {type(exc).__name__}：{exc}", file=sys.stderr, flush=True)
            return 1
    report = {
        "mode": "static_ui" if args.self_test_ui else
                "lsl_timing" if args.check_lsl_timing else "diagnose" if args.diagnose else "eeg",
        "status": "failed", "errors": [], "exit_code": 0,
    }
    try:
        print("[启动] 正在加载运行依赖…", flush=True)
        report["checks"] = check_environment(static_ui=args.self_test_ui)
        cfg = Config.from_toml(args.config) if args.config else replace(CONFIG)
        if args.fbcca_profile:
            cfg = configure_fbcca_profile(cfg, args.fbcca_profile)
        if args.protocol:
            cfg.protocol = args.protocol
        if args.notch_mode:
            cfg.notch_mode = args.notch_mode
        if args.line_regression_hz is not None:
            cfg.line_regression_hz = args.line_regression_hz
        if args.response_delay is not None:
            cfg.response_delay_s = args.response_delay
        if args.seed is not None:
            cfg.random_seed = args.seed
        if args.windowed:
            cfg.full_screen, cfg.window_size = False, (1280, 800)
        if args.screen is not None:
            cfg.screen_index = args.screen
        if args.record_root:
            cfg.record_root = args.record_root
        for signal in SYNC_CHANNELS:
            index = getattr(args, signal + "_channel")
            if index is not None:
                setattr(cfg, signal + "_channel", index)
        if args.timing_calibration:
            cfg.timing_calibration_file = args.timing_calibration
        if args.participant:
            cfg.participant_id = args.participant
        if args.calibration:
            cfg.calibration_file = args.calibration
        if args.warmup_seconds is not None:
            cfg.warmup_s = args.warmup_seconds
        elif args.check_lsl_timing:
            cfg.warmup_s = max(cfg.warmup_s, 30.0)
        if args.warmup_seconds is not None or args.check_lsl_timing:
            cfg.startup_timeout_s = max(cfg.startup_timeout_s, cfg.startup_buffer_s + 15.0)
        cfg.validate()
        if args.self_test_ui:
            report.update(static_ui_self_test(cfg, seconds=args.seconds))
        elif args.diagnose:
            report["checks"]["save_directory"] = probe_save_directory(cfg.record_root)
            report["status"] = "passed"
            print("环境诊断通过；尚未验证 EEG 流、真实采样或显示窗口。", flush=True)
        elif args.check_lsl_timing:
            collector = ContinuousLSL(cfg)
            try:
                collector.start()
                observation = report["checks"]["continuous_observation"] = {}
                observe_lsl_timing(cfg, collector, args.seconds, report=observation)
                collector.check_health()
                report["checks"]["acquisition_review"] = preflight_transport(cfg, collector)
                assert collector.buffer is not None
                _, _, local, _, _ = collector.buffer.snapshot()
                age = collector.clock() - local[-1]
                report["checks"]["lsl_timing"] = {
                    "stream": collector.metadata["name"],
                    "declared_fs_hz": collector.fs,
                    "observed_fs_hz": collector.estimated_fs,
                    "latest_sample_age_s": age,
                    "post_warmup_observation_s": args.seconds,
                    "timestamp_processing": collector.metadata["timestamp_processing"],
                }
                if not observation["passed"]:
                    raise RuntimeFault("ACQUISITION",
                        f"观察期间有{observation['invalid_window_count']}个异常窗；"
                        "完整逐窗诊断已保留，不能仅凭最后窗口通过而判定全程通过")
            finally:
                collector.close()
            report["status"] = "passed"
            print(f"LSL时间轴检查通过：{collector.metadata['name']}；"
                  f"估计{collector.estimated_fs:.3f}Hz；最新样本年龄{age:.3f}s。"
                  "此检查不测量设备到屏幕的物理延迟。", flush=True)
        else:
            result = main(cfg=cfg)
            report.update({key: result[key] for key in
                           ("exit_code", "errors", "trial_faults", "stop_reason", "session_artifact")})
            report["status"] = "failed" if report["exit_code"] else "finished"
    except KeyboardInterrupt:
        report.update(status="cancelled", exit_code=130)
    except Exception as exc:
        report["status"] = "failed"
        fault = fault_record(exc)
        report["errors"].append(fault)
        report["exit_code"] = fault["exit_code"]
        print_fault(fault)
    if write_diagnostic(report) is None and not report["exit_code"]:
        report["exit_code"] = 50
    return report["exit_code"]



__all__ = [name for name in globals() if not name.startswith('__')]

