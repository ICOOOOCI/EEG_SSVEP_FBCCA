"""模式3字母页的提示式原始 EEG 采集；复用现有界面，不进行在线分类。"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
from pathlib import Path
import random
import uuid
import zipfile

import numpy as np

from .acquisition import ContinuousLSL
from .analysis import frame_diagnostics, preflight_transport
from .common import AbortSession, ROOT, check_environment, probe_save_directory
from .config import Config
from .events import EventMarkers, EventStamp, TrialLedger, _json_safe
from .recording import SessionRecorder
from .ui import (CHAT_BG, CHATBOX_PAGE1_FREQUENCY_BY_SYMBOL, CHATBOX_PAGE1_SYMBOLS,
                 CHATBOX_SUGGESTION_SYMBOLS, PsychoPyKeyboard, _chatbox_display_label)


TASK = "chatbox_letter_collection"
SUGGESTION_WORDS = ("YES", "NO", "HELP")


IDLE_SYMBOL = "IDLE"


def trial_schedule(repeats: int, seed: int, idle_per_block: int = 0) -> list[dict]:
    """一次会话中每类重复 repeats 次；每个平衡采集段各类出现一次。

    idle_per_block 个 IDLE 试次（标签0）随机混入每段：键盘照常闪烁，但用户
    看聊天区文字、不看任何键，用于标定“不打算输入”时的误选率。
    """
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats 必须是正整数")
    if isinstance(idle_per_block, bool) or not isinstance(idle_per_block, int) or idle_per_block < 0:
        raise ValueError("idle_per_block 必须是非负整数")
    rng = random.Random(seed)
    trials = []
    for block in range(1, repeats + 1):
        symbols = list(CHATBOX_PAGE1_SYMBOLS) + [IDLE_SYMBOL] * idle_per_block
        rng.shuffle(symbols)
        if trials and symbols[0] == trials[-1]["symbol"]:
            symbols[0], symbols[1] = symbols[1], symbols[0]
        for symbol in symbols:
            if symbol == IDLE_SYMBOL:
                label, display, frequency = 0, IDLE_SYMBOL, None
            else:
                label = CHATBOX_PAGE1_SYMBOLS.index(symbol) + 1
                display = (SUGGESTION_WORDS[CHATBOX_SUGGESTION_SYMBOLS.index(symbol)]
                           if symbol in CHATBOX_SUGGESTION_SYMBOLS else _chatbox_display_label(symbol))
                frequency = CHATBOX_PAGE1_FREQUENCY_BY_SYMBOL[symbol]
            trials.append({"trial_id": len(trials) + 1, "collection_round": 1,
                           "block_id": block, "repetition": block, "label": label,
                           "symbol": symbol, "display_label": display,
                           "frequency_hz": frequency, "status": "pending"})
    return trials


class CollectionKeyboard(PsychoPyKeyboard):
    """只显示现有字母页；提示使用现有聊天文字和选中边框。"""

    def _build_visuals(self):
        layout = super()._build_visuals()
        for item in self.chat_page_visuals[1]:
            if item["symbol"] in CHATBOX_SUGGESTION_SYMBOLS:
                item["label"].text = SUGGESTION_WORDS[
                    CHATBOX_SUGGESTION_SYMBOLS.index(item["symbol"])]
        # 对话面板在采集中始终收起，提示文字改放进顶部输入框。
        self.chat_empty_text.pos = self.chat_draft_box.pos
        return layout

    def _draw_draft(self):
        self.chat_draft_box.draw()
        self.chat_empty_text.draw()

    def _draw_startup_check(self):
        self.win.color = CHAT_BG
        self.chat_empty_text.text = "EEG collection: letter page / 34 targets"
        self._draw_chatbox_keyboard(1)


def _markers(cfg, clock):
    """Marker 流的目标描述采用本页34类编号，不复用旧40类编号。"""
    from mne_lsl.lsl import StreamInfo, StreamOutlet

    info = StreamInfo(cfg.marker_stream_name, "Markers", 1, 0, "string",
                      f"chatbox-collection-{uuid.uuid4()}")
    info.desc.append_child_value("label_namespace", TASK)
    info.desc.append_child_value("clock", "local_clock in PsychoPy callOnFlip")
    info.desc.append_child_value("physical_display_latency", "UNMEASURED")
    targets = info.desc.append_child("targets")
    for label, symbol in enumerate(CHATBOX_PAGE1_SYMBOLS, 1):
        node = targets.append_child("target")
        for key, value in {"label": label, "symbol": symbol,
                           "frequency_hz": CHATBOX_PAGE1_FREQUENCY_BY_SYMBOL[symbol]}.items():
            node.append_child_value(key, str(value))
    return EventMarkers(cfg, clock, outlet=StreamOutlet(info, chunk_size=1))


def _phase(app, markers, trial, name, seconds, *, flicker=False, highlight=False):
    """首帧和首个静止帧分别记录起止事件；刺激阶段不保留提示边框。"""
    details = {key: trial[key] for key in (
        "label", "symbol", "display_label", "frequency_hz", "collection_round", "block_id", "repetition")}
    details["label_namespace"] = TASK
    onset = EventStamp(f"chatbox_collection_{name}_onset", trial["trial_id"],
                       trial["label"], details)
    offset = EventStamp(f"chatbox_collection_{name}_offset", trial["trial_id"],
                        trial["label"], details)
    flips, lsl_times = [], []
    frame = 0
    frequencies = np.asarray(app._chatbox_page_frequencies(1))
    previous = app.win.recordFrameIntervals
    app.win.recordFrameIntervals = flicker
    try:
        while onset.timestamp is None or app.clock() - onset.timestamp < seconds:
            app._check_abort()
            if markers.error is not None:
                raise RuntimeError(f"LSL Marker 发送失败：{markers.error}")
            app._draw_chatbox_keyboard(
                1, selected_index=trial["label"] - 1 if highlight and trial["label"] > 0 else None,
                flicker_rgb=app._flicker_rgb(frequencies, frame) if flicker else None)
            if frame == 0:
                app.win.callOnFlip(markers.mark, onset)
            if flicker:
                app.win.callOnFlip(lambda: lsl_times.append(app.clock()))
            flip = app.win.flip()
            if flicker:
                flips.append(flip)
            frame += 1
        app._draw_chatbox_keyboard(1)
        app.win.callOnFlip(markers.mark, offset)
        offset_flip = app.win.flip()
    finally:
        app.win.recordFrameIntervals = previous
    result = {"onset": onset.timestamp, "offset": offset.timestamp}
    if flicker:
        quality = frame_diagnostics(
            flips + [offset_flip], app.refresh_hz, app.cfg.frame_long_factor,
            app.cfg.frame_short_factor, app.cfg.max_warning_frame_anomalies,
            app.cfg.frame_rate_tolerance)
        quality.pop("intervals_s", None)
        result.update(frame_flip_times=flips, frame_lsl_times=lsl_times,
                      offset_flip_time=offset_flip, frame_quality=quality)
    return result


def _eeg_quality(timestamps, eeg, onset, offset, fs, cfg):
    """只检查原始样本和时间轴；不滤波、不插值、不修改接收值。"""
    problems = []
    report = {"sample_count": len(timestamps), "nominal_fs_hz": fs}
    if len(timestamps) < 3:
        problems.append("insufficient_samples")
    elif not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
        problems.append("invalid_timestamps")
    else:
        dt = np.diff(timestamps)
        estimated = (len(timestamps) - 1) / (timestamps[-1] - timestamps[0])
        report.update(timestamp_estimated_fs=float(estimated), max_gap_s=float(dt.max()),
                      rate_error_fraction=float(abs(estimated / fs - 1)))
        if timestamps[0] - onset > 1.5 / fs or offset - timestamps[-1] > 1.5 / fs:
            problems.append("incomplete_eeg_coverage")
        # LSL 绝对时钟较大；减去浮点容差，避免恰好2个采样周期的缺口漏检。
        if np.any(dt >= cfg.hard_gap_factor / fs - 1e-9):
            problems.append("eeg_timestamp_gap")
        if abs(estimated / fs - 1) > cfg.hard_rate_tolerance + 1e-12:
            problems.append("eeg_sampling_rate_mismatch")
    if not np.isfinite(eeg).all():
        problems.append("nonfinite_eeg")
    report.update(valid=not problems, problems=problems)
    return report


def _package_collection(path: Path, labels: dict, events, cfg: Config):
    """为标准连续录制 ZIP 添加本页标签、原始试次和独立事件映射。"""
    temporary = path.with_name(path.name + ".collection.tmp")
    with zipfile.ZipFile(path) as source:
        info = json.loads(source.read("info.json"))
        with np.load(io.BytesIO(source.read("eeg_raw.npz")), allow_pickle=False) as raw:
            t, data = raw["timestamps"], raw["data"]
            channels, fs = raw["eeg_channel_indices"], float(raw["fs"])
            if not math.isfinite(fs) or fs <= 0:
                raise RuntimeError("连续 EEG 缺少有效标称采样率，原始 ZIP 已保留")
            arrays = {"channel_indices": channels, "fs": np.asarray(fs),
                      "channel_labels": raw["channel_names"][channels]}
            retained = []
            for trial in labels["trials"]:
                if trial["status"] != "completed":
                    continue
                onset, offset = trial["stimulus_onset"], trial["stimulus_offset"]
                mask = (t >= onset) & (t < offset)
                eeg, stamps = data[channels][:, mask], t[mask]
                prefix = f"trial_{trial['trial_id']:04d}_"
                arrays.update({prefix + "eeg": eeg, prefix + "timestamps": stamps,
                               prefix + "label": np.asarray(trial["label"]),
                               prefix + "frame_lsl_times": np.asarray(trial["frame_lsl_times"]),
                               prefix + "frame_flip_times": np.asarray(trial["frame_flip_times"])})
                trial["eeg_quality"] = _eeg_quality(stamps, eeg, onset, offset, fs, cfg)
                trial["usable"] = bool(trial["eeg_quality"]["valid"] and trial["frame_quality"]["valid"])
                retained.append(trial)
            arrays.update(trial_ids=np.asarray([t["trial_id"] for t in retained], dtype=np.int64),
                          labels=np.asarray([t["label"] for t in retained], dtype=np.int64),
                          block_ids=np.asarray([t["block_id"] for t in retained], dtype=np.int64),
                          has_eeg=np.asarray([arrays[f"trial_{t['trial_id']:04d}_eeg"].shape[1] > 0
                                              for t in retained], dtype=bool),
                          usable=np.asarray([t["usable"] for t in retained], dtype=bool),
                          predictions=np.full(len(retained), -1, dtype=np.int64))
        labels["completed_trials"] = len(retained)
        labels["usable_trials"] = sum(t["usable"] for t in retained)
        class_labels = sorted(int(label) for label in labels["label_map"])
        labels["completed_per_class"] = {
            str(label): sum(t["label"] == label for t in retained) for label in class_labels}
        labels["usable_per_class"] = {
            str(label): sum(t["label"] == label and t["usable"] for t in retained)
            for label in class_labels}
        info.update(mode=TASK, page=1, label_namespace=TASK, label_map=labels["label_map"],
                    protocol=labels["protocol"], trials=labels["trials"],
                    stop_reason=labels["stop_reason"],
                    summary={"task": TASK, "planned_trials": len(labels["trials"]),
                             "completed_trials": len(retained), "usable_trials": labels["usable_trials"],
                             "online_decoding": False, "accuracy": None})
        trial_buffer = io.BytesIO()
        np.savez_compressed(trial_buffer, **arrays)
        csv_buffer = io.StringIO(newline="")
        writer = csv.DictWriter(csv_buffer, fieldnames=(
            "timestamp", "event", "trial_id", "label", "symbol", "frequency_hz", "block_id", "details"))
        writer.writeheader()
        for event in events:
            detail = event.details
            writer.writerow({"timestamp": event.timestamp, "event": event.kind,
                             "trial_id": event.trial_id, "label": detail.get("label"),
                             "symbol": detail.get("symbol"), "frequency_hz": detail.get("frequency_hz"),
                             "block_id": detail.get("block_id"),
                             "details": json.dumps(_json_safe(detail), ensure_ascii=False)})
        readme = (
            "模式3字母页34类提示式采集，单次会话；没有在线分类、打字、发消息或切页。\n"
            "标签1起，严格按 chatbox_labels.json 的 label_map；不是旧40类编号。\n"
            "eeg_raw.npz：全部LSL通道的连续原始接收值，含预热、提示、刺激、休息。\n"
            "trials.npz：按刺激起止截取的选定通道原始值，不滤波、不插值、不补零；含质量失败试次。\n"
            "labels/trial_ids/block_ids/usable 对应同序试次；predictions=-1 表示未分类。\n"
            "chatbox_labels.json：完整计划、真实软件起止、逐帧时间、各类次数及质量检查。\n"
            "events.csv：本页正确映射的提示/刺激/休息事件；均为LSL软件时间，非光电出光时间。\n"
            "后续分析先筛 status=completed 且 usable=true；质量失败不自动增加采集次数。\n"
            f"本次计划刺激{labels['protocol']['stimulus_s']:g}秒；建议从刺激后"
            f"{cfg.response_delay_s:g}秒起取{cfg.chat_window_s:g}秒分析，保留整段供离线调整。\n"
            "采集段block_id用于按试次/段留出验证；同一试次的重叠窗不得跨训练和验证。\n"
            + ("标签0为IDLE：键盘照常闪烁，用户看聊天区文字、不看任何键，可用于标定空闲误触发率。\n"
               if labels["protocol"]["idle_collected"] else
               "未采集IDLE类，不能用本数据标定空闲误触发率。\n")
            + "标签是注视指令，未验证实际眼动。\n"
            "接收端未处理不代表上游未滤波；LSL时间戳已同步/去抖，不是硬件原始时间戳。\n")
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as destination:
            replaced = {"info.json", "trials.npz", "README.txt", "events.csv", "chatbox_labels.json"}
            for name in source.namelist():
                if name not in replaced:
                    destination.writestr(name, source.read(name), compress_type=zipfile.ZIP_STORED
                                         if name.endswith(".npz") else zipfile.ZIP_DEFLATED)
            destination.writestr("info.json", json.dumps(_json_safe(info), ensure_ascii=False, indent=2))
            destination.writestr("chatbox_labels.json", json.dumps(_json_safe(labels), ensure_ascii=False, indent=2))
            destination.writestr("trials.npz", trial_buffer.getvalue(), compress_type=zipfile.ZIP_STORED)
            destination.writestr("events.csv", "\ufeff" + csv_buffer.getvalue())
            destination.writestr("README.txt", readme)
    with zipfile.ZipFile(temporary) as result:
        if result.testzip() is not None:
            raise RuntimeError("采集 ZIP 校验失败，原始 ZIP 已保留")
    os.replace(temporary, path)


def _seconds(value):
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("秒数必须为有限正数")
    return result


def cli(argv=None):
    parser = argparse.ArgumentParser(description="直接进入模式3字母页，34类每类3次，共一轮")
    parser.add_argument("--config", type=Path, default=ROOT / "config/default.toml")
    parser.add_argument("--repeats", type=int, default=3, help="同一次会话内每类次数，默认3")
    parser.add_argument("--idle-per-block", type=int, default=0,
                        help="每段混入的 IDLE 试次数（看聊天区、不看键），默认0")
    parser.add_argument("--participant", default="P01")
    parser.add_argument("--record-root", default="chatbox_collection")
    parser.add_argument("--seed", type=int, default=20261103)
    parser.add_argument("--cue-seconds", type=_seconds, default=2.)
    parser.add_argument("--stimulus-seconds", type=_seconds,
                        help="刺激秒数；默认至少5秒，并向上取整以覆盖响应延迟和分析窗")
    parser.add_argument("--rest-seconds", type=_seconds, default=2.)
    parser.add_argument("--block-rest-seconds", type=_seconds, default=30.)
    parser.add_argument("--screen", type=int)
    parser.add_argument("--windowed", dest="full_screen", action="store_false")
    parser.add_argument("--fullscreen", dest="full_screen", action="store_true")
    parser.set_defaults(full_screen=None)
    parser.add_argument("--dry-run", action="store_true", help="只打印完整计划，不连接EEG、不显示闪烁")
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("--repeats 必须为正整数")
    if args.idle_per_block < 0:
        parser.error("--idle-per-block 必须非负")
    if args.screen is not None and args.screen < 0:
        parser.error("--screen 必须非负")
    cfg = Config.from_toml(args.config)
    if not cfg.chatbox_predictions_enabled:
        parser.error("本采集要求现有字母页全部34类，请启用 chatbox_predictions_enabled")
    if args.stimulus_seconds is None:
        args.stimulus_seconds = max(5., float(math.ceil(cfg.response_delay_s + cfg.chat_window_s)))
    if args.stimulus_seconds < cfg.response_delay_s + cfg.chat_window_s:
        parser.error("刺激时长必须覆盖 response_delay_s + chat_window_s")
    cfg.record_root, cfg.participant_id = args.record_root, args.participant
    if args.full_screen is not None:
        cfg.full_screen = args.full_screen
    if args.screen is not None:
        cfg.screen_index = args.screen
    cfg.validate()
    trials = trial_schedule(args.repeats, args.seed, args.idle_per_block)
    duration = len(trials) * (args.cue_seconds + args.stimulus_seconds + args.rest_seconds)
    duration += (args.repeats - 1) * args.block_rest_seconds
    labels = {"task": TASK, "page": 1, "participant": args.participant, "seed": args.seed,
              "collection_rounds": 1, "planned_trials": len(trials), "trials": trials,
              "protocol": {"cue_s": args.cue_seconds, "stimulus_s": args.stimulus_seconds,
                           "rest_s": args.rest_seconds, "block_rest_s": args.block_rest_seconds,
                           "blocks": args.repeats, "repeats_per_class": args.repeats,
                           "estimated_seconds_excluding_setup": duration,
                           "cue_highlight_during_stimulus": False,
                           "all_targets_flicker": True, "online_decoding": False,
                           "idle_collected": args.idle_per_block > 0,
                           "idle_per_block": args.idle_per_block,
                           "idle_instruction": "keyboard flickers; look at the chat text, not at any key",
                           "suggestion_words": SUGGESTION_WORDS,
                           "analysis_delay_s": cfg.response_delay_s,
                           "analysis_window_s": cfg.chat_window_s,
                           "online_gaze_shift_s": cfg.chat_gaze_shift_s,
                           "online_min_margin": cfg.chat_min_margin},
              "label_map": {**({"0": {"symbol": IDLE_SYMBOL, "frequency_hz": None}}
                               if args.idle_per_block else {}),
                            **{str(i): {"symbol": s, "frequency_hz": CHATBOX_PAGE1_FREQUENCY_BY_SYMBOL[s]}
                               for i, s in enumerate(CHATBOX_PAGE1_SYMBOLS, 1)}},
              "completed_trials": 0, "stop_reason": "not_started"}
    print(f"计划：一轮，34类 × {args.repeats}次 + IDLE {args.idle_per_block}×{args.repeats}次"
          f" = {len(trials)}试次；"
          f"提示{args.cue_seconds:g}秒、刺激{args.stimulus_seconds:g}秒、休息{args.rest_seconds:g}秒；"
          f"每段之间额外休息{args.block_rest_seconds:g}秒；"
          f"约{int(duration // 60)}分{int(duration % 60)}秒，不含连接/预热。", flush=True)
    if args.dry_run:
        print(json.dumps(labels, ensure_ascii=False, indent=2))
        return 0
    check_environment()
    probe_save_directory(cfg.record_root)
    collector, app, markers = ContinuousLSL(cfg), None, None
    recorder = SessionRecorder(cfg, context={"mode_at_start": TASK, "protocol": labels["protocol"]})
    collector.recorder = recorder
    sidecar = recorder.archive_path.with_suffix(".chatbox_labels.json")
    reason, exit_code = "completed", 0
    review = {}
    try:
        SessionRecorder._atomic_json(sidecar, labels)
        print("正在连接 EEG LSL 并预热；随后直接显示字母页。", flush=True)
        collector.start()
        review = preflight_transport(cfg, collector)
        markers = _markers(cfg, collector.clock)
        app = CollectionKeyboard(cfg, collector, markers, TrialLedger(), static_only=True)
        labels["display_info"] = app.display_info
        app.chat_empty_text.text = f"{len(trials)} trials / SPACE: start / ESC: stop and save"
        app.event.clearEvents()
        while True:
            app._check_abort()
            app._draw_chatbox_keyboard(1)
            app.win.flip()
            if app.event.getKeys(keyList=["space"]):
                break
        for index, trial in enumerate(trials):
            app._graphics_gc.collect(app.win)
            trial["status"] = "started"
            progress = f"{trial['trial_id']}/{len(trials)}"
            target = trial["display_label"]
            idle = trial["label"] == 0
            print(f"[{progress}] " + ("IDLE：看聊天区文字，不看键" if idle else
                                      f"注视 {target} / {trial['frequency_hz']:g} Hz"), flush=True)
            app.chat_empty_text.text = (f"{progress}  IDLE - read this line, do NOT look at the keys"
                                        if idle else f"{progress}  LOOK AT: {target}")
            cue = _phase(app, markers, trial, "cue", args.cue_seconds, highlight=True)
            trial["cue_onset"], trial["cue_offset"] = cue["onset"], cue["offset"]
            app.chat_empty_text.text = (f"{progress}  IDLE - keep reading here, ignore the flicker"
                                        if idle else f"{progress}  KEEP LOOKING AT: {target}")
            stimulus = _phase(app, markers, trial, "stimulus", args.stimulus_seconds, flicker=True)
            trial.update(stimulus_onset=stimulus.pop("onset"), stimulus_offset=stimulus.pop("offset"),
                         **stimulus, status="completed")
            trial["actual_stimulus_s"] = trial["stimulus_offset"] - trial["stimulus_onset"]
            labels["completed_trials"] += 1
            app.chat_empty_text.text = f"{progress}  REST - blink and relax"
            rest = _phase(app, markers, trial, "rest", args.rest_seconds)
            trial["rest_onset"], trial["rest_offset"] = rest["onset"], rest["offset"]
            SessionRecorder._atomic_json(sidecar, labels)
            if index + 1 < len(trials) and trials[index + 1]["block_id"] != trial["block_id"]:
                app.chat_empty_text.text = f"{progress}  BREAK {args.block_rest_seconds:g}s - next block starts automatically"
                _phase(app, markers, trial, "block_rest", args.block_rest_seconds)
    except (AbortSession, KeyboardInterrupt):
        reason, exit_code = "aborted", 130
    except Exception as exc:
        reason, exit_code = f"{type(exc).__name__}: {exc}", 1
        print(f"[采集停止] {reason}", flush=True)
    finally:
        for trial in trials:
            if trial["status"] == "started":
                trial["status"] = "aborted"
        for resource in (app, collector, markers):
            if resource is not None:
                try:
                    resource.close()
                except Exception as exc:
                    reason, exit_code = reason + f"; close: {exc}", 1
        if markers is not None and markers.error is not None:
            reason, exit_code = reason + f"; marker: {markers.error}", 1
        labels["stop_reason"] = reason
        labels["completed_trials"] = sum(t["status"] == "completed" for t in trials)
        SessionRecorder._atomic_json(sidecar, labels)
        try:
            artifact = recorder.finalize(
                summary={"completed_trials": labels["completed_trials"], "task": TASK},
                events=markers.events if markers else [], acquisition_metadata=collector.metadata,
                acquisition_review=review, clock_updates=collector.clock_updates,
                display_info=labels.get("display_info", {}), typed_text="", stop_reason=reason)
            if artifact is not None:
                _package_collection(Path(artifact), labels, markers.events if markers else [], cfg)
                SessionRecorder._atomic_json(sidecar, labels)
                print(f"已呈现 {labels['completed_trials']}/{len(trials)} 次；"
                      f"EEG/软件时序通过 {labels['usable_trials']} 次：{artifact}", flush=True)
                if reason == "completed" and labels["usable_trials"] != len(trials):
                    exit_code = 1
                    print("部分试次质量未通过，原始数据已保留；未自动追加采集。", flush=True)
            else:
                exit_code = exit_code or 1
                print(f"未生成 EEG 会话；采集计划/停止原因：{sidecar}", flush=True)
        except Exception as exc:
            exit_code = 1
            print(f"[保存失败] {exc}；请保留 {recorder.session_dir}、原始ZIP及 {sidecar}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(cli())
