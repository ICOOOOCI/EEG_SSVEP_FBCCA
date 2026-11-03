"""主菜单三分类原始 EEG 采集，每轮每类一次，不进行在线分类。"""

import argparse
import io
import json
import os
from pathlib import Path
import random
import zipfile

import numpy as np

from .acquisition import ContinuousLSL
from .common import AbortSession, ROOT, check_environment, probe_save_directory
from .config import Config
from .events import EventMarkers, EventStamp, TrialLedger
from .recording import SessionRecorder
from .ui import MODE_FREQUENCIES_HZ, MODE_NAMES, PsychoPyKeyboard


def trial_schedule(repeats, seed):
    rng = random.Random(seed)
    trials = []
    for round_id in range(1, repeats + 1):
        labels = [1, 2, 3]
        rng.shuffle(labels)
        for label in labels:
            trials.append(dict(trial_id=len(trials) + 1, round=round_id, label=label,
                               target_name=MODE_NAMES[label - 1],
                               frequency_hz=MODE_FREQUENCIES_HZ[label - 1], status="pending"))
    return trials


def _package_menu(path, labels):
    """保留连续原始数组，并按真实刺激时间截取九个试次；不插值或滤波。"""
    temporary = path.with_suffix('.menu.tmp')
    with zipfile.ZipFile(path) as source:
        info = json.loads(source.read('info.json'))
        with np.load(io.BytesIO(source.read('eeg_raw.npz')), allow_pickle=False) as raw:
            t, data = raw['timestamps'], raw['data']
            channels = raw['eeg_channel_indices']
            trial_arrays = {'channel_indices': channels, 'fs': raw['fs']}
            for trial in labels['trials']:
                if trial['status'] != 'completed':
                    continue
                onset, offset = trial['stimulus_onset'], trial['stimulus_offset']
                if not len(t) or t[0] > onset or t[-1] + 1 / float(raw['fs']) < offset:
                    trial['status'] = 'incomplete_eeg_coverage'
                    continue
                mask = (t >= onset) & (t < offset)
                prefix = f"trial_{trial['trial_id']:04d}_"
                trial_arrays[prefix + 'eeg'] = data[channels][:, mask]
                trial_arrays[prefix + 'timestamps'] = t[mask]
                trial_arrays[prefix + 'label'] = np.asarray(trial['label'])
        labels['completed_trials'] = sum(t['status'] == 'completed' for t in labels['trials'])
        info.update(mode='menu_collection', label_map=labels['label_map'], trials=labels['trials'],
                    summary={'completed_trials': labels['completed_trials'], 'task': 'menu_collection'})
        trial_buffer = io.BytesIO()
        np.savez_compressed(trial_buffer, **trial_arrays)
        readme = ('主菜单三分类采集；未进行在线分类。\n'
                  'eeg_raw.npz：全部 LSL 通道的连续原始值，接收端未滤波、未重采样；上游可能已处理。\n'
                  'timestamps：与事件同一 LSL 时钟，已由 LSL 同步/去抖；不是原始硬件时间戳。\n'
                  'trials.npz：按刺激起止时间截取的选定 EEG 通道；没有补零或插值。\n'
                  'menu_labels.json：1=WHEELCHAIR(8Hz)，2=ROBOT ARM(15Hz)，3=AI CHAT(12Hz)。\n'
                  '只使用 status=completed 的试次。没有采集 IDLE 类。\n'
                  '标签为要求注视的目标；未确认眼动、屏幕出光时间或上游设备丢样。\n')
        with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as destination:
            for name in source.namelist():
                if name not in ('info.json', 'trials.npz', 'README.txt'):
                    destination.writestr(name, source.read(name))
            destination.writestr('info.json', json.dumps(info, ensure_ascii=False, indent=2))
            destination.writestr('menu_labels.json', json.dumps(labels, ensure_ascii=False, indent=2))
            destination.writestr('trials.npz', trial_buffer.getvalue())
            destination.writestr('README.txt', readme)
    with zipfile.ZipFile(temporary) as result:
        if result.testzip() is not None:
            raise RuntimeError('菜单采集 ZIP 校验失败，保留原始会话')
    os.replace(temporary, path)


def _phase(app, markers, trial, name, seconds, flicker=False, highlight=False):
    index = trial['label'] - 1
    stamp = EventStamp('menu_' + name + '_onset', trial['trial_id'], details={
        key: trial[key] for key in ('label', 'frequency_hz', 'round', 'target_name')})
    frame = 0
    times = []
    while stamp.timestamp is None or app.clock() - stamp.timestamp < seconds:
        app._check_abort()
        app._draw_mode_menu(selected=index if highlight else None,
                           flicker_rgb=app._flicker_rgb(np.asarray(MODE_FREQUENCIES_HZ), frame)
                           if flicker else None)
        if frame == 0:
            app.win.callOnFlip(markers.mark, stamp)
        if flicker:
            app.win.callOnFlip(lambda: times.append(app.clock()))
        app.win.flip()
        frame += 1
    offset = EventStamp('menu_' + name + '_offset', trial['trial_id'])
    app._draw_mode_menu()
    app.win.callOnFlip(markers.mark, offset)
    app.win.flip()
    return stamp.timestamp, offset.timestamp, times


def cli(argv=None):
    parser = argparse.ArgumentParser(description='主菜单三类原始 EEG，每类默认采集3次')
    parser.add_argument('--config', type=Path, default=ROOT / 'config/default.toml')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--participant', default='P01')
    parser.add_argument('--record-root', default='menu_collection')
    parser.add_argument('--seed', type=int, default=20261102)
    parser.add_argument('--dry-run', action='store_true', help='仅打印采集计划，不连接 EEG 或显示闪烁')
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error('--repeats 必须为正整数')
    trials = trial_schedule(args.repeats, args.seed)
    print(f'计划：3类 × {args.repeats}次 = {len(trials)}试次；提示1秒、刺激5秒、休息2秒。', flush=True)
    if args.dry_run:
        print(json.dumps(trials, ensure_ascii=False, indent=2))
        return 0
    cfg = Config.from_toml(args.config)
    cfg.record_root, cfg.participant_id = args.record_root, args.participant
    check_environment()
    probe_save_directory(cfg.record_root)
    collector, app, markers = ContinuousLSL(cfg), None, None
    recorder = SessionRecorder(cfg, context={'mode_at_start': 'menu_collection'})
    collector.recorder = recorder
    labels = {'task': 'menu_collection', 'participant': args.participant, 'seed': args.seed,
              'label_map': {str(i + 1): {'name': name, 'frequency_hz': frequency}
                            for i, (name, frequency) in enumerate(zip(MODE_NAMES, MODE_FREQUENCIES_HZ))},
              'trials': trials, 'completed_trials': 0,
              'timing_scope': '软件翻帧 LSL 时间，未测量实际出光延迟'}
    sidecar = recorder.archive_path.with_suffix('.menu_labels.json')
    reason, exit_code = 'completed', 0
    try:
        collector.start()
        markers = EventMarkers(cfg, collector.clock)
        app = PsychoPyKeyboard(cfg, collector, markers, TrialLedger(), static_only=True)
        labels['display_info'] = app.display_info
        app.mode_status.text = 'Look at the indicated target. Press SPACE to start; ESC to stop.'
        app.event.clearEvents()
        while True:
            app._check_abort()
            app._draw_mode_menu()
            app.win.flip()
            if app.event.getKeys(keyList=['space']):
                break
        for trial in trials:
            trial['status'] = 'started'
            app.mode_status.text = f"{trial['trial_id']}/{len(trials)}  LOOK AT: {trial['target_name']}"
            trial['cue_onset'], _, _ = _phase(app, markers, trial, 'cue', 1., highlight=True)
            trial['stimulus_onset'], trial['stimulus_offset'], trial['frame_lsl_times'] = _phase(
                app, markers, trial, 'stimulus', 5., flicker=True, highlight=True)
            trial['status'] = 'completed'
            SessionRecorder._atomic_json(sidecar, labels)
            app.mode_status.text = 'REST'
            _phase(app, markers, trial, 'rest', 2.)
    except (AbortSession, KeyboardInterrupt):
        reason, exit_code = 'aborted', 130
    except Exception as exc:
        reason, exit_code = f'{type(exc).__name__}: {exc}', 1
        print(reason, flush=True)
    finally:
        errors = []
        for resource in (app, collector, markers):
            if resource is not None:
                try:
                    resource.close()
                except Exception as exc:
                    errors.append(str(exc))
        if errors:
            exit_code, reason = 1, reason + '; ' + '; '.join(errors)
        for trial in trials:
            if trial['status'] == 'started':
                trial['status'] = 'aborted'
        labels['stop_reason'] = reason
        labels['completed_trials'] = sum(t['status'] == 'completed' for t in trials)
        SessionRecorder._atomic_json(sidecar, labels)
        artifact = recorder.finalize(
            summary={'completed_trials': labels['completed_trials'], 'task': 'menu_collection'},
            events=markers.events if markers else [], acquisition_metadata=collector.metadata,
            acquisition_review={}, clock_updates=collector.clock_updates,
            display_info=labels.get('display_info', {}), typed_text='', stop_reason=reason)
        _package_menu(Path(artifact), labels)
        SessionRecorder._atomic_json(sidecar, labels)
        print(f"已保存 {labels['completed_trials']}/{len(trials)} 个完整试次：{artifact}", flush=True)
    return exit_code


if __name__ == '__main__':
    raise SystemExit(cli())
