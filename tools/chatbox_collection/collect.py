"""ChatBox SSVEP data-collection paradigm (temporary tool; delete this folder when done).

Shows the real Mode 3 ChatBox keyboard with exactly the online flicker
(phase-0 sinusoid, same frequencies, same layout) and cues one key per trial.
Saves the continuous raw EEG and labelled events through the program's normal
session recorder, so the ZIP has the same format as every other session.

Paradigm (defaults)
-------------------
* One run = one ChatBox page. Page 1: 33 keys (26 letters, 123, SPACE, SEND,
  DEL, 3 suggestion keys). Page 2: 18 keys (digits , . MENU DEL ABC SPACE
  SEND REPLAY). Run page 1, take a break, then run page 2.
* 5 blocks per run. Each block shows every key exactly once in a new random
  order (seeded and saved), never the same key twice in a row across blocks.
* Trial: 1.0 s cue (target key turns yellow, keyboard still) -> 5.5 s flicker
  of all keys (covers the 5 s analysis window plus up to 0.5 s latency search)
  -> 1.0 s still keyboard (blink / swallow here, not during flicker).
* A trial with more than 2 dropped frames is repeated once at the end of its
  block; both copies are saved and marked valid/invalid.
* Starts by itself after a 10 s countdown; rests 30 s after each block and
  continues by itself (SPACE skips a countdown). No key press is needed.
  SPACE during a block pauses after the current trial (SPACE again resumes);
  ESC stops and saves.
* Page 1: 165 trials, about 21 min of trials plus 4 rests (~24 min).
  Page 2: 90 trials, about 11 min plus rests (~14 min).

Usage (from the project folder)
-------------------------------
    .venv\\Scripts\\python.exe tools\\chatbox_collection\\collect.py --page 1
    .venv\\Scripts\\python.exe tools\\chatbox_collection\\collect.py --page 2
Output: session_records\\chatbox_collection\\session_*.zip
Analyse with analyze.py in this folder.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import ssvep_fbcca.app as app_module  # noqa: E402
from ssvep_fbcca.common import AbortSession  # noqa: E402
from ssvep_fbcca.config import Config, configure_fbcca_profile  # noqa: E402
from ssvep_fbcca.events import EventStamp  # noqa: E402
from ssvep_fbcca.ui import PsychoPyKeyboard, _chatbox_display_label  # noqa: E402
from ssvep_fbcca.voice import make_speaker  # noqa: E402

PARADIGM_VERSION = "chatbox-collection-1"


def make_block_orders(n_keys: int, blocks: int, seed: int) -> list[list[int]]:
    """每个区组包含每个键一次的随机顺序；相邻区组交界处不重复同一个键。"""
    rng = random.Random(seed)
    orders: list[list[int]] = []
    for _ in range(blocks):
        order = list(range(n_keys))
        rng.shuffle(order)
        while orders and n_keys > 1 and order[0] == orders[-1][-1]:
            rng.shuffle(order)
        orders.append(order)
    return orders


class ChatBoxCollection(PsychoPyKeyboard):
    """Replaces the mode menu with the collection paradigm; everything else is the real app."""

    params: argparse.Namespace  # set by main() before the window is built

    def run(self) -> str:
        p = self.params
        page = p.page
        symbols = self._chatbox_page_symbols(page)
        frequencies = self._chatbox_page_frequencies(page)
        self._chat_page = page
        self.cfg.session_mode = "chatbox"
        self._apply_screen_layout("chatbox")
        self.output_box.autoDraw = self.output_text.autoDraw = self.output_caption.autoDraw = True
        self._speaker = make_speaker(self.cfg.chatbox_voice_enabled and p.voice,
                                     self.cfg.chatbox_voice_rate)
        if page == 1:
            self._refresh_suggestions()  # realistic word labels on the suggestion keys (local only)
        refresh = self.refresh_hz
        n_frames = math.ceil(p.stimulus_s * refresh)
        t = np.arange(n_frames)[:, None] / refresh
        luminance = .5 * (1 + np.sin(2 * np.pi * t * np.asarray(frequencies)[None, :]))
        frames = np.repeat((2 * luminance - 1)[:, :, None], 3, axis=2)
        orders = make_block_orders(len(symbols), p.blocks, p.seed)
        planned = len(symbols) * p.blocks
        self.markers.mark(EventStamp("collection_start", 0, details={
            "paradigm": PARADIGM_VERSION, "page": page, "symbols": list(symbols),
            "labels": [_chatbox_display_label(s) for s in symbols],
            "suggestion_labels": list(self._suggestions) if page == 1 else [],
            "frequencies_hz": list(frequencies), "phase_rad": 0.0,
            "luminance": "0.5*(1+sin(2*pi*f*t)), frame-sampled at measured refresh",
            "refresh_hz": refresh, "stimulus_frames": n_frames, "stimulus_s": n_frames / refresh,
            "cue_s": p.cue_s, "iti_s": p.iti_s, "rest_s": p.rest_s, "blocks": p.blocks,
            "seed": p.seed, "block_orders": orders, "planned_trials": planned,
            "max_long_frames": p.max_long_frames}))
        counts = {"completed": 0, "valid": 0, "repeated": 0}
        trial_id = 0
        self._intro(len(symbols), planned)
        for block_index, order in enumerate(orders, start=1):
            queue = [(key, False) for key in order]
            self.markers.mark(EventStamp("block_start", 0, details={"block": block_index}))
            position = 0
            while position < len(queue):
                key, is_repeat = queue[position]
                position += 1
                trial_id += 1
                valid = self._trial(trial_id, block_index, position, len(queue), key, is_repeat,
                                    symbols, frequencies, frames)
                counts["completed"] += 1
                counts["valid"] += int(valid)
                if not valid and not is_repeat:
                    queue.append((key, True))
                    counts["repeated"] += 1
                if self._pause_requested:
                    self._rest("Paused by operator.", minimum_s=0.0)
            self.markers.mark(EventStamp("block_end", 0, details={"block": block_index, **counts}))
            if block_index < len(orders):
                self._rest(f"Block {block_index} of {len(orders)} done.", minimum_s=p.rest_s)
        self.markers.mark(EventStamp("collection_end", 0, details=counts))
        self._speak("Recording finished. Thank you.")
        self._show_message("Recording finished. Thank you!\n\nSaving data...", seconds=2.0)
        return f"Collection complete: {counts['valid']} valid of {counts['completed']} trials"

    # --- trial -----------------------------------------------------------------------------
    def _trial(self, trial_id, block, position, block_len, key, is_repeat,
               symbols, frequencies, frames) -> bool:
        p = self.params
        details = {"page": p.page, "block": block, "trial_in_block": position,
                   "class_index": key + 1, "symbol": symbols[key],
                   "label": _chatbox_display_label(symbols[key]) if not symbols[key].startswith("SUGGEST")
                   else self._suggestion_word(symbols[key]),
                   "frequency_hz": float(frequencies[key]), "repeat": is_repeat}
        self.header.text = f"Block {block}/{p.blocks}  ·  Trial {position}/{block_len}"
        self.footer.text = "Look at the yellow key  ·  SPACE: pause after this trial  ·  ESC: stop and save"
        self.chat_status.text = "Recording"
        self._poll_keys()
        # Cue: target turns yellow, keyboard still.
        cue = EventStamp("cue_onset", trial_id, details=details)
        self.win.callOnFlip(self.markers.mark, cue)
        self._hold(p.cue_s, selected_index=key)
        # Stimulus: all keys flicker exactly as online.
        onset = EventStamp("stimulus_onset", trial_id, details=details)
        self.win.recordFrameIntervals = True
        self.win.frameIntervals = []
        try:
            for index, frame in enumerate(frames):
                self._poll_keys()
                self._draw_chatbox_keyboard(p.page, flicker_rgb=frame)
                if index == 0:
                    self.win.callOnFlip(self.markers.mark, onset)
                self.win.flip()
            intervals = np.asarray(self.win.frameIntervals[1:], dtype=float)
            offset = EventStamp("stimulus_offset", trial_id, details=details)
            self.win.callOnFlip(self.markers.mark, offset)
            self._draw_chatbox_keyboard(p.page)
            self.win.flip()
        finally:
            self.win.recordFrameIntervals = False
        long_frames = int(np.sum(intervals > 1.5 / self.refresh_hz)) if len(intervals) else 0
        valid = long_frames <= p.max_long_frames and self.markers.error is None
        self.markers.mark(EventStamp("trial_qc", trial_id, details={
            **details, "long_frames": long_frames, "valid": valid,
            "max_interval_ms": float(intervals.max() * 1000) if len(intervals) else None,
            "flicker_s": float(offset.timestamp - onset.timestamp)}))
        self._hold(p.iti_s)
        return valid

    # --- helpers ---------------------------------------------------------------------------
    _pause_requested = False

    def _poll_keys(self) -> None:
        self._check_abort()  # ESC -> AbortSession (data is still saved)
        if self.event.getKeys(keyList=["space"]):
            self._pause_requested = True
            self.chat_status.text = "Pausing after this trial"

    def _hold(self, seconds: float, selected_index=None) -> None:
        started = self.clock()
        first = True
        while first or self.clock() - started < seconds:
            first = False
            self._poll_keys()
            self._draw_chatbox_keyboard(self.params.page, selected_index=selected_index)
            self.win.flip()

    def _show_message(self, text: str, seconds: float) -> None:
        self.chat_history_text.text = text
        self.chat_scroll_hint.text = ""
        self._hold(seconds)

    def _intro(self, n_keys: int, planned: int) -> None:
        p = self.params
        intro = (f"EEG RECORDING  ·  {n_keys} keys x {p.blocks} blocks = {planned} trials\n\n"
                 "Each trial: one key turns YELLOW. Look at it and keep looking while all keys flicker.\n"
                 "Blink and swallow only while the keys are still. Keep your head still.\n"
                 f"After each block there is a {p.rest_s:g} s rest.")
        print(f"[Collection] Window ready; trials start automatically in {p.start_delay_s:g} s "
              "(SPACE: start now, ESC: stop and save).", flush=True)
        self._speak("Look at the yellow key, and keep looking while the keys flicker.")
        self._countdown(p.start_delay_s, f"ChatBox recording  ·  page {p.page}", intro, "Starting in")

    def _rest(self, title: str, minimum_s: float) -> None:
        self.markers.mark(EventStamp("rest_start", 0, details={"title": title}))
        if self._pause_requested:
            # Operator pressed SPACE during a block: hold until SPACE again.
            self.header.text = "Paused"
            self.footer.text = "SPACE: continue  ·  ESC: stop and save"
            self.chat_history_text.text = f"{title}\n\nPress SPACE to continue."
            self._wait_for_space()
        else:
            self._speak("Rest.")
            self._countdown(minimum_s, "Rest",
                            f"{title}\n\nRest your eyes. Close them if you like.", "Next block in")
        self._pause_requested = False
        self.markers.mark(EventStamp("rest_end", 0))
        self._speak("Get ready.")
        self.header.text = "Get ready"
        self.chat_history_text.text = "Look at the yellow key, and keep looking while the keys flicker."
        self._hold(1.5)

    def _countdown(self, seconds: float, header: str, text: str, label: str) -> None:
        """显示倒计时后自动继续；操作员可按空格提前开始。"""
        self.footer.text = "SPACE: start now  ·  ESC: stop and save"
        self.event.getKeys(keyList=["space"])
        started = self.clock()
        while self.clock() - started < seconds:
            self._check_abort()
            if self.event.getKeys(keyList=["space"]):
                break
            left = math.ceil(seconds - (self.clock() - started))
            self.header.text = f"{header}  ·  {label} {left} s"
            self.chat_history_text.text = f"{text}\n\n{label.upper()}  {left}  s"
            self._draw_chatbox_keyboard(self.params.page)
            self.win.flip()

    def _wait_for_space(self) -> None:
        self.event.getKeys(keyList=["space"])
        while True:
            self._check_abort()
            if self.event.getKeys(keyList=["space"]):
                return
            self._draw_chatbox_keyboard(self.params.page)
            self.win.flip()


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ChatBox SSVEP data collection")
    parser.add_argument("--page", type=int, choices=(1, 2), default=1)
    parser.add_argument("--blocks", type=int, default=5, help="repetitions per key")
    parser.add_argument("--stimulus-s", type=float, default=5.5)
    parser.add_argument("--cue-s", type=float, default=1.0)
    parser.add_argument("--iti-s", type=float, default=1.0)
    parser.add_argument("--rest-s", type=float, default=30.0, help="rest after each block, then auto-continue")
    parser.add_argument("--start-delay-s", type=float, default=10.0, help="countdown before the first trial")
    parser.add_argument("--max-long-frames", type=int, default=2)
    parser.add_argument("--seed", type=int, default=None, help="default: random, saved in the file")
    parser.add_argument("--participant", default=None)
    parser.add_argument("--no-voice", dest="voice", action="store_false")
    parser.add_argument("--windowed", action="store_true", help="1280x800 window (testing only)")
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "default.toml")
    args = parser.parse_args(argv)
    if args.seed is None:
        args.seed = random.SystemRandom().randrange(1, 2**31)
    for name in ("stimulus_s", "cue_s", "iti_s", "rest_s", "start_delay_s"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be >= 0")
    if args.blocks < 1 or args.stimulus_s <= 0:
        parser.error("--blocks and --stimulus-s must be positive")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = configure_fbcca_profile(Config.from_toml(args.config), "m3_5band")
    cfg = replace(cfg, protocol="standard_5s", response_delay_s=0.23, session_mode="chatbox",
                  chatbox_llm_predictions=False,
                  record_root=str(Path(cfg.record_root) / "chatbox_collection"),
                  participant_id=args.participant or cfg.participant_id)
    if args.windowed:
        cfg.full_screen, cfg.window_size = False, (1280, 800)
    print(f"[Collection] page {args.page}, {args.blocks} blocks, seed {args.seed}, "
          f"saving to {cfg.record_root}", flush=True)
    ChatBoxCollection.params = args
    # Reuse the program's own startup, LSL checks, recorder and saving; only the
    # interactive part (mode menu) is replaced by the collection paradigm.
    app_module.PsychoPyKeyboard = ChatBoxCollection
    result = app_module.main(cfg=cfg)
    print(f"[Collection] seed {args.seed}; file: {result.get('session_artifact')}", flush=True)
    return int(result.get("exit_code") or 0)


if __name__ == "__main__":
    raise SystemExit(main())
