"""Offline analysis for ChatBox collection sessions (temporary tool; delete with this folder).

1. Cuts one epoch per trial from the continuous raw EEG (resampled onto a uniform
   grid from the LSL timestamps) and saves them as a plain NPZ for your own work:
       X        trials x channels x samples   (raw, unfiltered, EEG channels only)
       times    seconds relative to flicker onset (starts at -pre_s)
       y        1-based key index on the page; freq / symbol / block / valid per trial
2. Scores the current FBCCA (same code and settings as online) at several window
   lengths, as a baseline to beat.

Usage:
    .venv\\Scripts\\python.exe tools\\chatbox_collection\\analyze.py session_records\\chatbox_collection\\session_XXXX.zip
Options: --delay 0.23 (latency added after onset), --windows 1 2 3 4 5, --out DIR
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ssvep_fbcca.analysis import itr_bits_per_minute  # noqa: E402
from ssvep_fbcca.config import Config, configure_fbcca_profile  # noqa: E402
from ssvep_fbcca.signal import classify_eeg_window  # noqa: E402


def load_session(path: Path) -> dict:
    """读取连续EEG与事件表。"""
    with zipfile.ZipFile(path) as archive:
        with np.load(io.BytesIO(archive.read("eeg_raw.npz")), allow_pickle=False) as raw:
            eeg = {key: raw[key] for key in raw.files}
        events = list(csv.DictReader(io.StringIO(archive.read("events.csv").decode("utf-8-sig"))))
        info = json.loads(archive.read("info.json"))
    for event in events:
        event["timestamp"] = float(event["timestamp"]) if event["timestamp"] else None
        event["details"] = json.loads(event["details"] or "{}")
    return {"eeg": eeg, "events": events, "info": info}


def extract_epochs(session: dict, pre_s: float, post_s: float) -> dict:
    """以 stimulus_onset 为零点截取 [-pre, stimulus+post] 的均匀采样数据。"""
    eeg, events = session["eeg"], session["events"]
    start = next(e for e in events if e["event"] == "collection_start")["details"]
    fs = float(eeg["fs"])
    data = np.asarray(eeg["data"], dtype=float)[np.asarray(eeg["eeg_channel_indices"], dtype=int)]
    stamps = np.asarray(eeg["timestamps"], dtype=float)
    qc = {e["trial_id"]: e["details"] for e in events if e["event"] == "trial_qc"}
    onsets = [e for e in events if e["event"] == "stimulus_onset"]
    times = np.arange(-round(pre_s * fs), round((start["stimulus_s"] + post_s) * fs)) / fs
    X, keep, rows = [], [], []
    for event in onsets:
        grid = event["timestamp"] + times
        if grid[0] < stamps[0] or grid[-1] > stamps[-1]:
            continue  # trial not fully covered by the recording (e.g. stopped with ESC)
        X.append(np.vstack([np.interp(grid, stamps, channel) for channel in data]))
        d = event["details"]
        trial_qc = qc.get(event["trial_id"], {})
        rows.append((d["class_index"], d["frequency_hz"], d["symbol"], d.get("label") or d["symbol"],
                     d["block"], bool(trial_qc.get("valid", False)), int(event["trial_id"])))
    if not X:
        raise SystemExit("No complete trials found in this session.")
    y, freq, symbol, label, block, valid, trial_id = map(np.asarray, zip(*rows))
    return {"X": np.stack(X), "times": times, "fs": fs, "y": y.astype(int), "freq": freq.astype(float),
            "symbol": symbol.astype(str), "label": label.astype(str), "block": block.astype(int),
            "valid": valid.astype(bool), "trial_id": trial_id.astype(int),
            "page": int(start["page"]), "page_frequencies_hz": np.asarray(start["frequencies_hz"]),
            "page_symbols": np.asarray(start["symbols"], dtype=str),
            "channel_names": np.asarray(eeg["channel_names"])[np.asarray(eeg["eeg_channel_indices"], dtype=int)]}


def evaluate(epochs: dict, cfg: Config, windows: list[float], delay: float) -> None:
    """用与在线相同的 FBCCA 设置逐窗长评估（只用有效试次）。"""
    fs, times = epochs["fs"], epochs["times"]
    freqs = epochs["page_frequencies_hz"]
    mask = epochs["valid"]
    n_classes = len(freqs)
    print(f"\nBaseline FBCCA ({cfg.filter_bank_profile}, {cfg.n_harmonics} harmonics), "
          f"{mask.sum()} valid trials, {n_classes} classes, delay {delay:g} s")
    print(" window   accuracy   ITR (bits/min, +1.6 s gaze shift)")
    start = int(np.searchsorted(times, delay))
    for window in windows:
        if delay + window > times[-1] + 1e-9:
            print(f" {window:5.2f}s  skipped (longer than recorded flicker)")
            continue
        predictions = []
        for x in epochs["X"][mask]:
            result = classify_eeg_window(
                x[:, start:], fs, window_s=window, onset_s=0.0, target_frequencies_hz=freqs,
                n_harmonics=cfg.n_harmonics, target_fs=cfg.target_fs, notch_hz=cfg.notch_hz,
                a=cfg.weight_a, b=cfg.weight_b, regularization=cfg.cca_regularization,
                min_valid_channels=cfg.min_valid_channels, filter_bands=cfg.filter_bands,
                line_regression_hz=cfg.line_regression_hz)
            predictions.append(int(result["prediction"]))
        predictions = np.asarray(predictions)
        accuracy = float(np.mean(predictions == epochs["y"][mask]))
        itr = itr_bits_per_minute(accuracy, window + 1.6, n_classes)
        print(f" {window:5.2f}s  {accuracy * 100:7.1f}%   {itr:6.1f}")
        if window == windows[-1]:
            wrong = predictions != epochs["y"][mask]
            labels = epochs["page_symbols"]
            pairs: dict[tuple[str, str], int] = {}
            for true, pred in zip(epochs["y"][mask][wrong], predictions[wrong]):
                key = (labels[true - 1], labels[pred - 1])
                pairs[key] = pairs.get(key, 0) + 1
            worst = sorted(pairs.items(), key=lambda item: -item[1])[:8]
            if worst:
                print("   most common confusions (target -> predicted, count): "
                      + ", ".join(f"{a}->{b} ({n})" for (a, b), n in worst))
            per_class = [(labels[c - 1], float(np.mean(predictions[epochs['y'][mask] == c] == c)))
                         for c in range(1, n_classes + 1) if np.any(epochs["y"][mask] == c)]
            low = sorted(per_class, key=lambda item: item[1])[:6]
            print("   weakest keys: " + ", ".join(f"{s} {a * 100:.0f}%" for s, a in low))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Analyse a ChatBox collection session")
    parser.add_argument("session", type=Path)
    parser.add_argument("--delay", type=float, default=0.23)
    parser.add_argument("--windows", type=float, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--pre", type=float, default=0.5, help="seconds kept before onset")
    parser.add_argument("--post", type=float, default=0.5, help="seconds kept after flicker")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "default.toml")
    parser.add_argument("--no-eval", action="store_true")
    args = parser.parse_args(argv)
    session = load_session(args.session)
    epochs = extract_epochs(session, args.pre, args.post)
    print(f"Page {epochs['page']}: {len(epochs['y'])} trials "
          f"({epochs['valid'].sum()} valid), {epochs['X'].shape[1]} channels, "
          f"{epochs['X'].shape[2]} samples at {epochs['fs']:g} Hz")
    out = args.out or args.session.with_name(args.session.stem + "_epochs.npz")
    np.savez_compressed(out, **epochs)
    print(f"Epochs saved: {out}")
    if not args.no_eval:
        cfg = configure_fbcca_profile(Config.from_toml(args.config), "m3_5band")
        evaluate(epochs, cfg, sorted(args.windows), args.delay)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
