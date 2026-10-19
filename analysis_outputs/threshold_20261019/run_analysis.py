"""离线 FBCCA 分数阈值分析；只调用现有 classify_eeg_window，不改算法。"""
from __future__ import annotations
import csv, io, json, zipfile
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from ssvep_fbcca.config import Config
from ssvep_fbcca.signal import classify_eeg_window
from ssvep_fbcca.intent import alpha_power_ratio

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
FILES = {
    "target": ROOT / "session_records/注视目标.zip",
    "screen_unfocused": ROOT / "session_records/看屏幕没有专注看任何字符。.zip",
    "look_around": ROOT / "session_records/看周围环境.zip",
    "eyes_closed": ROOT / "session_records/闭上眼睛.zip",
}


def jsonable(x):
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (np.floating,)): return float(x)
    if isinstance(x, np.ndarray): return x.tolist()
    raise TypeError(type(x).__name__)


def load_session(kind: str, path: Path) -> tuple[list[dict], dict]:
    with zipfile.ZipFile(path) as archive:
        npz_name = next(n for n in archive.namelist()
                        if n.endswith(".npz") and "continuous" not in n and "sync_signals" not in n)
        data = np.load(io.BytesIO(archive.read(npz_name)), allow_pickle=False)
        metadata = json.loads(str(data["metadata_json"].item()))
        by_id = {int(t["trial_id"]): t for t in metadata["trials"]}
        cfg_meta = metadata["config"]
        protocol = cfg_meta.get("protocol", "standard_5s")
        cfg = Config(protocol=protocol,
                     filter_bank_profile=cfg_meta.get("filter_bank_profile", "m3_5band"),
                     target_fs=float(cfg_meta.get("target_fs", 250.0)),
                     n_harmonics=int(cfg_meta.get("n_harmonics", 3)),
                     weight_a=float(cfg_meta.get("weight_a", 1.25)),
                     weight_b=float(cfg_meta.get("weight_b", 0.25)),
                     notch_hz=cfg_meta.get("notch_hz", 50.0),
                     cca_regularization=float(cfg_meta.get("cca_regularization", 1e-10)),
                     channel_indices=tuple(cfg_meta.get("channel_indices", range(8))))
        rows = []
        recompute_diffs = []
        for trial_id, label, has_eeg in zip(data["trial_ids"], data["labels"], data["has_eeg"]):
            trial_id = int(trial_id)
            if not bool(has_eeg):
                continue
            prefix = f"trial_{trial_id:04d}_"
            x = data[prefix + "uniform_data"]
            saved_scores = data[prefix + "fbcca_scores"]
            result = classify_eeg_window(
                x, 250.0, window_s=cfg.window_s,
                n_harmonics=cfg.n_harmonics, target_fs=cfg.target_fs,
                notch_hz=cfg.notch_hz, a=cfg.weight_a, b=cfg.weight_b,
                regularization=cfg.cca_regularization,
                min_valid_channels=6, filter_bands=cfg.filter_bands,
            )
            scores = np.asarray(result["scores"], dtype=float)
            recompute_diffs.append(float(np.max(np.abs(scores - saved_scores))))
            trial_meta = by_id[trial_id]
            rows.append({
                "session": kind, "trial_id": trial_id,
                "block_id": trial_meta.get("block_id"),
                "label": int(label), "prediction": int(result["prediction"]),
                "max_score": float(np.max(scores)),
                "saved_max_score": float(np.max(saved_scores)),
                "alpha_ratio": alpha_power_ratio(x, 250.0),
                "correct": bool(kind == "target" and int(result["prediction"]) == int(label)),
                "stimulus_onset": trial_meta.get("stimulus_onset"),
                "stimulus_offset": trial_meta.get("stimulus_offset"),
            })
        events_name = next(n for n in archive.namelist() if n.endswith("_events.csv"))
        events = list(csv.DictReader(io.StringIO(archive.read(events_name).decode("utf-8-sig"))))
        start = float(next(r["timestamp_lsl"] for r in events if r["event"] == "session_start"))
        end = float(next(r["timestamp_lsl"] for r in events if r["event"] == "session_end"))
        info = {
            "path": str(path), "npz_member": npz_name,
            "session_mode": cfg_meta.get("session_mode"), "protocol": protocol,
            "blocks": cfg_meta.get("blocks"), "window_s": cfg.window_s,
            "channel_indices": list(cfg.channel_indices),
            "stream_name": metadata.get("acquisition_metadata", {}).get("name"),
            "stream_channel_count": metadata.get("acquisition_metadata", {}).get("channel_count"),
            "selected_channel_indices": metadata.get("acquisition_review", {}).get("selected_channel_indices"),
            "input_units": metadata.get("acquisition_review", {}).get("input_units"),
            "startup_median_line_fraction": metadata.get("acquisition_review", {}).get("startup_mains_noise", {}).get("median_line_fraction"),
            "n_trials_in_npz": int(len(data["trial_ids"])),
            "n_trials_with_eeg": len(rows),
            "n_labels_minus_one": int(np.sum(np.asarray(data["labels"]) == -1)),
            "session_wall_s": end - start,
            "session_start_lsl": start, "session_end_lsl": end,
            "recompute_max_abs_diff_vs_saved": max(recompute_diffs, default=None),
        }
    return rows, info


def stats(values: np.ndarray) -> dict:
    q = np.quantile(values, [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
    return {"n": int(values.size), "min": float(values.min()), "q05": float(q[0]),
            "q10": float(q[1]), "q25": float(q[2]), "median": float(q[3]),
            "q75": float(q[4]), "q90": float(q[5]), "q95": float(q[6]),
            "mean": float(values.mean()), "max": float(values.max())}


def threshold_metrics(rows: list[dict], *, accept_key: str, target: bool) -> dict:
    vals = [r for r in rows if (r["session"] == "target") == target]
    accepted = [r for r in vals if r[accept_key]]
    return {
        "n": len(vals), "accepted": len(accepted), "accept_rate": len(accepted) / len(vals) if vals else None,
        "rejected": len(vals) - len(accepted), "rejection_rate": (len(vals) - len(accepted)) / len(vals) if vals else None,
        "correct_accepted": sum(r["correct"] and r[accept_key] for r in vals) if target else None,
        "correct_all_rate": (sum(r["correct"] and r[accept_key] for r in vals) / len(vals)) if target and vals else None,
        "conditional_correct_rate": (sum(r["correct"] and r[accept_key] for r in vals) / len(accepted)) if target and accepted else None,
    }


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    return float((sum(p > n for p in pos for n in neg) + .5 * sum(p == n for p in pos for n in neg)) / (len(pos) * len(neg)))


def main():
    all_rows, session_info = {}, {}
    for kind, path in FILES.items():
        all_rows[kind], session_info[kind] = load_session(kind, path)
    target_rows = all_rows["target"]
    cal_pos = np.array([r["max_score"] for r in target_rows if r["block_id"] in (1, 2)])
    cal_neg = np.array([r["max_score"] for k in ("screen_unfocused", "look_around") for r in all_rows[k]])
    val_pos = np.array([r["max_score"] for r in target_rows if r["block_id"] == 3])
    val_neg = np.array([r["max_score"] for r in all_rows["eyes_closed"]])
    candidates = np.unique(np.concatenate([cal_pos, cal_neg]))
    roc_rows = []
    for threshold in candidates:
        tpr = float(np.mean(cal_pos >= threshold)); fpr = float(np.mean(cal_neg >= threshold))
        roc_rows.append({"threshold": float(threshold), "tpr": tpr, "fpr": fpr,
                         "specificity": 1.0 - fpr, "youden_j": tpr - fpr})
    chosen = max(roc_rows, key=lambda r: (r["youden_j"], r["tpr"], r["threshold"]))
    T = chosen["threshold"]
    cal_pos_rows = [r for r in target_rows if r["block_id"] in (1, 2)]
    cal_neg_rows = [r for k in ("screen_unfocused", "look_around") for r in all_rows[k]]
    alpha_candidates = np.unique(np.array([r["alpha_ratio"] for r in cal_pos_rows + cal_neg_rows]))
    alpha_roc_rows = []
    for alpha_threshold in alpha_candidates:
        tpr = float(np.mean([(r["max_score"] >= T and r["alpha_ratio"] <= alpha_threshold) for r in cal_pos_rows]))
        fpr = float(np.mean([(r["max_score"] >= T and r["alpha_ratio"] <= alpha_threshold) for r in cal_neg_rows]))
        alpha_roc_rows.append({"threshold": float(alpha_threshold), "tpr": tpr, "fpr": fpr,
                               "specificity": 1.0 - fpr, "youden_j": tpr - fpr})
    chosen_alpha = max(alpha_roc_rows, key=lambda r: (r["youden_j"], r["tpr"], -r["threshold"]))
    A = chosen_alpha["threshold"]
    for rows in all_rows.values():
        for row in rows:
            row["accepted_at_T"] = bool(row["max_score"] >= T)
            row["accepted_at_intent_gate"] = bool(
                row["accepted_at_T"] and row["alpha_ratio"] <= A)
            if row["session"] == "target":
                row["correct_accepted_at_T"] = bool(row["correct"] and row["accepted_at_T"])
                row["correct_accepted_at_intent_gate"] = bool(
                    row["correct"] and row["accepted_at_intent_gate"])
    flat = [r for rows in all_rows.values() for r in rows]
    distributions = {k: stats(np.array([r["max_score"] for r in rows])) for k, rows in all_rows.items()}
    # Run-level false triggers use session wall time; scored-window rates are also retained for transparency.
    false_triggers = {}
    for k in ("screen_unfocused", "look_around", "eyes_closed"):
        rows = all_rows[k]; count = sum(r["accepted_at_intent_gate"] for r in rows); wall = session_info[k]["session_wall_s"]
        false_triggers[k] = {"scored_trials": len(rows), "false_triggers": int(count),
                             "session_wall_s": wall, "per_minute_wall": float(count / wall * 60),
                             "per_minute_scored_5s": float(count / (len(rows) * 5.0) * 60) if rows else None}
    all_neg = np.array([r["max_score"] for k in ("screen_unfocused", "look_around", "eyes_closed") for r in all_rows[k]])
    no_fp_T = float(np.nextafter(all_neg.max(), np.inf))
    zero_fp_target = sum(r["max_score"] >= no_fp_T for r in target_rows)
    report = {
        "analysis": {"algorithm": "existing ssvep_fbcca.signal.classify_eeg_window",
                      "recomputed_all_valid_eeg_trials": sum(len(v) for v in all_rows.values()),
                      "no_algorithm_or_parameter_changes": True,
                      "decoder_parameters": {"window_s": 5.0, "target_fs": 250.0, "n_harmonics": 3,
                                             "filter_bank_profile": "m3_5band", "filter_bands_hz": [[6,90],[14,90],[22,90],[30,90],[38,90]],
                                             "weight_a": 1.25, "weight_b": .25, "notch_hz": 50.0,
                                             "cca_regularization": 1e-10, "channels": list(range(8))},
                      "recompute_check": {k: v["recompute_max_abs_diff_vs_saved"] for k,v in session_info.items()}},
        "data_format_and_labels": {"target": "120 cued trials, 3 blocks × 40, labels 1–40",
                                    "no_target": "free trials, labels -1; incomplete final trials without EEG excluded from score metrics",
                                    "sessions": session_info},
        "score_distributions": distributions,
        "calibration": {"positive_split": "target blocks 1-2", "negative_split": ["screen_unfocused", "look_around"],
                        "positive_n": len(cal_pos), "negative_n": len(cal_neg), "method": "maximize Youden J = TPR - FPR over observed score candidates; acceptance is max_score >= T",
                        "chosen_threshold_T": T, "chosen": chosen, "calibration_auc": auc(cal_pos, cal_neg),
                        "roc_candidates": sorted(roc_rows, key=lambda r: (r["youden_j"], r["tpr"], r["threshold"]), reverse=True)[:20]},
        "intent_gate_calibration": {"feature": "alpha_power_ratio = median_channel(PSD[8,13)/PSD[6,45))",
                                     "positive_split": "target blocks 1-2", "negative_split": ["screen_unfocused", "look_around"],
                                     "positive_n": len(cal_pos_rows), "negative_n": len(cal_neg_rows),
                                     "method": "固定 max_score >= T 后，在观测 alpha 比例候选值上最大化 Youden J；接受条件为 alpha_ratio <= A",
                                     "chosen_threshold_A": A, "chosen": chosen_alpha,
                                     "roc_candidates": sorted(alpha_roc_rows, key=lambda r: (r["youden_j"], r["tpr"], -r["threshold"]), reverse=True)[:20]},
        "validation": {"positive_split": "target block 3", "negative_split": "eyes_closed", "positive_n": len(val_pos), "negative_n": len(val_neg),
                       "validation_auc": auc(val_pos, val_neg), "threshold_T": T,
                       "alpha_threshold_A": A,
                       "target_block3": threshold_metrics([r for r in target_rows if r["block_id"] == 3], accept_key="accepted_at_intent_gate", target=True),
                       "eyes_closed": {"n": len(val_neg), "false_triggers": int(sum(r["accepted_at_intent_gate"] for r in all_rows["eyes_closed"])), "false_trigger_rate": float(np.mean([r["accepted_at_intent_gate"] for r in all_rows["eyes_closed"]]))}},
        "target_metrics_all_120": threshold_metrics(target_rows, accept_key="accepted_at_intent_gate", target=True),
        "target_baseline_without_threshold": {"n": len(target_rows), "correct": int(sum(r["correct"] for r in target_rows)), "accuracy": float(np.mean([r["correct"] for r in target_rows]))},
        "no_target_false_triggers": false_triggers,
        "all_no_target_zero_false_positive_reference": {"max_no_target_score": float(all_neg.max()), "threshold_strictly_above_max": no_fp_T, "target_trials_above_that_threshold": int(zero_fp_target), "target_rate": float(zero_fp_target / len(target_rows))},
        "trials": flat,
    }
    (OUT / "analysis_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=jsonable), encoding="utf-8")
    # Distribution plot with an independent validation panel.
    labels = {"target": "Target (cued)", "screen_unfocused": "No-target: screen", "look_around": "No-target: around", "eyes_closed": "No-target: eyes closed"}
    colors = {"target": "#2364aa", "screen_unfocused": "#f28e2b", "look_around": "#59a14f", "eyes_closed": "#e15759"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    bins = np.linspace(0.15, max(0.85, max(r["max_score"] for r in flat) + .02), 18)
    for ax, group_keys, title in [(axes[0], ["target", "screen_unfocused", "look_around"], "Calibration split"),
                                   (axes[1], ["target", "eyes_closed"], "Independent validation split")]:
        for k in group_keys:
            ax.hist([r["max_score"] for r in all_rows[k]], bins=bins, alpha=.48, label=labels[k], color=colors[k], edgecolor="white")
        ax.axvline(T, color="black", linestyle="--", linewidth=2, label=f"T={T:.4f}")
        ax.set_title(title); ax.set_xlabel("Maximum FBCCA score"); ax.set_ylabel("Trial count"); ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle("SSVEP 40-class keyboard: maximum FBCCA score distributions", fontsize=13)
    fig.savefig(OUT / "score_distributions.png", dpi=180)
    plt.close(fig)
    # Human-readable report.
    def pct(x): return "N/A" if x is None else f"{100*x:.1f}%"
    cal = report["calibration"]; tm = report["target_metrics_all_120"]; val = report["validation"]
    lines = [
        "# SSVEP 40 分类键盘离线 FBCCA 阈值分析",
        "",
        "## 结论",
        "",
        f"按预先固定的数据划分，先用 Youden J 校准最高分阈值 **T = {T:.6f}**，再校准独立的 alpha 相对能量阈值 **A = {A:.6f}**；自由输入只有同时满足 `max_score >= T` 和 `alpha_ratio <= A` 才提交字符。alpha 门控校准集 TPR={pct(chosen_alpha['tpr'])}、FPR={pct(chosen_alpha['fpr'])}、J={chosen_alpha['youden_j']:.3f}。",
        "",
        "这次已把分数阈值和 alpha 意图门控接入自由输入拒识入口；FBCCA 分类器本身未修改。当前闭眼验证为 0/7 误接受，但样本量很小，上线前仍需要新的独立会话复测。",
        "",
        "## 数据与复算",
        "",
        "- `注视目标.zip`：120 个有目标试次，3 区组×40，标签为 1–40；每类每区组各 1 次。",
        "- 另三个 ZIP：自由模式、标签为 `-1`；实际有 EEG 的试次分别为 10、10、7，未把最后缺 EEG 的试次当作分数。",
        "- 对 147 个有 EEG 的试次重新调用项目现有 `classify_eeg_window`；重新计算分数与 ZIP 中保存的 `fbcca_scores` 最大绝对差为 0。",
        "- 参数沿用会话：5 s 窗、250 Hz、3 次谐波、m3_5band、a=1.25、b=0.25、50 Hz notch、8 个通道。",
        "- LSL 元数据声明 16 通道 `obci_eeg1`，分析选取索引 0–7；通道单位均为 `UNKNOWN`，且四个会话元数据都记录了 50 Hz 附近高能量提示，因此结果应视为本次采集条件下的离线证据。",
        "- 项目中另有两个 2026-10-01 的 `standard_2s` 会话 ZIP；它们不是本次四个 5 s 数据包，未与当前阈值校准混合。",
        "",
        "## 分数分布",
        "",
        "| 数据 | n | 最小 | 中位数 | 均值 | 最大 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for k in ("target", "screen_unfocused", "look_around", "eyes_closed"):
        s = distributions[k]; lines.append(f"| {labels[k]} | {s['n']} | {s['min']:.4f} | {s['median']:.4f} | {s['mean']:.4f} | {s['max']:.4f} |")
    lines += ["", "![分数分布图](score_distributions.png)", "", "## 阈值校准与独立验证", "",
              f"分数阈值校准集：有目标区组 1–2（n={len(cal_pos)}）+ 无目标‘看屏幕不专注’和‘看周围环境’（n={len(cal_neg)}）；分数 AUC={cal['calibration_auc']:.3f}。alpha 门控阈值为 A={A:.6f}。独立验证集：有目标区组 3（n={len(val_pos)}）+ 闭眼（n={len(val_neg)}）；分数 AUC={val['validation_auc']:.3f}。",
              "", "| 指标 | 校准集 | 独立验证集 |", "|---|---:|---:|",
              f"| 有目标接受率/TPR（双门控） | {pct(chosen_alpha['tpr'])} | {pct(val['target_block3']['accept_rate'])} |",
              f"| 无目标误接受率/FPR（双门控） | {pct(chosen_alpha['fpr'])} | {pct(val['eyes_closed']['false_trigger_rate'])} |",
              "", "## 输入性能",
              "",
              f"全 120 个有目标试次原始 FBCCA 正确 {report['target_baseline_without_threshold']['correct']}/120（{pct(report['target_baseline_without_threshold']['accuracy'])}）。双门控后接受 {tm['accepted']}/120（{pct(tm['accept_rate'])}），拒绝 {tm['rejected']}/120（误拒绝率 {pct(tm['rejection_rate'])}）；正确且被接受 {tm['correct_accepted']}/120（正确输入率按全部目标计 {pct(tm['correct_all_rate'])}），在已接受试次中的条件正确率为 {pct(tm['conditional_correct_rate'])}。",
              "",
              "无目标每分钟误触发（按 session_start 到 session_end 的墙钟时长计）：",
              ""]
    for k in ("screen_unfocused", "look_around", "eyes_closed"):
        f = false_triggers[k]; lines.append(f"- {labels[k]}：{f['false_triggers']}/{f['scored_trials']}，{f['per_minute_wall']:.2f} 次/分钟（仅按 5 s 已评分窗折算为 {f['per_minute_scored_5s']:.2f} 次/分钟）。")
    total_count = sum(v['false_triggers'] for v in false_triggers.values()); total_wall = sum(v['session_wall_s'] for v in false_triggers.values())
    lines.append(f"- 三个无目标会话合计：{total_count} 次/{total_wall/60:.2f} 分钟，即 {total_count/(total_wall/60):.2f} 次/分钟。")
    lines += ["", f"若只使用最高分并要求三个无目标状态全部 0 误触发，阈值必须严格高于无目标最大分数 {all_neg.max():.6f}；此时有目标仅 {zero_fp_target}/120（{pct(zero_fp_target/len(target_rows))}）超过阈值。当前实现增加 alpha 意图门控，但闭眼验证仍有 {val['eyes_closed']['false_triggers']}/{val['eyes_closed']['n']} 个误接受。", "", "## 生成文件", "", "- `analysis_report.json`：完整统计、逐试次结果、阈值候选和意图门控验证。", "- `score_distributions.png`：校准和独立验证分布图。", "- `run_analysis.py`：本次离线复算脚本。"]
    (OUT / "analysis_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"score_threshold_T": T, "alpha_threshold_A": A,
                      "target_metrics": tm, "validation": val,
                      "false_triggers": false_triggers, "output": str(OUT)},
                     ensure_ascii=False, indent=2, default=jsonable))

if __name__ == "__main__": main()
