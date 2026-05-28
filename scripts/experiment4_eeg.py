from __future__ import annotations

import gc
import json
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import io, signal

from config import CACHE_DIR, DATA_OUT_DIR, FIG_DIR, TABLE_DIR, ensure_dirs, mat_path
from plot_utils import add_panel_labels, savefig, set_style, symmetric_limits, tf_plot
from signal_utils import cwt_power, mean_band


FS_TARGET = 250
EPOCH_START_S = -1.0
EPOCH_END_S = 4.0
BASELINE = (-0.8, -0.2)
TASK_WINDOW = (0.5, 3.5)
FREQS = np.arange(2.0, 46.0, 1.0)
SELECTED_CHANNELS = ["C3", "C4", "Cz", "FC3", "FC4", "CP3", "CP4"]
CONDITIONS = ["left", "right"]
BANDS = {"mu": (8.0, 13.0), "beta": (13.0, 30.0)}


def _as_list(value) -> list:
    if isinstance(value, np.ndarray):
        return value.ravel().tolist()
    if isinstance(value, list):
        return value
    return [value]


def _clean_str(value) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    return str(value).strip()


def load_split(var_name: str) -> tuple[np.ndarray, list[str], list[str], list[dict], dict]:
    print(f"[EEG] loading {var_name}")
    loaded = io.loadmat(mat_path(), simplify_cells=True, variable_names=[var_name])
    eeg = loaded[var_name]
    fs = int(round(float(eeg["fs"])))
    channels = [_clean_str(c) for c in _as_list(eeg["chan"])]
    channel_index = {name: i for i, name in enumerate(channels)}
    selected_idx = [channel_index[ch] for ch in SELECTED_CHANNELS]

    x = np.asarray(eeg["x"], dtype=np.float64)[:, selected_idx]
    raw_nan_count = int(np.isnan(x).sum())
    if raw_nan_count:
        col_mean = np.nanmean(x, axis=0)
        inds = np.where(np.isnan(x))
        x[inds] = np.take(col_mean, inds[1])

    sos = signal.butter(4, [1.0, 45.0], btype="bandpass", fs=fs, output="sos")
    x = signal.sosfiltfilt(sos, x, axis=0)

    events = np.asarray(eeg["t"], dtype=int).ravel()
    labels = [_clean_str(v).lower() for v in _as_list(eeg["y_class"])]
    gcd = math.gcd(fs, FS_TARGET)
    up = FS_TARGET // gcd
    down = fs // gcd
    pre = int(round(EPOCH_START_S * fs))
    post = int(round(EPOCH_END_S * fs))
    expected_len = int(round((EPOCH_END_S - EPOCH_START_S) * FS_TARGET))

    epochs = []
    kept_labels: list[str] = []
    split_names: list[str] = []
    qc_rows: list[dict] = []
    for i, (ev, label) in enumerate(zip(events, labels), start=1):
        ev0 = int(ev) - 1
        start = ev0 + pre
        end = ev0 + post
        in_bounds = start >= 0 and end <= x.shape[0]
        if not in_bounds:
            qc_rows.append({"split": var_name, "trial": i, "label": label, "kept": False, "reason": "epoch_out_of_bounds"})
            continue
        epoch = x[start:end, :]
        epoch_ds = signal.resample_poly(epoch, up=up, down=down, axis=0)
        if epoch_ds.shape[0] != expected_len:
            epoch_ds = epoch_ds[:expected_len, :] if epoch_ds.shape[0] > expected_len else np.pad(
                epoch_ds, ((0, expected_len - epoch_ds.shape[0]), (0, 0)), mode="edge"
            )
        finite = bool(np.isfinite(epoch_ds).all())
        p2p = np.ptp(epoch_ds, axis=0)
        qc_rows.append(
            {
                "split": var_name,
                "trial": i,
                "label": label,
                "kept": finite,
                "reason": "finite" if finite else "non_finite",
                "p2p_max": float(np.max(p2p)),
                **{f"p2p_{ch}": float(v) for ch, v in zip(SELECTED_CHANNELS, p2p)},
            }
        )
        if finite:
            epochs.append(epoch_ds.astype(np.float32))
            kept_labels.append(label)
            split_names.append(var_name.replace("EEG_MI_", ""))

    meta = {
        "split": var_name,
        "fs_original": fs,
        "fs_target": FS_TARGET,
        "raw_samples": int(x.shape[0]),
        "channels_total": len(channels),
        "selected_channels": SELECTED_CHANNELS,
        "selected_indices_1based": [idx + 1 for idx in selected_idx],
        "raw_nan_count": raw_nan_count,
        "events_total": int(events.size),
        "epochs_kept": len(epochs),
    }
    del loaded, eeg, x
    gc.collect()
    return np.stack(epochs), kept_labels, split_names, qc_rows, meta


def compute_tfr(epochs: np.ndarray, labels: list[str], splits: list[str], times: np.ndarray):
    n_cond = len(CONDITIONS)
    n_ch = len(SELECTED_CHANNELS)
    n_freq = len(FREQS)
    n_time = len(times)
    split_names = ["all", "train", "test"]
    sums = {name: np.zeros((n_cond, n_ch, n_freq, n_time), dtype=np.float64) for name in split_names}
    counts = {name: np.zeros((n_cond, n_ch), dtype=int) for name in split_names}
    trial_metrics: list[dict] = []

    baseline_mask = (times >= BASELINE[0]) & (times <= BASELINE[1])
    task_mask = (times >= TASK_WINDOW[0]) & (times <= TASK_WINDOW[1])

    for trial_idx in range(epochs.shape[0]):
        label = labels[trial_idx]
        split = splits[trial_idx]
        if label not in CONDITIONS:
            continue
        cond_idx = CONDITIONS.index(label)
        if trial_idx % 20 == 0:
            print(f"[EEG] CWT trial {trial_idx + 1}/{epochs.shape[0]}")
        for ch_idx, ch in enumerate(SELECTED_CHANNELS):
            actual_freqs, power = cwt_power(epochs[trial_idx, :, ch_idx], FS_TARGET, FREQS)
            base = np.nanmean(power[:, baseline_mask], axis=1, keepdims=True)
            db = 10.0 * np.log10(np.maximum(power, 1e-18) / np.maximum(base, 1e-18))
            for split_key in ["all", split]:
                sums[split_key][cond_idx, ch_idx] += db
                counts[split_key][cond_idx, ch_idx] += 1
            for band_name, band in BANDS.items():
                fmask = (actual_freqs >= band[0]) & (actual_freqs <= band[1])
                trial_metrics.append(
                    {
                        "split": split,
                        "condition": label,
                        "channel": ch,
                        "band": band_name,
                        "mean_db": float(np.nanmean(db[np.ix_(fmask, task_mask)])),
                    }
                )

    avgs = {}
    for split_key, arr in sums.items():
        denom = counts[split_key][:, :, None, None]
        avgs[split_key] = np.divide(arr, denom, out=np.full_like(arr, np.nan), where=denom > 0)
    return actual_freqs, avgs, counts, pd.DataFrame(trial_metrics)


def plot_epoch_design() -> None:
    fig, ax = plt.subplots(figsize=(10, 2.6), constrained_layout=True)
    ax.axvspan(EPOCH_START_S, EPOCH_END_S, color="#edf2f7", lw=0)
    ax.axvspan(BASELINE[0], BASELINE[1], color="#85c1e9", alpha=0.65, label="基线窗")
    ax.axvspan(TASK_WINDOW[0], TASK_WINDOW[1], color="#f7dc6f", alpha=0.55, label="指标任务窗")
    ax.axvline(0, color="#c0392b", lw=1.6, label="提示对齐点")
    ax.set_xlim(EPOCH_START_S, EPOCH_END_S)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("相对提示时间 (s)")
    ax.set_title("真实 EEG 试次截取、基线与任务指标时间窗")
    ax.legend(loc="upper center", ncol=3)
    savefig(fig, FIG_DIR / "exp4_epoch_design.png")


def plot_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(10, 3.2), constrained_layout=True)
    ax.axis("off")
    steps = [
        "连续 EEG\nx 与事件 t",
        "按导联名选择\nC3/C4/Cz 等",
        "1-45 Hz\n零相位滤波",
        "[-1,4] s\n试次截取",
        "降采样至\n250 Hz",
        "每试次 CWT\n与基线 dB",
        "按条件平均\nERD/偏侧化",
    ]
    xs = np.linspace(0.06, 0.94, len(steps))
    for i, (x, text) in enumerate(zip(xs, steps)):
        rect = plt.Rectangle((x - 0.055, 0.38), 0.11, 0.32, facecolor="#f8f9f9", edgecolor="#34495e", lw=1)
        ax.add_patch(rect)
        ax.text(x, 0.54, text, ha="center", va="center", fontsize=9)
        if i < len(steps) - 1:
            ax.annotate("", xy=(xs[i + 1] - 0.06, 0.54), xytext=(x + 0.06, 0.54), arrowprops=dict(arrowstyle="->", lw=1.1))
    ax.set_title("运动想象 EEG ERD 分析流程")
    savefig(fig, FIG_DIR / "exp4_processing_pipeline.png")


def plot_main_tfr(times, freqs, avgs) -> None:
    all_avg = avgs["all"]
    c3 = SELECTED_CHANNELS.index("C3")
    c4 = SELECTED_CHANNELS.index("C4")
    vals = [all_avg[ci, ch] for ci in range(2) for ch in [c3, c4]]
    vmin, vmax = symmetric_limits(vals, q=98, minimum=2.5)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.2), constrained_layout=True, sharex=True, sharey=True)
    for r, cond in enumerate(CONDITIONS):
        for c, ch_idx in enumerate([c3, c4]):
            tf_plot(
                axes[r, c],
                times,
                freqs,
                all_avg[r, ch_idx],
                title=f"{'左手' if cond == 'left' else '右手'}运动想象 - {SELECTED_CHANNELS[ch_idx]}",
                cmap="RdBu_r",
                vmin=vmin,
                vmax=vmax,
                colorbar=True,
                cbar_label="相对基线功率 (dB)",
                vlines=[0, TASK_WINDOW[0], TASK_WINDOW[1]],
            )
            axes[r, c].set_xlim(EPOCH_START_S, EPOCH_END_S)
    savefig(fig, FIG_DIR / "exp4_c3_c4_erd_tfr.png")

    diffs = [all_avg[ci, c3] - all_avg[ci, c4] for ci in range(2)]
    dvmin, dvmax = symmetric_limits(diffs, q=98, minimum=1.5)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1), constrained_layout=True, sharey=True)
    for ax, cond, diff in zip(axes, CONDITIONS, diffs):
        tf_plot(
            ax,
            times,
            freqs,
            diff,
            title=f"{'左手' if cond == 'left' else '右手'}：C3-C4 差异图",
            cmap="RdBu_r",
            vmin=dvmin,
            vmax=dvmax,
            colorbar=True,
            cbar_label="C3-C4 dB",
            vlines=[0, TASK_WINDOW[0], TASK_WINDOW[1]],
        )
        ax.set_xlim(EPOCH_START_S, EPOCH_END_S)
    savefig(fig, FIG_DIR / "exp4_c3_minus_c4_tfr.png")


def plot_band_curves(times, freqs, avgs) -> None:
    all_avg = avgs["all"]
    c3 = SELECTED_CHANNELS.index("C3")
    c4 = SELECTED_CHANNELS.index("C4")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4), constrained_layout=True, sharex=True)
    for r, (band_name, band) in enumerate(BANDS.items()):
        for c, cond in enumerate(CONDITIONS):
            ax = axes[r, c]
            cond_idx = CONDITIONS.index(cond)
            ax.axvspan(TASK_WINDOW[0], TASK_WINDOW[1], color="#f7dc6f", alpha=0.25, lw=0)
            for ch_idx, color in [(c3, "#1f618d"), (c4, "#b03a2e")]:
                curve = mean_band(all_avg[cond_idx, ch_idx], freqs, band)
                ax.plot(times, curve, lw=1.5, color=color, label=SELECTED_CHANNELS[ch_idx])
            ax.axhline(0, color="black", lw=0.8)
            ax.axvline(0, color="0.25", ls="--", lw=0.9)
            ax.set_title(f"{'左手' if cond == 'left' else '右手'} - {band_name} 频段平均 dB")
            ax.set_xlabel("时间 (s)")
            ax.set_ylabel("dB")
            ax.set_xlim(EPOCH_START_S, EPOCH_END_S)
            ax.legend(fontsize=8)
    savefig(fig, FIG_DIR / "exp4_mu_beta_time_curves.png")


def summarize_metrics(trial_metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_rows = trial_metrics.copy()
    all_rows["split"] = "all"
    trial_metrics = pd.concat([trial_metrics, all_rows], ignore_index=True)
    group = (
        trial_metrics.groupby(["split", "condition", "channel", "band"], as_index=False)
        .agg(mean_db=("mean_db", "mean"), std_db=("mean_db", "std"), n_trials=("mean_db", "size"))
        .sort_values(["split", "band", "condition", "channel"])
    )
    c3c4 = group[group["channel"].isin(["C3", "C4"])].copy()
    piv = c3c4.pivot_table(index=["split", "condition", "band"], columns="channel", values="mean_db").reset_index()
    piv["c3_minus_c4_db"] = piv["C3"] - piv["C4"]
    piv["expected_direction"] = np.where(piv["condition"] == "left", "左手预期 C4 ERD 更强，C3-C4 偏正", "右手预期 C3 ERD 更强，C3-C4 偏负")
    return group, piv


def plot_metric_bars(group: pd.DataFrame, lat: pd.DataFrame) -> None:
    main = group[(group["split"] == "all") & (group["channel"].isin(["C3", "C4"]))]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True, sharey=True)
    x = np.arange(2)
    width = 0.34
    for ax, band in zip(axes, BANDS):
        sub = main[main["band"] == band]
        for i, ch in enumerate(["C3", "C4"]):
            vals = [sub[(sub["condition"] == cond) & (sub["channel"] == ch)]["mean_db"].iloc[0] for cond in CONDITIONS]
            ax.bar(x + (i - 0.5) * width, vals, width=width, label=ch, color=["#1f618d", "#b03a2e"][i], alpha=0.85)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x, ["左手", "右手"])
        ax.set_title(f"{band} 频段任务窗平均 dB")
        ax.set_ylabel("相对基线功率 (dB)")
        ax.legend()
    savefig(fig, FIG_DIR / "exp4_band_mean_bars.png")

    fig, ax = plt.subplots(figsize=(8.8, 4.2), constrained_layout=True)
    sub = lat[lat["split"] == "all"].copy()
    labels = [f"{'左手' if r.condition == 'left' else '右手'}-{r.band}" for r in sub.itertuples()]
    colors = ["#7fb3d5" if r.condition == "left" else "#f1948a" for r in sub.itertuples()]
    ax.bar(labels, sub["c3_minus_c4_db"], color=colors, alpha=0.9)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("C3-C4 平均 dB")
    ax.set_title("左右中央区偏侧化半定量指标")
    ax.tick_params(axis="x", rotation=20)
    savefig(fig, FIG_DIR / "exp4_lateralization_index.png")


def plot_neighbor_heatmap(group: pd.DataFrame) -> None:
    sub = group[group["split"] == "all"].copy()
    rows = []
    for cond in CONDITIONS:
        for band in BANDS:
            rows.append((cond, band))
    matrix = np.full((len(rows), len(SELECTED_CHANNELS)), np.nan)
    for r, (cond, band) in enumerate(rows):
        for c, ch in enumerate(SELECTED_CHANNELS):
            value = sub[(sub["condition"] == cond) & (sub["band"] == band) & (sub["channel"] == ch)]["mean_db"]
            if not value.empty:
                matrix[r, c] = value.iloc[0]
    vmin, vmax = symmetric_limits([matrix], q=98, minimum=1.5)
    fig, ax = plt.subplots(figsize=(10, 4.2), constrained_layout=True)
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(SELECTED_CHANNELS)), SELECTED_CHANNELS)
    ax.set_yticks(np.arange(len(rows)), [f"{'左手' if c == 'left' else '右手'}-{b}" for c, b in rows])
    ax.set_title("中央区及邻近导联任务窗平均 dB")
    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            ax.text(c, r, f"{matrix[r, c]:.2f}", ha="center", va="center", fontsize=8)
    cb = fig.colorbar(im, ax=ax, pad=0.015)
    cb.set_label("相对基线功率 (dB)")
    savefig(fig, FIG_DIR / "exp4_neighbor_channel_heatmap.png")


def plot_split_consistency(lat: pd.DataFrame) -> None:
    sub = lat[lat["split"].isin(["train", "test"])].copy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True, sharey=True)
    for ax, band in zip(axes, BANDS):
        ss = sub[sub["band"] == band]
        x = np.arange(2)
        width = 0.34
        for i, split in enumerate(["train", "test"]):
            vals = [ss[(ss["condition"] == cond) & (ss["split"] == split)]["c3_minus_c4_db"].iloc[0] for cond in CONDITIONS]
            ax.bar(x + (i - 0.5) * width, vals, width=width, label=split, alpha=0.85)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x, ["左手", "右手"])
        ax.set_title(f"{band} 偏侧化 train/test 一致性")
        ax.set_ylabel("C3-C4 平均 dB")
        ax.legend()
    savefig(fig, FIG_DIR / "exp4_train_test_consistency.png")


def plot_qc(qc: pd.DataFrame, counts: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    kept = qc[qc["kept"] == True]
    axes[0].hist(kept["p2p_max"], bins=30, color="#5499c7", edgecolor="white")
    axes[0].axvline(300, color="#c0392b", ls="--", lw=1.1, label="300 µV 描述阈值")
    axes[0].set_title("保留试次峰-峰值分布")
    axes[0].set_xlabel("最大峰-峰值")
    axes[0].set_ylabel("试次数")
    axes[0].legend(fontsize=8)
    all_counts = counts["all"]
    labels = ["左手-C3", "左手-C4", "右手-C3", "右手-C4"]
    c3 = SELECTED_CHANNELS.index("C3")
    c4 = SELECTED_CHANNELS.index("C4")
    vals = [all_counts[0, c3], all_counts[0, c4], all_counts[1, c3], all_counts[1, c4]]
    axes[1].bar(labels, vals, color=["#7fb3d5", "#7fb3d5", "#f1948a", "#f1948a"])
    axes[1].set_ylim(0, max(vals) * 1.2)
    axes[1].set_title("进入平均的试次数")
    axes[1].set_ylabel("试次数")
    axes[1].tick_params(axis="x", rotation=20)
    savefig(fig, FIG_DIR / "exp4_quality_control.png")


def main() -> None:
    ensure_dirs()
    set_style()
    plot_epoch_design()
    plot_pipeline()

    all_epochs = []
    all_labels: list[str] = []
    all_splits: list[str] = []
    qc_rows: list[dict] = []
    meta_rows: list[dict] = []
    for var in ["EEG_MI_train", "EEG_MI_test"]:
        epochs, labels, splits, qc, meta = load_split(var)
        all_epochs.append(epochs)
        all_labels.extend(labels)
        all_splits.extend(splits)
        qc_rows.extend(qc)
        meta_rows.append(meta)
    epochs = np.concatenate(all_epochs, axis=0)
    times = np.arange(epochs.shape[1]) / FS_TARGET + EPOCH_START_S

    freqs, avgs, counts, trial_metrics = compute_tfr(epochs, all_labels, all_splits, times)
    group, lat = summarize_metrics(trial_metrics)

    np.savez_compressed(
        CACHE_DIR / "exp4_eeg_tfr_cache.npz",
        times=times,
        freqs=freqs,
        channels=np.array(SELECTED_CHANNELS),
        conditions=np.array(CONDITIONS),
        avg_all=avgs["all"],
        avg_train=avgs["train"],
        avg_test=avgs["test"],
        counts_all=counts["all"],
        counts_train=counts["train"],
        counts_test=counts["test"],
    )

    qc_df = pd.DataFrame(qc_rows)
    qc_df.to_csv(TABLE_DIR / "exp4_trial_qc.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(meta_rows).to_csv(TABLE_DIR / "exp4_metadata.csv", index=False, encoding="utf-8-sig")
    group.to_csv(TABLE_DIR / "exp4_band_metrics.csv", index=False, encoding="utf-8-sig")
    lat.to_csv(TABLE_DIR / "exp4_lateralization.csv", index=False, encoding="utf-8-sig")
    count_rows = []
    for split, arr in counts.items():
        for ci, cond in enumerate(CONDITIONS):
            for ch_idx, ch in enumerate(SELECTED_CHANNELS):
                count_rows.append({"split": split, "condition": cond, "channel": ch, "n": int(arr[ci, ch_idx])})
    pd.DataFrame(count_rows).to_csv(TABLE_DIR / "exp4_counts.csv", index=False, encoding="utf-8-sig")

    plot_main_tfr(times, freqs, avgs)
    plot_band_curves(times, freqs, avgs)
    plot_metric_bars(group, lat)
    plot_neighbor_heatmap(group)
    plot_split_consistency(lat)
    plot_qc(qc_df, counts)

    summary = {
        "mat_file": str(mat_path()),
        "epochs_total": int(epochs.shape[0]),
        "epoch_start_s": EPOCH_START_S,
        "epoch_end_s": EPOCH_END_S,
        "baseline_s": BASELINE,
        "task_window_s": TASK_WINDOW,
        "freqs_hz": [float(FREQS[0]), float(FREQS[-1])],
        "cwt_wavelet": "cmor1.5-1.0",
        "selected_channels": SELECTED_CHANNELS,
        "condition_counts": pd.Series(all_labels).value_counts().to_dict(),
        "quality_high_p2p_over_300": int((qc_df.get("p2p_max", pd.Series(dtype=float)) > 300).sum()),
    }
    (DATA_OUT_DIR / "exp4_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
