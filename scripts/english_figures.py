from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[0]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from config import CACHE_DIR, ROOT, TABLE_DIR, ensure_dirs  # noqa: E402
from experiment1 import make_x1, make_x2  # noqa: E402
from experiment2 import make_x3  # noqa: E402
from experiment3 import make_x4, make_xmra  # noqa: E402
from plot_utils import add_panel_labels, db_power, savefig, set_style, symmetric_limits, tf_plot  # noqa: E402
from signal_utils import (  # noqa: E402
    band_time_metrics,
    cwt_power,
    dwt_components,
    indicator,
    mean_band,
    prominent_peaks,
    ridge_frequency,
    single_sided_spectrum,
    stft_power,
    time_vector,
)


FS = 250.0
DURATION = 4.0
FIG_DIR = ROOT / "results_en" / "figures"
BANDS = {"mu": (8.0, 13.0), "beta": (13.0, 30.0)}
SELECTED_CHANNELS = ["C3", "C4", "Cz", "FC3", "FC4", "CP3", "CP4"]
CONDITIONS = ["left", "right"]
TASK_WINDOW = (0.5, 3.5)
BASELINE = (-0.8, -0.2)
EPOCH_START_S = -1.0
EPOCH_END_S = 4.0


def en_tf_plot(ax, times, freqs, values, *, title, cmap="magma", vmin=None, vmax=None, colorbar=False, cbar_label="Power (dB)", overlay=None, vlines=()):
    return tf_plot(
        ax,
        times,
        freqs,
        values,
        title=title,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        ylabel="Frequency (Hz)",
        xlabel="Time (s)",
        colorbar=colorbar,
        cbar_label=cbar_label,
        overlay=overlay,
        vlines=vlines,
    )


def annotate_segments_en(ax, segments):
    ymin, ymax = ax.get_ylim()
    y_text = ymax - 0.08 * (ymax - ymin)
    for start, end, label in segments:
        ax.axvspan(start, end, color="0.90", alpha=0.35, lw=0)
        ax.text((start + end) / 2, y_text, label, ha="center", va="top", fontsize=9)


def plot_exp1() -> None:
    t = time_vector(FS, DURATION)
    for name, x, stft, segments, true_freq in [
        ("x1", make_x1(t), stft_power(make_x1(t), FS, 0.5, 0.05, window="hamming", fmin=0, fmax=60), [(0, 1, "8 Hz"), (1, 2, "16 Hz"), (2, 3, "32 Hz"), (3, 4, "12 Hz")], None),
        ("x2", make_x2(t), stft_power(make_x2(t), FS, 0.5, 0.05, window="hamming", fmin=0, fmax=60), [], 5 + 10 * t),
    ]:
        freqs, amp, _ = single_sided_spectrum(x, FS)
        fig, axes = plt.subplots(3, 1, figsize=(9.8, 9.2), constrained_layout=True)
        axes[0].plot(t, x, color="#1b4f72", lw=1.0)
        axes[0].set_title(f"{name} waveform")
        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel("Amplitude")
        axes[0].set_xlim(0, DURATION)
        if segments:
            annotate_segments_en(axes[0], segments)
        axes[1].plot(freqs, amp, color="#7d3c98", lw=1.4)
        axes[1].set_title("Single-sided Fourier amplitude spectrum")
        axes[1].set_xlabel("Frequency (Hz)")
        axes[1].set_ylabel("Amplitude")
        axes[1].set_xlim(0, 60)
        axes[1].set_ylim(bottom=0)
        overlay = (t, true_freq, "True instantaneous frequency") if true_freq is not None else None
        en_tf_plot(
            axes[2],
            stft.times,
            stft.freqs,
            db_power(stft.power),
            title=f"STFT spectrogram (Hamming, {stft.actual_window_s:.3f} s window, {stft.actual_hop_s:.3f} s hop)",
            colorbar=True,
            overlay=overlay,
        )
        axes[2].set_xlim(0, DURATION)
        add_panel_labels(axes)
        savefig(fig, FIG_DIR / f"exp1_{name}_overview.png")

    x2 = make_x2(t)
    stft2 = stft_power(x2, FS, 0.5, 0.05, window="hamming", fmin=0, fmax=60)
    ridge = ridge_frequency(stft2.power, stft2.freqs, 3, 50)
    true_at_stft = 5 + 10 * stft2.times
    ridge_error = ridge - true_at_stft
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 6.4), constrained_layout=True, sharex=True)
    axes[0].plot(stft2.times, true_at_stft, color="black", lw=1.8, label="True instantaneous frequency")
    axes[0].plot(stft2.times, ridge, color="#d35400", lw=1.4, label="STFT ridge")
    axes[0].set_ylabel("Frequency (Hz)")
    axes[0].set_title("Linear chirp ridge tracking")
    axes[0].legend(fontsize=8)
    axes[1].plot(stft2.times, ridge_error, color="#a93226", lw=1.2)
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Error (Hz)")
    axes[1].set_title("Ridge error")
    savefig(fig, FIG_DIR / "exp1_chirp_ridge_error.png")


def plot_exp2() -> None:
    t = time_vector(FS, DURATION)
    x = make_x3(t)
    freqs, amp, _ = single_sided_spectrum(x, FS)
    fig, axes = plt.subplots(2, 1, figsize=(9.4, 6.6), constrained_layout=True)
    axes[0].plot(t, x, color="#1b4f72", lw=1.0)
    axes[0].set_title("x3 waveform: sustained 10 Hz, local 12 Hz, transient 35 Hz burst")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_xlim(0, DURATION)
    annotate_segments_en(axes[0], [(1.2, 2.2, "12 Hz local"), (2.6, 2.9, "35 Hz burst")])
    axes[1].plot(freqs, amp, color="#7d3c98", lw=1.4)
    axes[1].set_xlim(0, 60)
    axes[1].set_ylim(bottom=0)
    axes[1].set_title("Single-sided Fourier amplitude spectrum")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Amplitude")
    savefig(fig, FIG_DIR / "exp2_x3_wave_fft.png")

    length_results = []
    for w in [0.25, 0.5, 1.0]:
        stft = stft_power(x, FS, w, 0.05, window="hamming", fmin=0, fmax=60)
        length_results.append((f"Hamming window {w:.2f} s (actual {stft.actual_window_s:.3f} s)", stft))
    plot_stft_stack(length_results, "STFT window length effect on time-frequency resolution", FIG_DIR / "exp2_window_lengths.png")

    function_results = []
    for win, label in [("boxcar", "Rectangular window"), ("hann", "Hann window"), ("hamming", "Hamming window")]:
        function_results.append((label, stft_power(x, FS, 0.5, 0.05, window=win, fmin=0, fmax=60)))
    plot_stft_stack(function_results, "STFT results with different window functions", FIG_DIR / "exp2_window_functions.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for label, stft in length_results:
        mask_f = (stft.freqs >= 8) & (stft.freqs <= 15)
        mask_t = (stft.times >= 1.2) & (stft.times <= 2.2)
        profile = np.nanmean(stft.power[np.ix_(mask_f, mask_t)], axis=1)
        profile = profile / max(np.nanmax(profile), np.finfo(float).eps)
        axes[0].plot(stft.freqs[mask_f], profile, lw=1.5, label=label)
    axes[0].set_title("Window length: average 10-12 Hz profile")
    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("Normalized power")
    axes[0].legend(fontsize=8)
    for label, stft in function_results:
        mask_f = (stft.freqs >= 8) & (stft.freqs <= 15)
        mask_t = (stft.times >= 1.2) & (stft.times <= 2.2)
        profile = np.nanmean(stft.power[np.ix_(mask_f, mask_t)], axis=1)
        profile = profile / max(np.nanmax(profile), np.finfo(float).eps)
        axes[1].plot(stft.freqs[mask_f], profile, lw=1.5, label=label)
    axes[1].set_title("Window function: average 10-12 Hz profile")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Normalized power")
    axes[1].legend(fontsize=8)
    savefig(fig, FIG_DIR / "exp2_low_band_profiles.png")

    exploration = [
        ("Goal: localize 35 Hz burst; short Hamming window 0.25 s, hop 0.025 s", stft_power(x, FS, 0.25, 0.025, window="hamming", fmin=0, fmax=60)),
        ("Goal: separate 10 Hz and 12 Hz; long Hamming window 1.0 s, hop 0.05 s", stft_power(x, FS, 1.0, 0.05, window="hamming", fmin=0, fmax=60)),
    ]
    plot_stft_stack(exploration, "Goal-oriented STFT parameter exploration", FIG_DIR / "exp2_parameter_exploration.png")


def plot_stft_stack(results, title, out_path) -> None:
    all_db = [db_power(stft.power) for _, stft in results]
    vals = np.concatenate([v.ravel() for v in all_db])
    vmin, vmax = np.nanpercentile(vals, [5, 99])
    fig, axes = plt.subplots(len(results), 1, figsize=(9.6, 3.35 * len(results)), constrained_layout=True, sharex=True)
    if len(results) == 1:
        axes = [axes]
    for ax, (label, stft), power_db in zip(axes, results, all_db):
        en_tf_plot(
            ax,
            stft.times,
            stft.freqs,
            power_db,
            title=label,
            vmin=float(vmin),
            vmax=float(vmax),
            colorbar=True,
            vlines=[1.2, 2.2, 2.6, 2.9],
        )
        ax.set_xlim(0, DURATION)
    fig.suptitle(title, fontsize=14)
    savefig(fig, out_path)


def plot_exp3() -> None:
    t = time_vector(FS, DURATION)
    x4 = make_x4(t)
    short = stft_power(x4, FS, 0.25, 0.05, window="hamming", fmin=2, fmax=60)
    long = stft_power(x4, FS, 1.0, 0.05, window="hamming", fmin=2, fmax=60)
    cwt_freqs = np.arange(2.0, 60.5, 0.5)
    actual_cwt_freqs, cwt = cwt_power(x4, FS, cwt_freqs)
    panels = [
        ("Short-window STFT: 0.25 s Hamming", short.times, short.freqs, db_power(short.power)),
        ("Long-window STFT: 1.0 s Hamming", long.times, long.freqs, db_power(long.power)),
        ("CWT: Morlet-like wavelet", t, actual_cwt_freqs, db_power(cwt)),
    ]
    vals = np.concatenate([p[3].ravel() for p in panels])
    vmin, vmax = np.nanpercentile(vals, [5, 99])
    fig, axes = plt.subplots(3, 1, figsize=(9.8, 10), constrained_layout=True)
    for ax, (title, tt, ff, val) in zip(axes, panels):
        en_tf_plot(ax, tt, ff, val, title=title, vmin=float(vmin), vmax=float(vmax), colorbar=True, vlines=[1.0, 2.5, 3.0, 3.25])
        ax.set_xlim(0, DURATION)
    add_panel_labels(axes)
    savefig(fig, FIG_DIR / "exp3_x4_stft_cwt_compare.png")

    xmra = make_xmra(t)
    freqs, amp, _ = single_sided_spectrum(xmra, FS)
    actual_cwt_freqs, cwt = cwt_power(xmra, FS, np.arange(1.0, 60.5, 0.5))
    fig, axes = plt.subplots(3, 1, figsize=(9.8, 9), constrained_layout=True)
    axes[0].plot(t, xmra, color="#1b4f72", lw=1.0)
    axes[0].set_title("xMRA waveform")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_xlim(0, DURATION)
    annotate_segments_en(axes[0], [(2.4, 2.8, "45 Hz burst")])
    axes[1].plot(freqs, amp, color="#7d3c98", lw=1.4)
    axes[1].set_xlim(0, 60)
    axes[1].set_ylim(bottom=0)
    axes[1].set_title("xMRA single-sided Fourier amplitude spectrum")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Amplitude")
    en_tf_plot(axes[2], t, actual_cwt_freqs, db_power(cwt), title="xMRA CWT spectrogram", colorbar=True, vlines=[2.4, 2.8])
    axes[2].set_xlim(0, DURATION)
    add_panel_labels(axes)
    savefig(fig, FIG_DIR / "exp3_mra_signal_fft_cwt.png")

    components, _ = dwt_components(xmra, FS, wavelet="sym6", level=6)
    ordered = ["Original", "D1", "D2", "D3", "D4", "D5", "D6", "A6"]
    series = {"Original": xmra, **components}
    fig, axes = plt.subplots(len(ordered), 1, figsize=(10.2, 13.5), constrained_layout=True, sharex=True)
    for ax, name in zip(axes, ordered):
        ax.plot(t, series[name], lw=0.85, color="#154360" if name == "Original" else "#7d3c98")
        ax.set_ylabel(name)
        ax.set_xlim(0, DURATION)
        if name == "D2":
            ax.axvspan(2.4, 2.8, color="#f5b041", alpha=0.25, lw=0)
    axes[0].set_title("Discrete wavelet multiresolution decomposition (sym6, 6 levels)")
    axes[-1].set_xlabel("Time (s)")
    savefig(fig, FIG_DIR / "exp3_mra_components.png")

    recon_low = components["A6"] + components["D6"] + components["D5"] + components["D4"]
    recon_burst = components["D2"]
    fig, axes = plt.subplots(3, 1, figsize=(9.8, 7.5), constrained_layout=True, sharex=True)
    axes[0].plot(t, xmra, color="black", lw=1.0)
    axes[0].set_title("Original xMRA")
    axes[1].plot(t, recon_low, color="#1f618d", lw=1.0)
    axes[1].set_title("Reconstruction A6+D6+D5+D4: slow trend, 6 Hz, and about 12 Hz")
    axes[2].plot(t, recon_burst, color="#b03a2e", lw=1.0)
    axes[2].set_title("Reconstruction D2: emphasized 45 Hz burst")
    axes[2].set_xlabel("Time (s)")
    for ax in axes:
        ax.set_xlim(0, DURATION)
        ax.set_ylabel("Amplitude")
    savefig(fig, FIG_DIR / "exp3_mra_reconstruction.png")


def plot_exp4() -> None:
    cache = np.load(CACHE_DIR / "exp4_eeg_tfr_cache.npz", allow_pickle=True)
    times = cache["times"]
    freqs = cache["freqs"]
    avg_all = cache["avg_all"]
    avg_train = cache["avg_train"]
    avg_test = cache["avg_test"]
    counts = {"all": cache["counts_all"], "train": cache["counts_train"], "test": cache["counts_test"]}
    group = pd.read_csv(TABLE_DIR / "exp4_band_metrics.csv")
    lat = pd.read_csv(TABLE_DIR / "exp4_lateralization.csv")
    qc = pd.read_csv(TABLE_DIR / "exp4_trial_qc.csv")

    plot_epoch_design()
    plot_pipeline()
    plot_eeg_tfr(times, freqs, avg_all)
    plot_band_curves(times, freqs, avg_all)
    plot_metric_bars(group, lat)
    plot_neighbor_heatmap(group)
    plot_split_consistency(lat)
    plot_qc(qc, counts)


def plot_epoch_design() -> None:
    fig, ax = plt.subplots(figsize=(10, 2.6), constrained_layout=True)
    ax.axvspan(EPOCH_START_S, EPOCH_END_S, color="#edf2f7", lw=0)
    ax.axvspan(BASELINE[0], BASELINE[1], color="#85c1e9", alpha=0.65, label="Baseline")
    ax.axvspan(TASK_WINDOW[0], TASK_WINDOW[1], color="#f7dc6f", alpha=0.55, label="Task metric window")
    ax.axvline(0, color="#c0392b", lw=1.6, label="Cue onset")
    ax.set_xlim(EPOCH_START_S, EPOCH_END_S)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Time relative to cue (s)")
    ax.set_title("EEG epoch, baseline, and task metric windows")
    ax.legend(loc="upper center", ncol=3)
    savefig(fig, FIG_DIR / "exp4_epoch_design.png")


def plot_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(10, 3.2), constrained_layout=True)
    ax.axis("off")
    steps = ["Continuous EEG\nx and events t", "Select channels\nC3/C4/Cz etc.", "1-45 Hz\nzero-phase filter", "[-1,4] s\nepoching", "Downsample\nto 250 Hz", "Trial-wise CWT\nand baseline dB", "Condition average\nERD/lateralization"]
    xs = np.linspace(0.06, 0.94, len(steps))
    for i, (x, text) in enumerate(zip(xs, steps)):
        rect = plt.Rectangle((x - 0.055, 0.38), 0.11, 0.32, facecolor="#f8f9f9", edgecolor="#34495e", lw=1)
        ax.add_patch(rect)
        ax.text(x, 0.54, text, ha="center", va="center", fontsize=8.2)
        if i < len(steps) - 1:
            ax.annotate("", xy=(xs[i + 1] - 0.06, 0.54), xytext=(x + 0.06, 0.54), arrowprops=dict(arrowstyle="->", lw=1.1))
    ax.set_title("Motor imagery EEG ERD analysis workflow")
    savefig(fig, FIG_DIR / "exp4_processing_pipeline.png")


def plot_eeg_tfr(times, freqs, avg_all) -> None:
    c3, c4 = SELECTED_CHANNELS.index("C3"), SELECTED_CHANNELS.index("C4")
    vals = [avg_all[ci, ch] for ci in range(2) for ch in [c3, c4]]
    vmin, vmax = symmetric_limits(vals, q=98, minimum=2.5)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.2), constrained_layout=True, sharex=True, sharey=True)
    for r, cond in enumerate(CONDITIONS):
        for c, ch_idx in enumerate([c3, c4]):
            en_tf_plot(
                axes[r, c],
                times,
                freqs,
                avg_all[r, ch_idx],
                title=f"{cond.capitalize()} hand imagery - {SELECTED_CHANNELS[ch_idx]}",
                cmap="RdBu_r",
                vmin=vmin,
                vmax=vmax,
                colorbar=True,
                cbar_label="Baseline-corrected power (dB)",
                vlines=[0, TASK_WINDOW[0], TASK_WINDOW[1]],
            )
            axes[r, c].set_xlim(EPOCH_START_S, EPOCH_END_S)
    savefig(fig, FIG_DIR / "exp4_c3_c4_erd_tfr.png")

    diffs = [avg_all[ci, c3] - avg_all[ci, c4] for ci in range(2)]
    dvmin, dvmax = symmetric_limits(diffs, q=98, minimum=1.5)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1), constrained_layout=True, sharey=True)
    for ax, cond, diff in zip(axes, CONDITIONS, diffs):
        en_tf_plot(ax, times, freqs, diff, title=f"{cond.capitalize()} hand: C3-C4 difference", cmap="RdBu_r", vmin=dvmin, vmax=dvmax, colorbar=True, cbar_label="C3-C4 dB", vlines=[0, TASK_WINDOW[0], TASK_WINDOW[1]])
        ax.set_xlim(EPOCH_START_S, EPOCH_END_S)
    savefig(fig, FIG_DIR / "exp4_c3_minus_c4_tfr.png")


def plot_band_curves(times, freqs, avg_all) -> None:
    c3, c4 = SELECTED_CHANNELS.index("C3"), SELECTED_CHANNELS.index("C4")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4), constrained_layout=True, sharex=True)
    for r, (band_name, band) in enumerate(BANDS.items()):
        for c, cond in enumerate(CONDITIONS):
            ax = axes[r, c]
            cond_idx = CONDITIONS.index(cond)
            ax.axvspan(TASK_WINDOW[0], TASK_WINDOW[1], color="#f7dc6f", alpha=0.25, lw=0)
            for ch_idx, color in [(c3, "#1f618d"), (c4, "#b03a2e")]:
                curve = mean_band(avg_all[cond_idx, ch_idx], freqs, band)
                ax.plot(times, curve, lw=1.5, color=color, label=SELECTED_CHANNELS[ch_idx])
            ax.axhline(0, color="black", lw=0.8)
            ax.axvline(0, color="0.25", ls="--", lw=0.9)
            ax.set_title(f"{cond.capitalize()} hand - {band_name} band mean dB")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("dB")
            ax.set_xlim(EPOCH_START_S, EPOCH_END_S)
            ax.legend(fontsize=8)
    savefig(fig, FIG_DIR / "exp4_mu_beta_time_curves.png")


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
        ax.set_xticks(x, ["Left", "Right"])
        ax.set_title(f"{band} band task-window mean dB")
        ax.set_ylabel("Baseline-corrected power (dB)")
        ax.legend()
    savefig(fig, FIG_DIR / "exp4_band_mean_bars.png")

    fig, ax = plt.subplots(figsize=(8.8, 4.2), constrained_layout=True)
    sub = lat[lat["split"] == "all"].copy()
    labels = [f"{r.condition.capitalize()}-{r.band}" for r in sub.itertuples()]
    colors = ["#7fb3d5" if r.condition == "left" else "#f1948a" for r in sub.itertuples()]
    ax.bar(labels, sub["c3_minus_c4_db"], color=colors, alpha=0.9)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("C3-C4 mean dB")
    ax.set_title("C3-C4 lateralization metric")
    ax.tick_params(axis="x", rotation=20)
    savefig(fig, FIG_DIR / "exp4_lateralization_index.png")


def plot_neighbor_heatmap(group: pd.DataFrame) -> None:
    sub = group[group["split"] == "all"].copy()
    rows = [(cond, band) for cond in CONDITIONS for band in BANDS]
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
    ax.set_yticks(np.arange(len(rows)), [f"{c.capitalize()}-{b}" for c, b in rows])
    ax.set_title("Task-window mean dB over central and neighboring channels")
    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            ax.text(c, r, f"{matrix[r, c]:.2f}", ha="center", va="center", fontsize=8)
    cb = fig.colorbar(im, ax=ax, pad=0.015)
    cb.set_label("Baseline-corrected power (dB)")
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
        ax.set_xticks(x, ["Left", "Right"])
        ax.set_title(f"{band} lateralization: train/test consistency")
        ax.set_ylabel("C3-C4 mean dB")
        ax.legend()
    savefig(fig, FIG_DIR / "exp4_train_test_consistency.png")


def plot_qc(qc: pd.DataFrame, counts: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    kept = qc[qc["kept"] == True]
    axes[0].hist(kept["p2p_max"], bins=30, color="#5499c7", edgecolor="white")
    axes[0].axvline(300, color="#c0392b", ls="--", lw=1.1, label="300 uV descriptive threshold")
    axes[0].set_title("Peak-to-peak distribution of retained trials")
    axes[0].set_xlabel("Maximum peak-to-peak amplitude")
    axes[0].set_ylabel("Trial count")
    axes[0].legend(fontsize=8)
    all_counts = counts["all"]
    labels = ["Left-C3", "Left-C4", "Right-C3", "Right-C4"]
    c3, c4 = SELECTED_CHANNELS.index("C3"), SELECTED_CHANNELS.index("C4")
    vals = [all_counts[0, c3], all_counts[0, c4], all_counts[1, c3], all_counts[1, c4]]
    axes[1].bar(labels, vals, color=["#7fb3d5", "#7fb3d5", "#f1948a", "#f1948a"])
    axes[1].set_ylim(0, max(vals) * 1.2)
    axes[1].set_title("Trials entering condition averages")
    axes[1].set_ylabel("Trial count")
    axes[1].tick_params(axis="x", rotation=20)
    savefig(fig, FIG_DIR / "exp4_quality_control.png")


def main() -> None:
    ensure_dirs()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    set_style()
    plot_exp1()
    plot_exp2()
    plot_exp3()
    plot_exp4()
    print(f"[english_figures] wrote figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
