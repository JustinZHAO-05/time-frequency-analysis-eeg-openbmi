from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

from config import DATA_OUT_DIR, FIG_DIR, TABLE_DIR, ensure_dirs
from plot_utils import add_panel_labels, annotate_segments, db_power, savefig, set_style, tf_plot
from signal_utils import band_time_metrics, indicator, single_sided_spectrum, stft_power, time_vector


FS = 250.0
DURATION = 4.0


def make_x3(t: np.ndarray) -> np.ndarray:
    return (
        np.sin(2 * np.pi * 10 * t)
        + 0.8 * indicator(t, 1.2, 2.2) * np.sin(2 * np.pi * 12 * t)
        + 1.2 * indicator(t, 2.6, 2.9) * np.sin(2 * np.pi * 35 * t)
    )


def resolvability_metric(stft) -> dict[str, float | str]:
    mask_f = (stft.freqs >= 8) & (stft.freqs <= 15)
    mask_t = (stft.times >= 1.2) & (stft.times <= 2.2)
    ff = stft.freqs[mask_f]
    profile = np.nanmean(stft.power[np.ix_(mask_f, mask_t)], axis=1)
    if profile.size < 3:
        return {"peak_10_hz": np.nan, "peak_12_hz": np.nan, "valley_ratio": np.nan, "separable": "否"}
    idx10 = int(np.nanargmax(profile[(ff >= 9) & (ff <= 10.8)]))
    freq10_candidates = np.where((ff >= 9) & (ff <= 10.8))[0]
    idx10 = int(freq10_candidates[idx10])
    idx12 = int(np.nanargmax(profile[(ff >= 11.2) & (ff <= 13.2)]))
    freq12_candidates = np.where((ff >= 11.2) & (ff <= 13.2))[0]
    idx12 = int(freq12_candidates[idx12])
    lo, hi = sorted([idx10, idx12])
    valley = float(np.nanmin(profile[lo : hi + 1]))
    weaker_peak = float(min(profile[idx10], profile[idx12]))
    valley_ratio = 1.0 - valley / max(weaker_peak, np.finfo(float).eps)
    return {
        "peak_10_hz": float(ff[idx10]),
        "peak_12_hz": float(ff[idx12]),
        "valley_ratio": float(valley_ratio),
        "separable": "是" if valley_ratio >= 0.12 else "弱/否",
    }


def plot_wave_fft(t: np.ndarray, x: np.ndarray) -> None:
    freqs, amp, _ = single_sided_spectrum(x, FS)
    fig, axes = plt.subplots(2, 1, figsize=(9.4, 6.6), constrained_layout=True)
    axes[0].plot(t, x, color="#1b4f72", lw=1.0)
    axes[0].set_title("x3 时域波形：持续 10 Hz、局部 12 Hz 与短时 35 Hz 突发")
    axes[0].set_xlabel("时间 (s)")
    axes[0].set_ylabel("幅值")
    axes[0].set_xlim(0, DURATION)
    annotate_segments(axes[0], [(1.2, 2.2, "12 Hz 局部"), (2.6, 2.9, "35 Hz 突发")])
    axes[1].plot(freqs, amp, color="#7d3c98", lw=1.4)
    axes[1].set_xlim(0, 60)
    axes[1].set_ylim(bottom=0)
    axes[1].set_title("整段傅里叶幅度谱")
    axes[1].set_xlabel("频率 (Hz)")
    axes[1].set_ylabel("幅度")
    add_panel_labels(axes)
    savefig(fig, FIG_DIR / "exp2_x3_wave_fft.png")


def plot_stft_panel(results: list[tuple[str, object]], filename: str, title: str) -> None:
    values = [db_power(r.power) for _, r in results]
    vmin = float(np.nanpercentile(np.concatenate([v.ravel() for v in values]), 5))
    vmax = float(np.nanpercentile(np.concatenate([v.ravel() for v in values]), 99))
    fig, axes = plt.subplots(len(results), 1, figsize=(9.6, 3.3 * len(results)), constrained_layout=True, sharex=True)
    if len(results) == 1:
        axes = [axes]
    for ax, (label, stft), val in zip(axes, results, values):
        tf_plot(
            ax,
            stft.times,
            stft.freqs,
            val,
            title=label,
            vmin=vmin,
            vmax=vmax,
            colorbar=True,
            cbar_label="功率 (dB)",
            vlines=[1.2, 2.2, 2.6, 2.9],
        )
        ax.set_xlim(0, DURATION)
    fig.suptitle(title, fontsize=14)
    savefig(fig, FIG_DIR / filename)


def plot_low_band_profiles(length_results, func_results) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for label, stft in length_results:
        mask_f = (stft.freqs >= 8) & (stft.freqs <= 15)
        mask_t = (stft.times >= 1.2) & (stft.times <= 2.2)
        profile = np.nanmean(stft.power[np.ix_(mask_f, mask_t)], axis=1)
        profile = profile / np.nanmax(profile)
        axes[0].plot(stft.freqs[mask_f], profile, lw=1.5, label=label)
    axes[0].set_title("窗长比较：10-12 Hz 平均谱形")
    axes[0].set_xlabel("频率 (Hz)")
    axes[0].set_ylabel("归一化功率")
    axes[0].legend(fontsize=8)
    for label, stft in func_results:
        mask_f = (stft.freqs >= 8) & (stft.freqs <= 15)
        mask_t = (stft.times >= 1.2) & (stft.times <= 2.2)
        profile = np.nanmean(stft.power[np.ix_(mask_f, mask_t)], axis=1)
        profile = profile / np.nanmax(profile)
        axes[1].plot(stft.freqs[mask_f], profile, lw=1.5, label=label)
    axes[1].set_title("窗函数比较：10-12 Hz 平均谱形")
    axes[1].set_xlabel("频率 (Hz)")
    axes[1].set_ylabel("归一化功率")
    axes[1].legend(fontsize=8)
    savefig(fig, FIG_DIR / "exp2_low_band_profiles.png")


def main() -> None:
    ensure_dirs()
    set_style()
    t = time_vector(FS, DURATION)
    x = make_x3(t)
    plot_wave_fft(t, x)

    length_results = []
    rows = []
    for w in [0.25, 0.5, 1.0]:
        stft = stft_power(x, FS, w, 0.05, window="hamming", fmin=0, fmax=60)
        label = f"Hamming 窗长 {w:.2f} s（实际 {stft.actual_window_s:.3f} s）"
        length_results.append((label, stft))
        burst = band_time_metrics(stft.power, stft.freqs, stft.times, (32, 38))
        resolv = resolvability_metric(stft)
        rows.append(
            {
                "comparison": "window_length",
                "setting": f"{w:.2f}s",
                "actual_window_s": stft.actual_window_s,
                "actual_hop_s": stft.actual_hop_s,
                **burst,
                **resolv,
            }
        )
    plot_stft_panel(length_results, "exp2_window_lengths.png", "STFT 窗长对时间-频率分辨率的影响")

    func_results = []
    for win in ["boxcar", "hann", "hamming"]:
        stft = stft_power(x, FS, 0.5, 0.05, window=win, fmin=0, fmax=60)
        label = {"boxcar": "矩形窗", "hann": "Hann 窗", "hamming": "Hamming 窗"}[win]
        func_results.append((label, stft))
        burst = band_time_metrics(stft.power, stft.freqs, stft.times, (32, 38))
        resolv = resolvability_metric(stft)
        side_mask = ((stft.freqs >= 20) & (stft.freqs <= 30)) | ((stft.freqs >= 40) & (stft.freqs <= 50))
        burst_t = (stft.times >= 2.6) & (stft.times <= 2.9)
        leakage = float(np.nanmean(stft.power[np.ix_(side_mask, burst_t)]))
        rows.append(
            {
                "comparison": "window_function",
                "setting": win,
                "actual_window_s": stft.actual_window_s,
                "actual_hop_s": stft.actual_hop_s,
                "leakage_proxy": leakage,
                **burst,
                **resolv,
            }
        )
    plot_stft_panel(func_results, "exp2_window_functions.png", "不同窗函数在相同窗长下的 STFT 结果")
    plot_low_band_profiles(length_results, func_results)

    explore = [
        ("目标：定位 35 Hz 突发；短 Hamming 窗 0.25 s, hop 0.025 s", stft_power(x, FS, 0.25, 0.025, window="hamming", fmin=0, fmax=60)),
        ("目标：区分 10 Hz 与 12 Hz；长 Hamming 窗 1.0 s, hop 0.05 s", stft_power(x, FS, 1.0, 0.05, window="hamming", fmin=0, fmax=60)),
    ]
    plot_stft_panel(explore, "exp2_parameter_exploration.png", "按分析目标选择 STFT 参数")

    metrics = pd.DataFrame(rows)
    metrics.to_csv(TABLE_DIR / "exp2_stft_metrics.csv", index=False, encoding="utf-8-sig")
    summary = {
        "x3_formula": "sin(2*pi*10*t)+0.8*I[1.2,2.2)*sin(2*pi*12*t)+1.2*I[2.6,2.9)*sin(2*pi*35*t)",
        "window_lengths": [r[1].actual_window_s for r in length_results],
        "hop_for_0.05_s_request": length_results[0][1].actual_hop_s,
    }
    (DATA_OUT_DIR / "exp2_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
