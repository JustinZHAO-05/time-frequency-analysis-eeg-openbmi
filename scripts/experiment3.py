from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import DATA_OUT_DIR, FIG_DIR, TABLE_DIR, ensure_dirs
from plot_utils import add_panel_labels, annotate_segments, db_power, savefig, set_style, tf_plot
from signal_utils import (
    band_time_metrics,
    cwt_power,
    dwt_components,
    indicator,
    mean_band,
    single_sided_spectrum,
    stft_power,
    time_vector,
)


FS = 250.0
DURATION = 4.0


def make_x4(t: np.ndarray) -> np.ndarray:
    return (
        np.sin(2 * np.pi * 6 * t)
        + 0.8 * indicator(t, 1.0, 2.5) * np.sin(2 * np.pi * 14 * t)
        + 1.2 * indicator(t, 3.0, 3.25) * np.sin(2 * np.pi * 45 * t)
    )


def make_xmra(t: np.ndarray) -> np.ndarray:
    return (
        0.5 * np.sin(2 * np.pi * 1 * t)
        + 0.8 * np.sin(2 * np.pi * 6 * t)
        + 0.6 * np.sin(2 * np.pi * 12 * t)
        + indicator(t, 2.4, 2.8) * np.sin(2 * np.pi * 45 * t)
    )


def plot_x4_compare(t: np.ndarray, x: np.ndarray) -> pd.DataFrame:
    short = stft_power(x, FS, 0.25, 0.05, window="hamming", fmin=2, fmax=60)
    long = stft_power(x, FS, 1.0, 0.05, window="hamming", fmin=2, fmax=60)
    cwt_freqs = np.arange(2.0, 60.5, 0.5)
    actual_cwt_freqs, cwt = cwt_power(x, FS, cwt_freqs)
    cwt_times = t

    panels = [
        ("短窗 STFT：0.25 s Hamming", short.times, short.freqs, db_power(short.power)),
        ("长窗 STFT：1.0 s Hamming", long.times, long.freqs, db_power(long.power)),
        ("CWT：Morlet 类小波", cwt_times, actual_cwt_freqs, db_power(cwt)),
    ]
    all_vals = np.concatenate([p[3].ravel() for p in panels])
    vmin, vmax = np.nanpercentile(all_vals, [5, 99])
    fig, axes = plt.subplots(3, 1, figsize=(9.8, 10), constrained_layout=True, sharex=False)
    for ax, (title, tt, ff, val) in zip(axes, panels):
        tf_plot(
            ax,
            tt,
            ff,
            val,
            title=title,
            vmin=float(vmin),
            vmax=float(vmax),
            colorbar=True,
            vlines=[1.0, 2.5, 3.0, 3.25],
        )
        ax.set_xlim(0, DURATION)
    add_panel_labels(axes)
    savefig(fig, FIG_DIR / "exp3_x4_stft_cwt_compare.png")

    rows = []
    for method, tt, ff, power in [
        ("STFT_short", short.times, short.freqs, short.power),
        ("STFT_long", long.times, long.freqs, long.power),
        ("CWT", cwt_times, actual_cwt_freqs, cwt),
    ]:
        burst = band_time_metrics(power, ff, tt, (40, 50))
        low_curve = mean_band(power, ff, (5, 7))
        low_norm = low_curve / max(np.nanpercentile(low_curve, 95), np.finfo(float).eps)
        rows.append(
            {
                "method": method,
                **burst,
                "low_6hz_continuity_ratio": float(np.nanmean(low_norm > 0.35)),
                "low_6hz_curve_cv": float(np.nanstd(low_curve) / max(np.nanmean(low_curve), np.finfo(float).eps)),
            }
        )
    return pd.DataFrame(rows)


def plot_mra(t: np.ndarray, x: np.ndarray) -> pd.DataFrame:
    freqs, amp, _ = single_sided_spectrum(x, FS)
    cwt_freqs = np.arange(1.0, 60.5, 0.5)
    actual_cwt_freqs, cwt = cwt_power(x, FS, cwt_freqs)

    fig, axes = plt.subplots(3, 1, figsize=(9.8, 9), constrained_layout=True)
    axes[0].plot(t, x, color="#1b4f72", lw=1.0)
    axes[0].set_title("xMRA 时域波形")
    axes[0].set_xlabel("时间 (s)")
    axes[0].set_ylabel("幅值")
    axes[0].set_xlim(0, DURATION)
    annotate_segments(axes[0], [(2.4, 2.8, "45 Hz 突发")])
    axes[1].plot(freqs, amp, color="#7d3c98", lw=1.4)
    axes[1].set_xlim(0, 60)
    axes[1].set_ylim(bottom=0)
    axes[1].set_title("xMRA 单边傅里叶幅度谱")
    axes[1].set_xlabel("频率 (Hz)")
    axes[1].set_ylabel("幅度")
    tf_plot(
        axes[2],
        t,
        actual_cwt_freqs,
        db_power(cwt),
        title="xMRA CWT 时频图",
        colorbar=True,
        vlines=[2.4, 2.8],
    )
    axes[2].set_xlim(0, DURATION)
    add_panel_labels(axes)
    savefig(fig, FIG_DIR / "exp3_mra_signal_fft_cwt.png")

    components, bands = dwt_components(x, FS, wavelet="sym6", level=6)
    ordered = ["原始", "D1", "D2", "D3", "D4", "D5", "D6", "A6"]
    series = {"原始": x, **components}
    fig, axes = plt.subplots(len(ordered), 1, figsize=(10.2, 13.5), constrained_layout=True, sharex=True)
    for ax, name in zip(axes, ordered):
        if name not in series:
            continue
        ax.plot(t, series[name], lw=0.85, color="#154360" if name == "原始" else "#7d3c98")
        ax.set_ylabel(name)
        ax.set_xlim(0, DURATION)
        if name == "D2":
            ax.axvspan(2.4, 2.8, color="#f5b041", alpha=0.25, lw=0)
    axes[0].set_title("离散小波多分辨率分解（sym6, 6 层）")
    axes[-1].set_xlabel("时间 (s)")
    savefig(fig, FIG_DIR / "exp3_mra_components.png")

    recon_low = components["A6"] + components["D6"] + components["D5"] + components["D4"]
    recon_burst = components["D2"]
    fig, axes = plt.subplots(3, 1, figsize=(9.8, 7.5), constrained_layout=True, sharex=True)
    axes[0].plot(t, x, color="black", lw=1.0)
    axes[0].set_title("原始 xMRA")
    axes[1].plot(t, recon_low, color="#1f618d", lw=1.0)
    axes[1].set_title("重构组合 A6+D6+D5+D4：保留慢变、6 Hz 与约 12 Hz 成分")
    axes[2].plot(t, recon_burst, color="#b03a2e", lw=1.0)
    axes[2].set_title("重构 D2：突出 45 Hz 短时突发")
    axes[2].set_xlabel("时间 (s)")
    for ax in axes:
        ax.set_xlim(0, DURATION)
        ax.set_ylabel("幅值")
    savefig(fig, FIG_DIR / "exp3_mra_reconstruction.png")

    band_df = pd.DataFrame(bands)
    band_df.to_csv(TABLE_DIR / "exp3_mra_frequency_bands.csv", index=False, encoding="utf-8-sig")
    return band_df


def main() -> None:
    ensure_dirs()
    set_style()
    t = time_vector(FS, DURATION)
    x4 = make_x4(t)
    xmra = make_xmra(t)

    metrics = plot_x4_compare(t, x4)
    metrics.to_csv(TABLE_DIR / "exp3_stft_cwt_metrics.csv", index=False, encoding="utf-8-sig")
    bands = plot_mra(t, xmra)

    summary = {
        "cwt_wavelet": "cmor1.5-1.0",
        "dwt_wavelet": "sym6",
        "dwt_level": 6,
        "frequency_note": "任务书中 xMRA 公式含 12 Hz；若文字处出现 142Hz，按公式解释为 12 Hz。",
        "bands": bands.to_dict(orient="records"),
    }
    (DATA_OUT_DIR / "exp3_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
