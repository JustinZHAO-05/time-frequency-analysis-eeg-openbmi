from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import DATA_OUT_DIR, FIG_DIR, TABLE_DIR, ensure_dirs
from plot_utils import add_panel_labels, annotate_segments, db_power, savefig, set_style, tf_plot
from signal_utils import indicator, prominent_peaks, ridge_frequency, single_sided_spectrum, stft_power, time_vector


FS = 250.0
DURATION = 4.0


def make_x1(t: np.ndarray) -> np.ndarray:
    return (
        indicator(t, 0, 1) * np.sin(2 * np.pi * 8 * t)
        + indicator(t, 1, 2) * np.sin(2 * np.pi * 16 * t)
        + indicator(t, 2, 3) * np.sin(2 * np.pi * 32 * t)
        + indicator(t, 3, 4) * np.sin(2 * np.pi * 12 * t)
    )


def make_x2(t: np.ndarray) -> np.ndarray:
    return np.sin(2 * np.pi * (5 * t + 5 * t**2))


def plot_overview(
    name: str,
    t: np.ndarray,
    x: np.ndarray,
    stft,
    *,
    segments: list[tuple[float, float, str]] | None = None,
    true_freq: np.ndarray | None = None,
) -> None:
    freqs, amp, _ = single_sided_spectrum(x, FS)
    fig, axes = plt.subplots(3, 1, figsize=(9.8, 9.2), constrained_layout=True)
    axes[0].plot(t, x, color="#1b4f72", lw=1.0)
    axes[0].set_title(f"{name} 时域波形")
    axes[0].set_xlabel("时间 (s)")
    axes[0].set_ylabel("幅值")
    axes[0].set_xlim(0, DURATION)
    if segments:
        annotate_segments(axes[0], segments)
    axes[1].plot(freqs, amp, color="#7d3c98", lw=1.4)
    axes[1].set_title("整段单边傅里叶幅度谱")
    axes[1].set_xlabel("频率 (Hz)")
    axes[1].set_ylabel("幅度")
    axes[1].set_xlim(0, 60)
    axes[1].set_ylim(bottom=0)
    overlay = (t, true_freq, "真实瞬时频率") if true_freq is not None else None
    tf_plot(
        axes[2],
        stft.times,
        stft.freqs,
        db_power(stft.power),
        title=f"STFT 时频图（Hamming, {stft.actual_window_s:.3f} s 窗, {stft.actual_hop_s:.3f} s hop）",
        colorbar=True,
        overlay=overlay,
        cbar_label="功率 (dB)",
    )
    axes[2].set_xlim(0, DURATION)
    add_panel_labels(axes)
    savefig(fig, FIG_DIR / f"exp1_{name}_overview.png")


def main() -> None:
    ensure_dirs()
    set_style()
    t = time_vector(FS, DURATION)
    x1 = make_x1(t)
    x2 = make_x2(t)

    stft1 = stft_power(x1, FS, 0.5, 0.05, window="hamming", fmin=0, fmax=60)
    stft2 = stft_power(x2, FS, 0.5, 0.05, window="hamming", fmin=0, fmax=60)

    plot_overview(
        "x1",
        t,
        x1,
        stft1,
        segments=[(0, 1, "8 Hz"), (1, 2, "16 Hz"), (2, 3, "32 Hz"), (3, 4, "12 Hz")],
    )
    true_freq = 5 + 10 * t
    plot_overview("x2", t, x2, stft2, true_freq=true_freq)

    ridge = ridge_frequency(stft2.power, stft2.freqs, 3, 50)
    true_at_stft = 5 + 10 * stft2.times
    ridge_error = ridge - true_at_stft
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 6.4), constrained_layout=True, sharex=True)
    axes[0].plot(stft2.times, true_at_stft, color="black", lw=1.8, label="真实瞬时频率")
    axes[0].plot(stft2.times, ridge, color="#d35400", lw=1.4, label="STFT 主能量轨迹")
    axes[0].set_ylabel("频率 (Hz)")
    axes[0].set_title("线性调频信号的 STFT ridge 与真实瞬时频率")
    axes[0].legend()
    axes[1].plot(stft2.times, ridge_error, color="#a93226", lw=1.2)
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_xlabel("时间 (s)")
    axes[1].set_ylabel("误差 (Hz)")
    axes[1].set_title("主频轨迹误差")
    savefig(fig, FIG_DIR / "exp1_chirp_ridge_error.png")

    rows = []
    for name, x, stft in [("x1", x1, stft1), ("x2", x2, stft2)]:
        freqs, amp, _ = single_sided_spectrum(x, FS)
        for rank, (freq, value) in enumerate(prominent_peaks(freqs, amp, 0, 60, n=8), start=1):
            rows.append({"signal": name, "rank": rank, "fft_peak_hz": freq, "amplitude": value})
        if name == "x2":
            rows.append(
                {
                    "signal": "x2_ridge",
                    "rank": 1,
                    "fft_peak_hz": float(np.nanmean(np.abs(ridge_error))),
                    "amplitude": float(np.sqrt(np.nanmean(ridge_error**2))),
                }
            )
    pd.DataFrame(rows).to_csv(TABLE_DIR / "exp1_fft_peaks.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        {
            "time_s": stft2.times,
            "ridge_hz": ridge,
            "true_hz": true_at_stft,
            "error_hz": ridge_error,
        }
    ).to_csv(DATA_OUT_DIR / "exp1_chirp_ridge.csv", index=False, encoding="utf-8-sig")
    summary = {
        "x2_ridge_mae_hz": float(np.nanmean(np.abs(ridge_error))),
        "x2_ridge_rmse_hz": float(np.sqrt(np.nanmean(ridge_error**2))),
        "stft_window_samples": stft1.nperseg,
        "stft_hop_samples": stft1.hop,
        "stft_actual_window_s": stft1.actual_window_s,
        "stft_actual_hop_s": stft1.actual_hop_s,
    }
    (DATA_OUT_DIR / "exp1_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
