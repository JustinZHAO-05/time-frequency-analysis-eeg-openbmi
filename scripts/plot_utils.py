from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


def _pick_chinese_font() -> str:
    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Arial Unicode MS",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            return name
    return "DejaVu Sans"


def set_style() -> None:
    font = _pick_chinese_font()
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [font, "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.dpi": 120,
            "savefig.dpi": 320,
            "savefig.bbox": "tight",
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.5,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.frameon": False,
        }
    )


def savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def annotate_segments(ax: plt.Axes, segments: Iterable[tuple[float, float, str]], y: float | None = None) -> None:
    ymin, ymax = ax.get_ylim()
    y_text = y if y is not None else ymax - 0.08 * (ymax - ymin)
    for start, end, label in segments:
        ax.axvspan(start, end, color="0.90", alpha=0.35, lw=0)
        ax.text((start + end) / 2, y_text, label, ha="center", va="top", fontsize=9)


def tf_plot(
    ax: plt.Axes,
    times: np.ndarray,
    freqs: np.ndarray,
    values: np.ndarray,
    *,
    title: str,
    cmap: str = "magma",
    vmin: float | None = None,
    vmax: float | None = None,
    ylabel: str = "频率 (Hz)",
    xlabel: str = "时间 (s)",
    colorbar: bool = False,
    cbar_label: str = "功率 (dB)",
    overlay: tuple[np.ndarray, np.ndarray, str] | None = None,
    vlines: Iterable[float] = (),
) -> mpl.collections.QuadMesh:
    mesh = ax.pcolormesh(times, freqs, values, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    for x in vlines:
        ax.axvline(x, color="white", lw=1.0, ls="--", alpha=0.85)
    if overlay is not None:
        tx, fy, label = overlay
        ax.plot(tx, fy, color="cyan", lw=1.5, label=label)
        ax.legend(loc="upper left", fontsize=8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(float(np.nanmin(freqs)), float(np.nanmax(freqs)))
    if colorbar:
        cb = ax.figure.colorbar(mesh, ax=ax, pad=0.012)
        cb.set_label(cbar_label)
    return mesh


def add_panel_labels(axes: Iterable[plt.Axes]) -> None:
    for i, ax in enumerate(axes):
        ax.text(
            -0.08,
            1.04,
            f"({chr(65 + i)})",
            transform=ax.transAxes,
            fontsize=11,
            fontweight="bold",
            va="bottom",
            ha="right",
        )


def db_power(power: np.ndarray, eps: float = 1e-14) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(power, eps))


def symmetric_limits(values: Iterable[np.ndarray], q: float = 98.0, minimum: float = 1.0) -> tuple[float, float]:
    arr = np.concatenate([np.asarray(v, dtype=float).ravel() for v in values])
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return -minimum, minimum
    m = float(np.nanpercentile(np.abs(arr), q))
    m = max(m, minimum)
    return -m, m
