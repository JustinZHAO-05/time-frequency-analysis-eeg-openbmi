from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pywt
from scipy import signal


WindowName = Literal["boxcar", "hann", "hamming"]


@dataclass(frozen=True)
class StftResult:
    freqs: np.ndarray
    times: np.ndarray
    power: np.ndarray
    nperseg: int
    hop: int
    actual_window_s: float
    actual_hop_s: float


def time_vector(fs: float, duration: float) -> np.ndarray:
    return np.arange(0, duration, 1.0 / fs)


def indicator(t: np.ndarray, start: float, end: float) -> np.ndarray:
    return ((t >= start) & (t < end)).astype(float)


def seconds_to_samples(seconds: float, fs: float) -> int:
    """Convert a positive duration to sample count with half-up rounding."""
    return int(np.floor(seconds * fs + 0.5 + 1e-12))


def single_sided_spectrum(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    n = x.size
    win = signal.windows.hann(n, sym=False)
    scale = np.sum(win) / 2.0
    spec = np.fft.rfft(x * win)
    freqs = np.fft.rfftfreq(n, d=1 / fs)
    amp = np.abs(spec) / max(scale, np.finfo(float).eps)
    power = (np.abs(spec) ** 2) / max(np.sum(win**2), np.finfo(float).eps)
    return freqs, amp, power


def prominent_peaks(freqs: np.ndarray, amp: np.ndarray, fmin: float, fmax: float, n: int = 6) -> list[tuple[float, float]]:
    mask = (freqs >= fmin) & (freqs <= fmax)
    ff = freqs[mask]
    aa = amp[mask]
    if aa.size == 0:
        return []
    peaks, props = signal.find_peaks(aa, prominence=np.nanmax(aa) * 0.03)
    if peaks.size == 0:
        order = np.argsort(aa)[-n:][::-1]
    else:
        order = peaks[np.argsort(props["prominences"])[-n:][::-1]]
    return [(float(ff[i]), float(aa[i])) for i in order[:n]]


def stft_power(
    x: np.ndarray,
    fs: float,
    window_s: float,
    hop_s: float,
    *,
    window: WindowName = "hamming",
    fmin: float = 0.0,
    fmax: float | None = None,
    nfft_min: int = 1024,
) -> StftResult:
    nperseg = seconds_to_samples(window_s, fs)
    hop = seconds_to_samples(hop_s, fs)
    nperseg = max(nperseg, 2)
    hop = max(hop, 1)
    noverlap = max(nperseg - hop, 0)
    nfft = max(nfft_min, int(2 ** np.ceil(np.log2(max(nperseg, 2)))))
    freqs, times, zxx = signal.stft(
        x,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        detrend=False,
        boundary=None,
        padded=False,
    )
    power = np.abs(zxx) ** 2
    if fmax is None:
        fmax = fs / 2.0
    mask = (freqs >= fmin) & (freqs <= fmax)
    return StftResult(
        freqs=freqs[mask],
        times=times,
        power=power[mask, :],
        nperseg=nperseg,
        hop=hop,
        actual_window_s=nperseg / fs,
        actual_hop_s=hop / fs,
    )


def ridge_frequency(tf_power: np.ndarray, freqs: np.ndarray, fmin: float, fmax: float) -> np.ndarray:
    mask = (freqs >= fmin) & (freqs <= fmax)
    sub = tf_power[mask, :]
    ff = freqs[mask]
    idx = np.nanargmax(sub, axis=0)
    return ff[idx]


def band_time_metrics(
    tf_power: np.ndarray,
    freqs: np.ndarray,
    times: np.ndarray,
    band: tuple[float, float],
    *,
    threshold_ratio: float = 0.5,
) -> dict[str, float]:
    mask = (freqs >= band[0]) & (freqs <= band[1])
    curve = np.nanmean(tf_power[mask, :], axis=0)
    peak_idx = int(np.nanargmax(curve))
    peak = float(curve[peak_idx])
    threshold = peak * threshold_ratio
    above = np.where(curve >= threshold)[0]
    if above.size == 0:
        start = end = times[peak_idx]
    else:
        start = times[int(above[0])]
        end = times[int(above[-1])]
    return {
        "peak_time_s": float(times[peak_idx]),
        "spread_start_s": float(start),
        "spread_end_s": float(end),
        "spread_width_s": float(max(end - start, 0.0)),
        "peak_power": peak,
    }


def cwt_power(
    x: np.ndarray,
    fs: float,
    freqs: np.ndarray,
    *,
    wavelet: str = "cmor1.5-1.0",
) -> tuple[np.ndarray, np.ndarray]:
    center = pywt.central_frequency(wavelet)
    scales = center * fs / freqs
    coeff, actual_freqs = pywt.cwt(x, scales, wavelet, sampling_period=1 / fs)
    return actual_freqs, np.abs(coeff) ** 2


def dwt_components(
    x: np.ndarray,
    fs: float,
    *,
    wavelet: str = "sym6",
    level: int = 6,
) -> tuple[dict[str, np.ndarray], list[dict[str, float | str]]]:
    coeffs = pywt.wavedec(x, wavelet=wavelet, mode="periodization", level=level)
    n = len(x)
    components: dict[str, np.ndarray] = {}
    rows: list[dict[str, float | str]] = []

    zero = [np.zeros_like(c) for c in coeffs]
    ca = [z.copy() for z in zero]
    ca[0] = coeffs[0].copy()
    components[f"A{level}"] = pywt.waverec(ca, wavelet=wavelet, mode="periodization")[:n]
    rows.append({"component": f"A{level}", "f_low_hz": 0.0, "f_high_hz": fs / (2 ** (level + 1))})

    for j in range(level, 0, -1):
        idx = level - j + 1
        cc = [z.copy() for z in zero]
        cc[idx] = coeffs[idx].copy()
        components[f"D{j}"] = pywt.waverec(cc, wavelet=wavelet, mode="periodization")[:n]
        rows.append(
            {
                "component": f"D{j}",
                "f_low_hz": fs / (2 ** (j + 1)),
                "f_high_hz": fs / (2**j),
            }
        )
    return components, rows


def mean_band(values: np.ndarray, freqs: np.ndarray, band: tuple[float, float]) -> np.ndarray:
    mask = (freqs >= band[0]) & (freqs <= band[1])
    return np.nanmean(values[mask, :], axis=0)
