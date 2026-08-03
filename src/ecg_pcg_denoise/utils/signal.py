from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import signal


EPS = 1e-8


@dataclass(frozen=True)
class FilterSpec:
    highpass_hz: float | None = None
    lowpass_hz: float | None = None
    order: int = 4


def as_float_1d(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32).squeeze()
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D signal, got shape {arr.shape}")
    return arr


def bandpass_filter(x: np.ndarray, fs: int, spec: FilterSpec) -> np.ndarray:
    x = as_float_1d(x)
    nyq = fs / 2.0
    low = spec.highpass_hz
    high = spec.lowpass_hz
    if low is None and high is None:
        return x.astype(np.float32, copy=False)
    if low is not None and high is not None:
        btype = "bandpass"
        wn = [low / nyq, high / nyq]
    elif low is not None:
        btype = "highpass"
        wn = low / nyq
    else:
        btype = "lowpass"
        wn = high / nyq
    sos = signal.butter(spec.order, wn, btype=btype, output="sos")
    return signal.sosfiltfilt(sos, x).astype(np.float32)


def resample_to(x: np.ndarray, fs_in: int, fs_out: int) -> np.ndarray:
    x = as_float_1d(x)
    if fs_in == fs_out:
        return x.astype(np.float32, copy=False)
    gcd = math.gcd(int(fs_in), int(fs_out))
    up = fs_out // gcd
    down = fs_in // gcd
    return signal.resample_poly(x, up, down).astype(np.float32)


def robust_normalize(x: np.ndarray) -> np.ndarray:
    x = as_float_1d(x)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad < EPS:
        scale = np.std(x) + EPS
    else:
        scale = 1.4826 * mad + EPS
    return ((x - med) / scale).astype(np.float32)


def peak_normalize(x: np.ndarray, peak: float = 0.95) -> np.ndarray:
    x = as_float_1d(x)
    max_abs = np.max(np.abs(x)) + EPS
    return (x / max_abs * peak).astype(np.float32)


def detect_r_peaks(ecg: np.ndarray, fs: int) -> np.ndarray:
    """Simple dependency-light R-peak detector suitable for preprocessing.

    This is intentionally conservative. It filters ECG, emphasizes slope energy,
    and finds peaks separated by at least 300 ms.
    """
    ecg = as_float_1d(ecg)
    filtered = bandpass_filter(ecg, fs, FilterSpec(5.0, 25.0, order=3))
    diff = np.diff(filtered, prepend=filtered[0])
    energy = diff * diff
    smooth_len = max(1, int(0.08 * fs))
    kernel = np.ones(smooth_len, dtype=np.float32) / smooth_len
    integrated = np.convolve(energy, kernel, mode="same")
    height = np.median(integrated) + 2.5 * np.std(integrated)
    min_distance = max(1, int(0.3 * fs))
    peaks, _ = signal.find_peaks(integrated, height=height, distance=min_distance)
    if peaks.size == 0:
        fallback_height = np.percentile(integrated, 97.5)
        peaks, _ = signal.find_peaks(integrated, height=fallback_height, distance=min_distance)
    return peaks.astype(np.int64)


def beat_map_from_r_peaks(
    r_peaks: np.ndarray,
    n_samples: int,
    fs: int,
    sigma_sec: float = 0.06,
) -> np.ndarray:
    t = np.arange(n_samples, dtype=np.float32)
    sigma = max(1.0, sigma_sec * fs)
    beat = np.zeros(n_samples, dtype=np.float32)
    for peak in np.asarray(r_peaks, dtype=np.int64):
        lo = max(0, int(peak - 4 * sigma))
        hi = min(n_samples, int(peak + 4 * sigma) + 1)
        if lo >= hi:
            continue
        beat[lo:hi] = np.maximum(beat[lo:hi], np.exp(-0.5 * ((t[lo:hi] - peak) / sigma) ** 2))
    return beat.astype(np.float32)


def weak_s1s2_from_r_peaks(
    r_peaks: np.ndarray,
    n_samples: int,
    fs: int,
    s1_window: tuple[float, float] = (-0.10, 0.20),
    s2_window: tuple[float, float] = (0.20, 0.50),
) -> np.ndarray:
    labels = np.zeros((2, n_samples), dtype=np.float32)
    sigma = max(1.0, 0.05 * fs)
    t = np.arange(n_samples, dtype=np.float32)
    for peak in np.asarray(r_peaks, dtype=np.int64):
        for row, window in enumerate((s1_window, s2_window)):
            center = peak + int(((window[0] + window[1]) / 2.0) * fs)
            radius = int(max(abs(window[0]), abs(window[1])) * fs)
            lo = max(0, center - radius)
            hi = min(n_samples, center + radius + 1)
            if lo < hi:
                labels[row, lo:hi] = np.maximum(
                    labels[row, lo:hi], np.exp(-0.5 * ((t[lo:hi] - center) / sigma) ** 2)
                )
    return labels


def frame_signal(x: np.ndarray, window_samples: int, hop_samples: int) -> list[tuple[int, np.ndarray]]:
    x = as_float_1d(x)
    if len(x) < window_samples:
        return []
    frames: list[tuple[int, np.ndarray]] = []
    for start in range(0, len(x) - window_samples + 1, hop_samples):
        frames.append((start, x[start : start + window_samples].copy()))
    return frames


def mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> tuple[np.ndarray, np.ndarray]:
    clean = as_float_1d(clean)
    noise = as_float_1d(noise)
    if len(noise) != len(clean):
        raise ValueError("Noise and clean signals must have the same length before SNR mixing.")
    clean_centered = clean - np.mean(clean)
    noise_centered = noise - np.mean(noise)
    clean_rms = np.sqrt(np.mean(clean_centered**2)) + EPS
    noise_rms = np.sqrt(np.mean(noise_centered**2)) + EPS
    target_noise_rms = clean_rms / (10.0 ** (snr_db / 20.0))
    scaled_noise = noise_centered / noise_rms * target_noise_rms
    noisy = clean_centered + scaled_noise
    return noisy.astype(np.float32), scaled_noise.astype(np.float32)


def snr_db(estimate: np.ndarray, reference: np.ndarray) -> float:
    estimate = as_float_1d(estimate)
    reference = as_float_1d(reference)
    noise = reference - estimate
    return float(10.0 * np.log10((np.sum(reference**2) + EPS) / (np.sum(noise**2) + EPS)))


def waveform_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = as_float_1d(x)
    y = as_float_1d(y)
    if len(x) != len(y):
        raise ValueError("Correlation inputs must have equal length.")
    x = x - np.mean(x)
    y = y - np.mean(y)
    denom = np.sqrt(np.sum(x**2) * np.sum(y**2)) + EPS
    return float(np.sum(x * y) / denom)
