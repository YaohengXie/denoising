from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ecg_pcg_denoise.utils.metrics import evaluate_pair  # noqa: E402
from ecg_pcg_denoise.utils.signal import (  # noqa: E402
    FilterSpec,
    bandpass_filter,
    beat_map_from_r_peaks,
    detect_r_peaks,
    mix_at_snr,
    resample_to,
    robust_normalize,
    waveform_corr,
)


def synthetic_ecg(fs: int, seconds: float) -> tuple[np.ndarray, np.ndarray]:
    n = int(fs * seconds)
    ecg = np.zeros(n, dtype=np.float32)
    peaks = np.arange(int(0.5 * fs), n, int(1.0 * fs))
    for peak in peaks:
        width = int(0.025 * fs)
        lo = max(0, peak - width)
        hi = min(n, peak + width + 1)
        x = np.arange(lo, hi) - peak
        ecg[lo:hi] += np.exp(-0.5 * (x / (0.008 * fs)) ** 2)
    ecg += 0.02 * np.random.default_rng(1).normal(size=n)
    return ecg.astype(np.float32), peaks.astype(np.int64)


def test_signal_processing_smoke() -> None:
    """Exercise the released signal-processing and metric path without project data."""
    fs = 2000
    seconds = 4.0
    t = np.arange(int(fs * seconds)) / fs
    clean = (0.3 * np.sin(2 * np.pi * 45 * t) + 0.1 * np.sin(2 * np.pi * 90 * t)).astype(np.float32)
    ecg, true_peaks = synthetic_ecg(fs, seconds)

    filtered = bandpass_filter(clean, fs, FilterSpec(20.0, 800.0))
    normalized = robust_normalize(filtered)
    down_up = resample_to(resample_to(normalized, fs, 1000), 1000, fs)
    assert len(down_up) == len(clean)

    detected = detect_r_peaks(ecg, fs)
    assert len(detected) >= len(true_peaks) - 1

    beat = beat_map_from_r_peaks(detected, len(ecg), fs)
    assert beat.shape == ecg.shape
    assert float(beat.max()) > 0.9

    noise = np.random.default_rng(2).normal(size=len(clean)).astype(np.float32)
    noisy, scaled_noise = mix_at_snr(clean, noise, 0.0)
    assert noisy.shape == clean.shape
    assert scaled_noise.shape == clean.shape
    assert waveform_corr(clean, clean) > 0.999

    metrics = evaluate_pair(noisy, clean, clean, fs)
    assert metrics["delta_snr"] > 20.0


def main() -> None:
    """Retain the original direct-execution smoke-check interface."""
    test_signal_processing_smoke()
    print("smoke_test passed")


if __name__ == "__main__":
    main()
