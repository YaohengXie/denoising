from __future__ import annotations

import numpy as np
from scipy import signal

from ecg_pcg_denoise.utils.signal import EPS, as_float_1d, snr_db, waveform_corr


def si_sdr_db(estimate: np.ndarray, reference: np.ndarray) -> float:
    estimate = as_float_1d(estimate)
    reference = as_float_1d(reference)
    estimate = estimate - np.mean(estimate)
    reference = reference - np.mean(reference)
    scale = np.sum(estimate * reference) / (np.sum(reference**2) + EPS)
    target = scale * reference
    noise = estimate - target
    return float(10.0 * np.log10((np.sum(target**2) + EPS) / (np.sum(noise**2) + EPS)))


def log_spectral_distance(estimate: np.ndarray, reference: np.ndarray, fs: int) -> float:
    estimate = as_float_1d(estimate)
    reference = as_float_1d(reference)
    _, _, est = signal.stft(estimate, fs=fs, nperseg=256, noverlap=224)
    _, _, ref = signal.stft(reference, fs=fs, nperseg=256, noverlap=224)
    est_log = np.log1p(np.abs(est))
    ref_log = np.log1p(np.abs(ref))
    return float(np.mean(np.abs(est_log - ref_log)))


def evaluate_pair(noisy: np.ndarray, estimate: np.ndarray, clean: np.ndarray, fs: int) -> dict[str, float]:
    return {
        "snr_noisy": snr_db(noisy, clean),
        "snr_estimate": snr_db(estimate, clean),
        "delta_snr": snr_db(estimate, clean) - snr_db(noisy, clean),
        "si_sdr_noisy": si_sdr_db(noisy, clean),
        "si_sdr_estimate": si_sdr_db(estimate, clean),
        "delta_si_sdr": si_sdr_db(estimate, clean) - si_sdr_db(noisy, clean),
        "corr_noisy": waveform_corr(noisy, clean),
        "corr_estimate": waveform_corr(estimate, clean),
        "log_spectral_distance": log_spectral_distance(estimate, clean, fs),
    }
