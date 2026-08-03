from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy import signal

from ecg_pcg_denoise.data.noise_mixer import sqi_target_from_snr
from ecg_pcg_denoise.utils.signal import EPS, as_float_1d


IMU_FEATURE_NAMES = (
    "dynamic_ax_mean",
    "dynamic_ay_mean",
    "dynamic_az_mean",
    "dynamic_acc_rms",
    "jerk_rms",
    "magnitude_deviation",
)

SYNTHESIS_MODES = (
    "motion_artifact",
    "combined",
    "independent_artifact",
    "motion_clean",
    "clean_identity",
)


@dataclass(frozen=True)
class IMUFeatureStats:
    median: np.ndarray
    scale: np.ndarray
    activity_scale: float

    def to_dict(self) -> dict[str, object]:
        return {
            "feature_names": list(IMU_FEATURE_NAMES),
            "median": np.asarray(self.median, dtype=float).tolist(),
            "scale": np.asarray(self.scale, dtype=float).tolist(),
            "activity_scale": float(self.activity_scale),
        }

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> "IMUFeatureStats":
        names = tuple(str(name) for name in values["feature_names"])  # type: ignore[index]
        if names != IMU_FEATURE_NAMES:
            raise ValueError(f"Unexpected IMU feature order: {names}")
        median = np.asarray(values["median"], dtype=np.float32)
        scale = np.asarray(values["scale"], dtype=np.float32)
        if median.shape != (len(IMU_FEATURE_NAMES),) or scale.shape != median.shape:
            raise ValueError("IMU normalization statistics have invalid dimensions.")
        return cls(
            median=median,
            scale=np.maximum(scale, 1e-6),
            activity_scale=max(float(values["activity_scale"]), 1e-6),
        )


@dataclass(frozen=True)
class IMUSynthesisResult:
    noisy_pcg: np.ndarray
    noise_signal: np.ndarray
    snr_db: float
    achieved_snr_db: float
    sqi_target: float
    components: str


def stft_frame_count(n_samples: int, hop_length: int) -> int:
    """Frame count for torch.stft(center=True), used by the M7 pipeline."""
    if n_samples <= 0 or hop_length <= 0:
        raise ValueError("n_samples and hop_length must be positive.")
    return 1 + n_samples // hop_length


def _fill_nonfinite_columns(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(f"Expected IMU shape [samples, 3], got {values.shape}")
    finite = np.all(np.isfinite(values), axis=1)
    output = values.copy()
    indices = np.arange(len(values))
    for axis in range(3):
        good = np.isfinite(output[:, axis])
        if not np.any(good):
            raise ValueError(f"IMU axis {axis} has no finite values.")
        output[:, axis] = np.interp(indices, indices[good], output[good, axis])
    return output, finite


def _moving_mean(values: np.ndarray, width: int) -> np.ndarray:
    width = max(1, min(int(width), len(values)))
    kernel = np.ones(width, dtype=np.float64) / width
    left = (width - 1) // 2
    right = width - 1 - left
    padded = np.pad(np.asarray(values, dtype=np.float64), (left, right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def extract_imu_frame_features(
    imu: np.ndarray,
    fs: int,
    hop_length: int,
    win_length: int,
    n_frames: int | None = None,
    gravity_cutoff_hz: float = 0.7,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert interpolated three-axis acceleration to STFT-aligned motion features.

    The Motema window stores acceleration on the 2 kHz PCG time grid. This
    function intentionally reduces it to one feature vector per PCG STFT frame;
    it does not claim that the interpolated acceleration has a 2 kHz bandwidth.
    """
    if fs <= 0 or hop_length <= 0 or win_length <= 0:
        raise ValueError("fs, hop_length, and win_length must be positive.")
    imu_filled, finite = _fill_nonfinite_columns(imu)
    if len(imu_filled) < max(32, win_length):
        raise ValueError("IMU window is too short for feature extraction.")
    if n_frames is None:
        n_frames = stft_frame_count(len(imu_filled), hop_length)

    nyquist = fs / 2.0
    cutoff = min(max(gravity_cutoff_hz, 0.05), nyquist * 0.25)
    sos = signal.butter(2, cutoff / nyquist, btype="lowpass", output="sos")
    trend = np.column_stack(
        [signal.sosfiltfilt(sos, imu_filled[:, axis]) for axis in range(3)]
    )
    dynamic = imu_filled.astype(np.float64) - trend
    jerk = np.gradient(dynamic, axis=0) * fs
    magnitude = np.linalg.norm(imu_filled, axis=1)
    magnitude_deviation = np.abs(magnitude - np.median(magnitude))

    feature_samples = np.column_stack(
        (
            *(_moving_mean(dynamic[:, axis], win_length) for axis in range(3)),
            np.sqrt(_moving_mean(np.sum(dynamic**2, axis=1), win_length) + EPS),
            np.sqrt(_moving_mean(np.sum(jerk**2, axis=1), win_length) + EPS),
            _moving_mean(magnitude_deviation, win_length),
        )
    )
    valid_samples = _moving_mean(finite.astype(np.float32), win_length)
    centers = np.minimum(np.arange(n_frames, dtype=np.int64) * hop_length, len(imu_filled) - 1)
    features = feature_samples[centers].T.astype(np.float32)
    frame_valid = (valid_samples[centers] >= 0.95).astype(np.float32)
    return features, frame_valid


def fit_imu_feature_stats(feature_arrays: Iterable[np.ndarray]) -> IMUFeatureStats:
    arrays = [np.asarray(values, dtype=np.float32) for values in feature_arrays]
    if not arrays:
        raise ValueError("At least one IMU feature array is required.")
    for values in arrays:
        if values.ndim != 2 or values.shape[0] != len(IMU_FEATURE_NAMES):
            raise ValueError(f"Unexpected IMU feature shape: {values.shape}")
    combined = np.concatenate(arrays, axis=1)
    median = np.median(combined, axis=1)
    q25, q75 = np.percentile(combined, (25.0, 75.0), axis=1)
    robust_scale = (q75 - q25) / 1.349
    std = np.std(combined, axis=1)
    scale = np.maximum(robust_scale, np.maximum(std * 0.05, 1e-6))
    normalized = (combined - median[:, None]) / scale[:, None]
    activity = _standardized_activity(normalized)
    positive = activity[activity > 0]
    activity_scale = float(np.percentile(positive, 95.0)) if positive.size else 1.0
    return IMUFeatureStats(
        median=median.astype(np.float32),
        scale=scale.astype(np.float32),
        activity_scale=max(activity_scale, 1e-6),
    )


def normalize_imu_features(features: np.ndarray, stats: IMUFeatureStats) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    if features.shape[0] != len(IMU_FEATURE_NAMES):
        raise ValueError(f"Unexpected IMU feature shape: {features.shape}")
    normalized = (features - stats.median[:, None]) / stats.scale[:, None]
    return np.clip(normalized, -8.0, 8.0).astype(np.float32)


def _standardized_activity(normalized_features: np.ndarray) -> np.ndarray:
    normalized_features = np.asarray(normalized_features, dtype=np.float32)
    dynamic_rms = np.maximum(normalized_features[3], 0.0)
    jerk_rms = np.maximum(normalized_features[4], 0.0)
    magnitude_deviation = np.maximum(normalized_features[5], 0.0)
    return 0.55 * dynamic_rms + 0.30 * jerk_rms + 0.15 * magnitude_deviation


def motion_envelope_from_features(
    normalized_features: np.ndarray,
    stats: IMUFeatureStats,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    activity = _standardized_activity(normalized_features) / stats.activity_scale
    activity = np.clip(activity, 0.0, 1.0)
    if len(activity) >= 3:
        padded = np.pad(activity, (1, 1), mode="edge")
        activity = np.convolve(padded, np.asarray([0.2, 0.6, 0.2]), mode="valid")
    if valid_mask is not None:
        activity *= np.asarray(valid_mask, dtype=np.float32)
    return np.clip(activity, 0.0, 1.0).astype(np.float32)


def motion_score(envelope: np.ndarray) -> float:
    envelope = np.asarray(envelope, dtype=np.float32)
    if envelope.size == 0:
        return 0.0
    return float(np.clip(0.6 * np.mean(envelope) + 0.4 * np.percentile(envelope, 90), 0, 1))


def choose_motion_conditioned_snr(
    snr_values: Iterable[float],
    activity_score: float,
    rng: np.random.Generator,
) -> float:
    values = sorted((float(value) for value in snr_values), reverse=True)
    if not values:
        raise ValueError("At least one SNR value is required.")
    if len(values) <= 2:
        choices = values
    elif activity_score < 0.20:
        choices = values[: max(2, len(values) // 2)]
    elif activity_score < 0.55:
        middle = len(values) // 2
        choices = values[max(0, middle - 1) : min(len(values), middle + 2)]
    else:
        choices = values[-max(2, len(values) // 2) :]
    return float(rng.choice(choices))


def _band_limited_noise(
    n_samples: int,
    fs: int,
    rng: np.random.Generator,
    low_hz: float,
    high_hz: float,
) -> np.ndarray:
    high_hz = min(high_hz, fs * 0.45)
    low_hz = min(max(low_hz, 0.5), high_hz * 0.8)
    raw = rng.normal(0.0, 1.0, size=n_samples)
    sos = signal.butter(
        3,
        [low_hz / (fs / 2.0), high_hz / (fs / 2.0)],
        btype="bandpass",
        output="sos",
    )
    filtered = signal.sosfiltfilt(sos, raw)
    return (filtered / (np.std(filtered) + EPS)).astype(np.float32)


def _envelope_to_samples(envelope: np.ndarray, n_samples: int, hop_length: int) -> np.ndarray:
    frames = np.arange(len(envelope), dtype=np.float64) * hop_length
    samples = np.arange(n_samples, dtype=np.float64)
    values = np.interp(samples, frames, envelope, left=envelope[0], right=envelope[-1])
    smooth_width = max(3, int(round(0.025 * n_samples / 4.0)))
    if smooth_width % 2 == 0:
        smooth_width += 1
    kernel = signal.windows.hann(smooth_width)
    kernel /= np.sum(kernel) + EPS
    padded = np.pad(values, (smooth_width // 2, smooth_width // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def _impact_artifacts(
    normalized_features: np.ndarray,
    n_samples: int,
    fs: int,
    hop_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    jerk = np.maximum(np.asarray(normalized_features[4], dtype=np.float32), 0.0)
    if not np.any(jerk > 0):
        return np.zeros(n_samples, dtype=np.float32)
    prominence = max(0.5, float(np.percentile(jerk, 75)) * 0.5)
    peaks, properties = signal.find_peaks(
        jerk,
        distance=max(1, int(round(0.12 * fs / hop_length))),
        prominence=prominence,
    )
    if len(peaks) > 6:
        order = np.argsort(properties["prominences"])[-6:]
        peaks = peaks[order]
    output = np.zeros(n_samples, dtype=np.float32)
    for peak in peaks:
        start = min(int(peak * hop_length), n_samples - 1)
        duration = min(int(rng.uniform(0.04, 0.18) * fs), n_samples - start)
        if duration < 4:
            continue
        local_t = np.arange(duration) / fs
        decay = np.exp(-local_t / float(rng.uniform(0.015, 0.070)))
        frequency = float(rng.uniform(25.0, 170.0))
        transient = decay * np.sin(2.0 * np.pi * frequency * local_t)
        transient += 0.25 * decay * rng.normal(size=duration)
        strength = float(np.clip(jerk[peak] / 5.0, 0.15, 1.5))
        output[start : start + duration] += (strength * transient).astype(np.float32)
    return output


def _scale_distortion_at_snr(
    clean: np.ndarray,
    distortion: np.ndarray,
    snr_db: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    clean = as_float_1d(clean)
    distortion = as_float_1d(distortion)
    clean_power = float(np.mean((clean - np.mean(clean)) ** 2)) + EPS
    centered_distortion = distortion - np.mean(distortion)
    distortion_power = float(np.mean(centered_distortion**2)) + EPS
    target_power = clean_power / (10.0 ** (snr_db / 10.0))
    scaled = centered_distortion * np.sqrt(target_power / distortion_power)
    noisy = clean + scaled
    achieved = 10.0 * np.log10(
        clean_power / (float(np.mean((noisy - clean) ** 2)) + EPS)
    )
    return noisy.astype(np.float32), scaled.astype(np.float32), float(achieved)


def build_imu_conditioned_noisy(
    clean: np.ndarray,
    normalized_imu_features: np.ndarray,
    motion_envelope: np.ndarray,
    fs: int,
    hop_length: int,
    snr_db: float,
    mode: str,
    rng: np.random.Generator,
) -> IMUSynthesisResult:
    if mode not in SYNTHESIS_MODES:
        raise ValueError(f"Unsupported M14 synthesis mode: {mode}")
    clean = as_float_1d(clean)
    envelope = np.asarray(motion_envelope, dtype=np.float32)
    if envelope.ndim != 1 or envelope.size == 0:
        raise ValueError("motion_envelope must be a non-empty 1D array.")

    if mode in {"motion_clean", "clean_identity"}:
        return IMUSynthesisResult(
            noisy_pcg=clean.copy(),
            noise_signal=np.zeros_like(clean),
            snr_db=100.0,
            achieved_snr_db=100.0,
            sqi_target=1.0,
            components="identity",
        )

    n_samples = len(clean)
    envelope_samples = _envelope_to_samples(envelope, n_samples, hop_length)
    components: list[str] = []
    distortion = np.zeros(n_samples, dtype=np.float32)

    if mode in {"motion_artifact", "combined"}:
        low_motion = _band_limited_noise(
            n_samples,
            fs,
            rng,
            float(rng.uniform(4.0, 15.0)),
            float(rng.uniform(55.0, 120.0)),
        )
        friction = _band_limited_noise(
            n_samples,
            fs,
            rng,
            float(rng.uniform(25.0, 70.0)),
            float(rng.uniform(250.0, 650.0)),
        )
        motion_gate = np.power(np.clip(envelope_samples, 0.0, 1.0), 0.7)
        distortion += motion_gate * (0.70 * low_motion + 0.30 * friction)
        impacts = _impact_artifacts(
            normalized_imu_features,
            n_samples,
            fs,
            hop_length,
            rng,
        )
        distortion += impacts
        contact_depth = float(rng.uniform(0.10, 0.55))
        contact_gain = np.clip(1.0 - contact_depth * motion_gate, 0.25, 1.0)
        distortion += (contact_gain - 1.0) * clean
        components.extend(("imu_motion", "imu_friction", "imu_impacts", "contact_gain"))

    if mode in {"independent_artifact", "combined"}:
        independent = _band_limited_noise(
            n_samples,
            fs,
            rng,
            float(rng.uniform(20.0, 80.0)),
            float(rng.uniform(300.0, 750.0)),
        )
        independent_envelope = signal.savgol_filter(
            rng.uniform(0.3, 1.0, size=17),
            window_length=7,
            polyorder=2,
        )
        independent_samples = np.interp(
            np.linspace(0, 1, n_samples),
            np.linspace(0, 1, len(independent_envelope)),
            independent_envelope,
        )
        weight = 0.45 if mode == "combined" else 1.0
        distortion += (weight * independent * independent_samples).astype(np.float32)
        components.append("independent_band_noise")

    if float(np.std(distortion)) < 1e-7:
        distortion = 0.01 * _band_limited_noise(n_samples, fs, rng, 10.0, 100.0)
        components.append("minimum_motion_carrier")

    noisy, scaled_distortion, achieved = _scale_distortion_at_snr(
        clean,
        distortion,
        snr_db,
    )
    return IMUSynthesisResult(
        noisy_pcg=noisy,
        noise_signal=scaled_distortion,
        snr_db=float(snr_db),
        achieved_snr_db=achieved,
        sqi_target=sqi_target_from_snr(float(snr_db)),
        components="+".join(components),
    )
