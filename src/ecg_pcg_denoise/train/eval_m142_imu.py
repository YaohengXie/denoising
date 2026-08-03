from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from ecg_pcg_denoise.models import DenoisingModel
from ecg_pcg_denoise.train.m14_imu_dataset import M14IMUDataset
from ecg_pcg_denoise.train.m142_imu_runtime import (
    build_m142_targets,
    prepare_imu_aux,
    reforward_m142_auxiliary,
)
from ecg_pcg_denoise.train.torch_audio import reconstruct_from_mag, stft_waveform
from ecg_pcg_denoise.train.train_denoise import (
    beat_to_frames,
    choose_device,
    s1s2_to_frames,
    stft_config,
)
from ecg_pcg_denoise.utils.config import get_nested, load_config, require_nested
from ecg_pcg_denoise.utils.files import ensure_dir
from ecg_pcg_denoise.utils.metrics import evaluate_pair


METRIC_NAMES = (
    "delta_snr",
    "delta_si_sdr",
    "corr_estimate",
    "log_spectral_distance",
)
SUMMARY_MEASURES = (
    *METRIC_NAMES,
    "mask_max_abs_vs_m7",
    "waveform_max_abs_vs_m7",
    "s1s2_max_abs_vs_m7",
    "s1s2_target_mae",
    "sqi_abs_error",
    "sqi_m7_abs_error",
    "sqi_mae_improvement_vs_m7",
    "sqi_delta_vs_m7",
    "sqi_confidence_raw",
    "sqi_confidence_calibrated",
    "sqi_confidence_brier",
    "motion_prediction_mean",
    "motion_frame_mae",
    "artifact_prediction_mean",
    "artifact_frame_mae",
    "reliability_prediction_mean",
    "reliability_frame_mae",
    "s1s2_confidence_prediction_mean",
    "s1s2_confidence_frame_mae",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_bound_calibration(
    path: Path,
    *,
    checkpoint_path: Path,
    fold: str,
    variant: str | None = None,
) -> dict[str, Any]:
    """Load validation calibration only when it belongs to this model state.

    Reusing a calibration file from another fold or checkpoint changes the
    reported test result without changing the model.  The controlled
    reproduction path therefore treats the provenance fields as mandatory,
    rather than silently accepting a stale file.
    """

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Calibration must be a JSON object: {path}")

    expected_hash = _sha256(checkpoint_path).lower()
    checks = {
        "fit_split": (payload.get("fit_split"), "val"),
        "fold": (payload.get("fold"), fold),
        "checkpoint_sha256": (
            str(payload.get("checkpoint_sha256", "")).lower(),
            expected_hash,
        ),
    }
    if variant is not None:
        checks["variant"] = (payload.get("variant"), variant)

    failures = [
        f"{name}={actual!r} (expected {expected!r})"
        for name, (actual, expected) in checks.items()
        if actual != expected
    ]
    if failures:
        raise RuntimeError(
            f"Calibration provenance mismatch in {path}: "
            + "; ".join(failures)
        )
    return payload


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _finite_mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    finite = array[np.isfinite(array)]
    return float(np.mean(finite)) if finite.size else float("nan")


def _json_number(value: float) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def _clip_probability(values: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=np.float64), 1e-7, 1.0 - 1e-7)


def _apply_platt(
    values: np.ndarray,
    slope: float,
    intercept: float,
) -> np.ndarray:
    probability = _clip_probability(values)
    logits = np.log(probability) - np.log1p(-probability)
    scaled = np.clip(
        max(float(slope), 1e-6) * logits + float(intercept),
        -40.0,
        40.0,
    )
    return 1.0 / (1.0 + np.exp(-scaled))


def _fit_platt(
    values: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, float, float, str]:
    """Fit monotone two-parameter Platt scaling without sklearn."""

    probability = _clip_probability(values)
    target = np.asarray(labels, dtype=np.float64)
    positives = int(np.sum(target == 1))
    negatives = int(np.sum(target == 0))
    if positives < 10 or negatives < 10:
        prior = (positives + 1.0) / (positives + negatives + 2.0)
        intercept = float(np.log(prior) - np.log1p(-prior))
        calibrated = _clip_probability(
            _apply_platt(probability, 1e-6, intercept)
        )
        loss = -float(
            np.mean(
                target * np.log(calibrated)
                + (1.0 - target) * np.log1p(-calibrated)
            )
        )
        return 1e-6, intercept, loss, "intercept_only_not_identifiable"

    logits = np.log(probability) - np.log1p(-probability)
    x = torch.tensor(logits, dtype=torch.float64)
    y = torch.tensor(target, dtype=torch.float64)
    raw_slope = torch.tensor(0.5413, dtype=torch.float64, requires_grad=True)
    intercept = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        (raw_slope, intercept),
        lr=0.5,
        max_iter=100,
        tolerance_grad=1e-10,
        tolerance_change=1e-12,
        line_search_fn="strong_wolfe",
    )

    def closure() -> Tensor:
        optimizer.zero_grad()
        slope_tensor = torch.nn.functional.softplus(raw_slope)
        loss_tensor = torch.nn.functional.binary_cross_entropy_with_logits(
            slope_tensor * x + intercept,
            y,
        )
        loss_tensor = loss_tensor + 1e-4 * (
            slope_tensor.square() + intercept.square()
        )
        loss_tensor.backward()
        return loss_tensor

    optimizer.step(closure)
    slope = float(torch.nn.functional.softplus(raw_slope).detach())
    bias = float(intercept.detach())
    calibrated = _clip_probability(_apply_platt(probability, slope, bias))
    loss = -float(
        np.mean(
            target * np.log(calibrated)
            + (1.0 - target) * np.log1p(-calibrated)
        )
    )
    return slope, bias, loss, "monotone_platt"


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + 1 + stop)
        start = stop
    return ranks


def _auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = _average_ranks(scores)
    rank_sum = float(np.sum(ranks[labels == 1]))
    return (
        rank_sum - positives * (positives + 1) / 2.0
    ) / (positives * negatives)


def _auprc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(np.sum(labels == 1))
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    cumulative_positive = np.cumsum(sorted_labels)
    precision = cumulative_positive / np.arange(1, labels.size + 1)
    return float(np.sum(precision * sorted_labels) / positives)


def _ece(labels: np.ndarray, scores: np.ndarray, bins: int = 10) -> float:
    labels = np.asarray(labels, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    total = max(1, labels.size)
    error = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        selected = (scores >= lower) & (
            (scores <= upper) if index == bins - 1 else (scores < upper)
        )
        if not np.any(selected):
            continue
        error += (
            float(np.sum(selected))
            / total
            * abs(
                float(np.mean(scores[selected]))
                - float(np.mean(labels[selected]))
            )
        )
    return float(error)


def _binary_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=np.int8).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    finite = np.isfinite(scores)
    labels = labels[finite]
    scores = np.clip(scores[finite], 0.0, 1.0)
    predictions = scores >= threshold
    positives = labels == 1
    negatives = ~positives
    tp = int(np.sum(predictions & positives))
    fp = int(np.sum(predictions & negatives))
    fn = int(np.sum((~predictions) & positives))
    tn = int(np.sum((~predictions) & negatives))
    sensitivity = tp / (tp + fn) if tp + fn else float("nan")
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    balanced_accuracy = (
        0.5 * (sensitivity + specificity)
        if np.isfinite(sensitivity) and np.isfinite(specificity)
        else float("nan")
    )
    return {
        "n": int(labels.size),
        "positives": int(np.sum(positives)),
        "negatives": int(np.sum(negatives)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "auroc": _auroc(labels, scores),
        "auprc": _auprc(labels, scores),
        "brier": float(np.mean((scores - labels) ** 2))
        if labels.size
        else float("nan"),
        "ece": _ece(labels, scores) if labels.size else float("nan"),
        "f1": (2.0 * tp / (2 * tp + fp + fn))
        if 2 * tp + fp + fn
        else float("nan"),
        "balanced_accuracy": balanced_accuracy,
    }


def _select_binary_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
) -> tuple[float, dict[str, float | int]]:
    """Select a validation-only threshold by balanced accuracy.

    F1 is the deterministic tie-breaker, followed by the threshold closest to
    0.5. Test data must only load the saved validation result.
    """

    labels = np.asarray(labels, dtype=np.int8).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    finite = np.isfinite(scores)
    labels = labels[finite]
    scores = np.clip(scores[finite], 0.0, 1.0)
    if labels.size == 0 or np.unique(labels).size < 2:
        metrics = _binary_metrics(labels, scores, threshold=0.5)
        return 0.5, metrics
    unique = np.unique(scores)
    if unique.size == 1:
        candidates = np.asarray([0.0, float(unique[0]), 1.0])
    else:
        midpoints = (unique[:-1] + unique[1:]) / 2.0
        candidates = np.unique(
            np.concatenate(
                (
                    np.asarray([0.0, 0.5, 1.0]),
                    unique,
                    midpoints,
                )
            )
        )
    best_threshold = 0.5
    best_metrics = _binary_metrics(labels, scores, threshold=best_threshold)

    def _key(
        threshold: float,
        metrics: dict[str, float | int],
    ) -> tuple[float, float, float]:
        balanced = float(metrics["balanced_accuracy"])
        f1 = float(metrics["f1"])
        return (
            balanced if np.isfinite(balanced) else -1.0,
            f1 if np.isfinite(f1) else -1.0,
            -abs(float(threshold) - 0.5),
        )

    best_key = _key(best_threshold, best_metrics)
    for candidate in candidates:
        threshold = float(candidate)
        metrics = _binary_metrics(labels, scores, threshold=threshold)
        candidate_key = _key(threshold, metrics)
        if candidate_key > best_key:
            best_threshold = threshold
            best_metrics = metrics
            best_key = candidate_key
    return best_threshold, best_metrics


def _load_models(
    config: dict[str, Any],
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[DenoisingModel, DenoisingModel, dict[str, Any], Path]:
    base_path = Path(require_nested(config, "paths.base_m7_checkpoint"))
    base_checkpoint = torch.load(base_path, map_location=device, weights_only=True)
    m7 = DenoisingModel.from_config(base_checkpoint["config"])
    if not isinstance(m7, DenoisingModel):
        raise TypeError("The M7 checkpoint is not a U-Net DenoisingModel.")
    m7 = m7.to(device)
    m7.load_state_dict(base_checkpoint["model"], strict=True)
    m7.eval()

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    m142 = DenoisingModel.from_config(config)
    if not isinstance(m142, DenoisingModel):
        raise TypeError("M14.2 requires a U-Net DenoisingModel.")
    m142 = m142.to(device)
    if checkpoint.get("artifact_type") == "m143_v2_imu_aux_adapter":
        expected_fold = str(require_nested(config, "m14_imu.fold"))
        if checkpoint.get("fold") != expected_fold:
            raise ValueError(
                f"Adapter fold {checkpoint.get('fold')!r} does not match "
                f"configuration fold {expected_fold!r}."
            )
        base_hash = _sha256(base_path)
        if checkpoint.get("base_checkpoint_sha256") != base_hash:
            raise ValueError("The IMU adapter is not bound to the supplied M7 checkpoint.")
        incompatible = m142.load_state_dict(m7.state_dict(), strict=False)
        expected_missing = {
            key for key in m142.state_dict() if key.startswith("imu_aux_adapter.")
        }
        if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                "M7-to-M14.3 state composition produced unexpected state keys: "
                f"missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}"
            )
        if m142.imu_aux_adapter is None:
            raise RuntimeError("M14.3 configuration did not create an IMU auxiliary adapter.")
        m142.imu_aux_adapter.load_state_dict(checkpoint["adapter"], strict=True)
    elif checkpoint.get("artifact_type") == "m143_v2_full_training_checkpoint":
        expected_fold = str(require_nested(config, "m14_imu.fold"))
        if checkpoint.get("fold") != expected_fold:
            raise ValueError(
                f"Training checkpoint fold {checkpoint.get('fold')!r} does not match "
                f"configuration fold {expected_fold!r}."
            )
        if checkpoint.get("base_checkpoint_sha256") != _sha256(base_path):
            raise ValueError(
                "The full M14.3 training checkpoint is not bound to the supplied "
                "M7 checkpoint."
            )
        m142.load_state_dict(checkpoint["model"], strict=True)
    else:
        raise ValueError(
            "Unsupported M14.3 checkpoint format; expected a published factorised "
            "adapter or a bound full training checkpoint."
        )
    m142.eval()
    if not bool(getattr(m142, "use_imu_aux", False)):
        raise ValueError("The supplied model does not enable model.use_imu_aux.")
    return m7, m142, checkpoint, base_path


def _prepare_variant_imu(
    batch: dict[str, Any],
    device: torch.device,
    variant: str,
    shift_frames: int,
) -> tuple[Tensor, Tensor]:
    mode = variant.rsplit("_", 1)[-1]
    return prepare_imu_aux(
        batch,
        device,
        mode,
        shift_frames=shift_frames if mode == "shift" else None,
    )


def _filter_dataset(
    dataset: M14IMUDataset,
    subjects: set[str] | None,
    max_windows: int | None,
) -> None:
    if dataset.fixed_rows is None:
        raise ValueError("M14.2 evaluation requires a fixed val/test split.")
    rows = dataset.fixed_rows
    if subjects is not None:
        rows = [
            row
            for row in rows
            if str(row["bsslab_subject_id"]) in subjects
        ]
    if max_windows is not None:
        rows = rows[: int(max_windows)]
    if not rows:
        raise ValueError("No evaluation windows remain after filtering.")
    dataset.fixed_rows = rows


def _parse_subjects(values: list[str] | None) -> set[str] | None:
    if values is None:
        return None
    parsed: set[str] = set()
    for value in values:
        parsed.update(part.strip() for part in value.split(",") if part.strip())
    return parsed or None


def _group_memberships(row: dict[str, Any]) -> list[tuple[str, str]]:
    mode = str(row["artifact_mode"])
    memberships = [
        ("aggregate", "all"),
        ("artifact_mode", mode),
        ("snr_db", str(row["snr_db"])),
        ("imu_condition", str(row["imu_condition"])),
        ("bsslab_subject", str(row["bsslab_subject_id"])),
    ]
    if mode in {"motion_artifact", "combined"}:
        relevance = "imu_relevant"
    elif mode in {"motion_clean", "clean_identity"}:
        relevance = "identity"
    else:
        relevance = "independent_artifact"
    memberships.append(("relevance", relevance))
    return memberships


def _calibration_rows(
    rows: list[dict[str, Any]],
    score_field: str,
    score_kind: str,
    bins: int = 10,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    auxiliary_variants = sorted(
        {str(row["variant"]) for row in rows if str(row["variant"]) != "m7"}
    )
    for variant in auxiliary_variants:
        selected_rows = [row for row in rows if row["variant"] == variant]
        scores = np.asarray(
            [float(row[score_field]) for row in selected_rows],
            dtype=np.float64,
        )
        labels = np.asarray(
            [int(row["sqi_correct_label"]) for row in selected_rows],
            dtype=np.int8,
        )
        for index in range(bins):
            lower = index / bins
            upper = (index + 1) / bins
            selected = (scores >= lower) & (
                (scores <= upper) if index == bins - 1 else (scores < upper)
            )
            result.append(
                {
                    "variant": variant,
                    "score_kind": score_kind,
                    "bin_index": index,
                    "bin_lower": lower,
                    "bin_upper": upper,
                    "n": int(np.sum(selected)),
                    "mean_confidence": float(np.mean(scores[selected]))
                    if np.any(selected)
                    else float("nan"),
                    "empirical_accuracy": float(np.mean(labels[selected]))
                    if np.any(selected)
                    else float("nan"),
                    "calibration_gap": (
                        float(np.mean(scores[selected]))
                        - float(np.mean(labels[selected]))
                    )
                    if np.any(selected)
                    else float("nan"),
                }
            )
    return result


def _selective_risk_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    auxiliary_variants = sorted(
        {str(row["variant"]) for row in rows if str(row["variant"]) != "m7"}
    )
    for variant in auxiliary_variants:
        selected_rows = [row for row in rows if row["variant"] == variant]
        selected_rows.sort(
            key=lambda row: float(row["sqi_confidence_calibrated"]),
            reverse=True,
        )
        total = len(selected_rows)
        for requested_coverage in np.linspace(0.1, 1.0, 10):
            count = max(1, int(math.ceil(total * float(requested_coverage))))
            retained = selected_rows[:count]
            risk = _finite_mean(float(row["sqi_abs_error"]) for row in retained)
            m7_risk = _finite_mean(
                float(row["sqi_m7_abs_error"]) for row in retained
            )
            result.append(
                {
                    "variant": variant,
                    "requested_coverage": float(requested_coverage),
                    "actual_coverage": count / total,
                    "n_selected": count,
                    "sqi_mae": risk,
                    "m7_sqi_mae_on_same_windows": m7_risk,
                    "sqi_mae_improvement_vs_m7": m7_risk - risk,
                    "mean_confidence": _finite_mean(
                        float(row["sqi_confidence_calibrated"])
                        for row in retained
                    ),
                }
            )
    return result


def evaluate_m142(
    config_path: str | Path,
    checkpoint_path: str | Path | None = None,
    split: str = "test",
    output_dir: str | Path | None = None,
    bsslab_subjects: set[str] | None = None,
    max_windows: int | None = None,
    imu_shift_seconds: float | None = None,
) -> dict[str, Any]:
    if split not in {"val", "test"}:
        raise ValueError(f"Unsupported split: {split}")
    config_path = Path(config_path)
    config = load_config(config_path)
    variant_prefix = str(
        get_nested(config, "evaluation.variant_prefix", "m142")
    ).strip()
    if not variant_prefix or "_" in variant_prefix:
        raise ValueError(
            "evaluation.variant_prefix must be a non-empty token without '_'."
        )
    correct_variant = f"{variant_prefix}_correct"
    auxiliary_variants = (
        correct_variant,
        f"{variant_prefix}_missing",
        f"{variant_prefix}_shuffle",
        f"{variant_prefix}_shift",
    )
    variants = ("m7", *auxiliary_variants)
    experiment_label = str(
        get_nested(
            config,
            "evaluation.experiment_label",
            "M14.2 IMU auxiliary SQI/confidence evaluation",
        )
    )
    device = choose_device(config)
    training_root = Path(require_nested(config, "paths.output_dir"))
    checkpoint = (
        Path(checkpoint_path)
        if checkpoint_path is not None
        else training_root / "checkpoints" / "best.pt"
    )
    eval_dir = ensure_dir(
        Path(output_dir) if output_dir is not None else training_root / f"eval_{split}"
    )
    m7, m142, m142_checkpoint, base_path = _load_models(
        config,
        checkpoint,
        device,
    )
    fold = str(require_nested(config, "m14_imu.fold"))
    dataset = M14IMUDataset.from_config(
        require_nested(config, "paths.data_config"),
        split,
        fold,
    )
    _filter_dataset(dataset, bsslab_subjects, max_windows)
    loader = DataLoader(
        dataset,
        batch_size=int(require_nested(config, "training.batch_size")),
        shuffle=False,
        num_workers=int(get_nested(config, "training.num_workers", 0)),
        pin_memory=device.type == "cuda",
    )

    n_fft, hop, win = stft_config(config)
    fs_model = int(require_nested(config, "data.fs_model"))
    requested_shift_sec = float(
        imu_shift_seconds
        if imu_shift_seconds is not None
        else get_nested(config, "evaluation.imu_shift_seconds", 0.25)
    )
    if requested_shift_sec <= 0.0:
        raise ValueError("IMU shift seconds must be positive.")
    shift_frames = max(1, int(round(requested_shift_sec * fs_model / hop)))
    actual_shift_sec = shift_frames * hop / fs_model
    confidence_tolerance = float(
        get_nested(config, "training.loss.sqi_confidence_tolerance", 0.10)
    )
    rows: list[dict[str, Any]] = []
    frame_collectors: dict[
        tuple[str, str], dict[str, list[np.ndarray]]
    ] = defaultdict(lambda: {"labels": [], "scores": []})
    processed = 0
    singleton_shuffle_batches = 0

    with torch.inference_mode():
        for batch in loader:
            noisy = batch["noisy_pcg"].to(device, non_blocking=True).float()
            clean = batch["clean_pcg"].to(device, non_blocking=True).float()
            noise = batch["noise_signal"].to(device, non_blocking=True).float()
            beat = batch["ecg_beat"].to(device, non_blocking=True).float()
            noisy_complex = stft_waveform(noisy, n_fft, hop, win)
            noisy_mag = noisy_complex.abs()
            clean_mag = stft_waveform(clean, n_fft, hop, win).abs()
            noise_mag = stft_waveform(noise, n_fft, hop, win).abs()
            beat_frames = beat_to_frames(beat, noisy_mag.shape[-1])
            m7_output = m7(
                pcg_stft=noisy_mag,
                ecg_beat=beat_frames,
                imu_feat=None,
                modality_mask={"pcg": 1, "ecg": 1, "imu": 0},
            )
            m7_estimate = reconstruct_from_mag(
                noisy_complex,
                m7_output["mask"] * noisy_mag,
                noisy.shape[-1],
                n_fft,
                hop,
                win,
            )
            outputs: dict[str, dict[str, Tensor]] = {"m7": m7_output}
            estimates: dict[str, Tensor] = {"m7": m7_estimate}
            if noisy.shape[0] == 1:
                singleton_shuffle_batches += 1
            correct_imu, correct_present = _prepare_variant_imu(
                batch,
                device,
                correct_variant,
                shift_frames,
            )
            correct_output = m142(
                pcg_stft=noisy_mag,
                ecg_beat=beat_frames,
                imu_feat=correct_imu,
                modality_mask={
                    "pcg": 1,
                    "ecg": 1,
                    "imu": correct_present,
                },
                return_aux_context=True,
            )
            outputs[correct_variant] = correct_output
            m142_estimate = reconstruct_from_mag(
                noisy_complex,
                correct_output["mask"] * noisy_mag,
                noisy.shape[-1],
                n_fft,
                hop,
                win,
            )
            estimates[correct_variant] = m142_estimate
            for variant in auxiliary_variants[1:]:
                variant_batch = batch
                if variant.endswith("_shuffle"):
                    # A LOSO test batch contains one held-out participant, so
                    # participant IDs alone cannot create a mismatch. Unique
                    # record/window keys force a different real IMU window
                    # while preserving the held-out participant distribution.
                    variant_batch = dict(batch)
                    variant_batch["imu_subject_id"] = [
                        f"{record}@{float(start):.6f}"
                        for record, start in zip(
                            batch["imu_record_id"],
                            batch["imu_window_start_sec"],
                            strict=True,
                        )
                    ]
                output = reforward_m142_auxiliary(
                    m142,
                    correct_output,
                    variant_batch,
                    device,
                    variant.rsplit("_", 1)[-1],
                    shift_frames=(
                        shift_frames if variant.endswith("_shift") else None
                    ),
                )
                outputs[variant] = output
                estimates[variant] = m142_estimate

            target_tensors = {
                "clean_mag": clean_mag,
                "noise_mag": noise_mag,
            }
            targets = build_m142_targets(
                outputs[correct_variant],
                target_tensors,
                batch,
                config,
            )
            s1s2_target = s1s2_to_frames(
                batch["s1s2_weak"].to(device, non_blocking=True),
                m7_output["s1s2_prob"].shape[-1],
            ).float()
            batch_size = int(noisy.shape[0])
            m7_sqi = m7_output["sqi_score"].detach().float()
            m7_mask = m7_output["mask"].detach().float()
            m7_s1s2 = m7_output["s1s2_prob"].detach().float()

            for variant in variants:
                output = outputs[variant]
                estimate = estimates[variant].detach().float()
                mask = output["mask"].detach().float()
                s1s2 = output["s1s2_prob"].detach().float()
                sqi = output["sqi_score"].detach().float()
                is_m142 = variant != "m7"
                if is_m142:
                    motion_prediction = (
                        output["imu_motion_probability"].detach().float()
                    )
                    artifact_prediction = (
                        output["imu_artifact_probability"].detach().float()
                    )
                    reliability_prediction = (
                        output["imu_reliability"].detach().float()
                    )
                    s1s2_confidence_prediction = (
                        output["s1s2_confidence"].detach().float()
                    )
                    confidence = output["sqi_confidence"].detach().float()
                    for task, prediction, target in (
                        (
                            "motion_frame_ge_0.5",
                            motion_prediction,
                            targets["motion"],
                        ),
                        (
                            "artifact_frame_ge_0.5",
                            artifact_prediction,
                            targets["coupled_artifact"],
                        ),
                        (
                            "reliability_frame_ge_0.5",
                            reliability_prediction,
                            targets["reliability"],
                        ),
                    ):
                        frame_collectors[(variant, task)]["labels"].append(
                            (target >= 0.5).detach().cpu().numpy().reshape(-1)
                        )
                        frame_collectors[(variant, task)]["scores"].append(
                            prediction.detach().cpu().numpy().reshape(-1)
                        )
                else:
                    motion_prediction = artifact_prediction = None
                    reliability_prediction = s1s2_confidence_prediction = None
                    confidence = None

                for index in range(batch_size):
                    noisy_array = noisy[index].detach().cpu().numpy()
                    clean_array = clean[index].detach().cpu().numpy()
                    estimate_array = estimate[index].detach().cpu().numpy()
                    metrics = evaluate_pair(
                        noisy_array,
                        estimate_array,
                        clean_array,
                        int(batch["fs"][index]),
                    )
                    sqi_target = float(targets["sqi"][index])
                    sqi_score = float(sqi[index])
                    m7_sqi_score = float(m7_sqi[index])
                    sqi_error = abs(sqi_score - sqi_target)
                    m7_sqi_error = abs(m7_sqi_score - sqi_target)
                    if is_m142:
                        motion_target = targets["motion"][index]
                        artifact_target = targets["coupled_artifact"][index]
                        reliability_target = targets["reliability"][index]
                        s1s2_confidence_target = targets[
                            "s1s2_confidence"
                        ][index]
                        motion_pred = motion_prediction[index]
                        artifact_pred = artifact_prediction[index]
                        reliability_pred = reliability_prediction[index]
                        s1s2_confidence_pred = s1s2_confidence_prediction[index]
                        confidence_value = float(confidence[index])
                    else:
                        motion_target = targets["motion"][index]
                        artifact_target = targets["coupled_artifact"][index]
                        reliability_target = targets["reliability"][index]
                        s1s2_confidence_target = targets[
                            "s1s2_confidence"
                        ][index]
                        motion_pred = artifact_pred = reliability_pred = None
                        s1s2_confidence_pred = None
                        confidence_value = float("nan")
                    rows.append(
                        {
                            "sample_index": processed + index,
                            "split": split,
                            "fold": fold,
                            "variant": variant,
                            "artifact_mode": str(batch["artifact_mode"][index]),
                            "artifact_window_label": int(
                                str(batch["artifact_mode"][index])
                                in {"motion_artifact", "combined"}
                            ),
                            "snr_db": float(batch["snr_db"][index]),
                            "achieved_snr_db": float(
                                batch["achieved_snr_db"][index]
                            ),
                            "motion_score": float(batch["motion_score"][index]),
                            "bsslab_subject_id": str(
                                batch["bsslab_subject_id"][index]
                            ),
                            "bsslab_record_id": str(
                                batch["bsslab_record_id"][index]
                            ),
                            "bsslab_window_start": int(
                                batch["bsslab_window_start"][index]
                            ),
                            "imu_subject_id": str(batch["imu_subject_id"][index]),
                            "imu_condition": str(batch["imu_condition"][index]),
                            "imu_record_id": str(batch["imu_record_id"][index]),
                            "imu_window_start_sec": float(
                                batch["imu_window_start_sec"][index]
                            ),
                            "fs": int(batch["fs"][index]),
                            "mask_max_abs_vs_m7": float(
                                torch.max(torch.abs(mask[index] - m7_mask[index]))
                            ),
                            "waveform_max_abs_vs_m7": float(
                                torch.max(
                                    torch.abs(
                                        estimate[index] - m7_estimate[index]
                                    )
                                )
                            ),
                            "s1s2_max_abs_vs_m7": float(
                                torch.max(
                                    torch.abs(s1s2[index] - m7_s1s2[index])
                                )
                            ),
                            "s1s2_target_mae": float(
                                torch.mean(
                                    torch.abs(s1s2[index] - s1s2_target[index])
                                )
                            ),
                            "sqi_target": sqi_target,
                            "sqi_variant_score": sqi_score,
                            "sqi_m7_score": m7_sqi_score,
                            "sqi_m142_score": (
                                sqi_score if is_m142 else float("nan")
                            ),
                            "sqi_abs_error": sqi_error,
                            "sqi_m7_abs_error": m7_sqi_error,
                            "sqi_mae_improvement_vs_m7": (
                                m7_sqi_error - sqi_error
                            ),
                            "sqi_delta_vs_m7": sqi_score - m7_sqi_score,
                            "sqi_confidence_raw": confidence_value,
                            "sqi_confidence_calibrated": float("nan"),
                            "sqi_correct_label": int(
                                sqi_error <= confidence_tolerance
                            ),
                            "sqi_confidence_brier": float("nan"),
                            "motion_target_mean": float(
                                torch.mean(motion_target)
                            ),
                            "motion_prediction_mean": (
                                float(torch.mean(motion_pred))
                                if motion_pred is not None
                                else float("nan")
                            ),
                            "motion_frame_mae": (
                                float(
                                    torch.mean(
                                        torch.abs(motion_pred - motion_target)
                                    )
                                )
                                if motion_pred is not None
                                else float("nan")
                            ),
                            "artifact_target_mean": float(
                                torch.mean(artifact_target)
                            ),
                            "artifact_prediction_mean": (
                                float(torch.mean(artifact_pred))
                                if artifact_pred is not None
                                else float("nan")
                            ),
                            "artifact_frame_mae": (
                                float(
                                    torch.mean(
                                        torch.abs(artifact_pred - artifact_target)
                                    )
                                )
                                if artifact_pred is not None
                                else float("nan")
                            ),
                            "reliability_target_mean": float(
                                torch.mean(reliability_target)
                            ),
                            "reliability_prediction_mean": (
                                float(torch.mean(reliability_pred))
                                if reliability_pred is not None
                                else float("nan")
                            ),
                            "reliability_frame_mae": (
                                float(
                                    torch.mean(
                                        torch.abs(
                                            reliability_pred
                                            - reliability_target
                                        )
                                    )
                                )
                                if reliability_pred is not None
                                else float("nan")
                            ),
                            "s1s2_confidence_target_mean": float(
                                torch.mean(s1s2_confidence_target)
                            ),
                            "s1s2_confidence_prediction_mean": (
                                float(torch.mean(s1s2_confidence_pred))
                                if s1s2_confidence_pred is not None
                                else float("nan")
                            ),
                            "s1s2_confidence_frame_mae": (
                                float(
                                    torch.mean(
                                        torch.abs(
                                            s1s2_confidence_pred
                                            - s1s2_confidence_target
                                        )
                                    )
                                )
                                if s1s2_confidence_pred is not None
                                else float("nan")
                            ),
                            **metrics,
                        }
                    )
            processed += batch_size
            if processed % 320 == 0 or processed == len(dataset):
                print(f"evaluated_windows={processed}/{len(dataset)}")

    correct_rows = [row for row in rows if row["variant"] == correct_variant]
    artifact_labels_correct = np.asarray(
        [int(row["artifact_window_label"]) for row in correct_rows],
        dtype=np.int8,
    )
    artifact_scores_correct = np.asarray(
        [float(row["artifact_prediction_mean"]) for row in correct_rows],
        dtype=np.float64,
    )
    artifact_threshold_enabled = bool(
        get_nested(
            config,
            "evaluation.calibrate_artifact_threshold",
            False,
        )
    )
    artifact_threshold_path = (
        training_root / "artifact_threshold_calibration.json"
    )
    artifact_threshold_status = "disabled_fixed_0.5"
    artifact_threshold = 0.5
    artifact_threshold_validation_metrics: dict[str, float | int] = (
        _binary_metrics(
            artifact_labels_correct,
            artifact_scores_correct,
            threshold=artifact_threshold,
        )
    )
    if artifact_threshold_enabled and split == "val":
        (
            artifact_threshold,
            artifact_threshold_validation_metrics,
        ) = _select_binary_threshold(
            artifact_labels_correct,
            artifact_scores_correct,
        )
        artifact_threshold_status = "fit_on_validation"
        artifact_threshold_payload = {
            "method": "max_balanced_accuracy_f1_tiebreak",
            "fit_split": split,
            "fold": fold,
            "variant": correct_variant,
            "threshold": artifact_threshold,
            "n": int(artifact_labels_correct.size),
            "positives": int(np.sum(artifact_labels_correct)),
            "metrics": artifact_threshold_validation_metrics,
            "checkpoint_sha256": _sha256(checkpoint),
        }
        artifact_threshold_path.write_text(
            json.dumps(artifact_threshold_payload, indent=2),
            encoding="utf-8",
        )
        (eval_dir / "artifact_threshold_calibration.json").write_text(
            json.dumps(artifact_threshold_payload, indent=2),
            encoding="utf-8",
        )
    elif artifact_threshold_enabled and artifact_threshold_path.exists():
        artifact_threshold_payload = _load_bound_calibration(
            artifact_threshold_path,
            checkpoint_path=checkpoint,
            fold=fold,
            variant=correct_variant,
        )
        artifact_threshold = float(artifact_threshold_payload["threshold"])
        artifact_threshold_status = "loaded_validation_threshold"
        artifact_threshold_validation_metrics = dict(
            artifact_threshold_payload.get("metrics", {})
        )
    elif artifact_threshold_enabled:
        artifact_threshold_status = "missing_validation_threshold_fixed_0.5"

    raw_confidence = np.asarray(
        [float(row["sqi_confidence_raw"]) for row in correct_rows],
        dtype=np.float64,
    )
    correct_labels = np.asarray(
        [int(row["sqi_correct_label"]) for row in correct_rows],
        dtype=np.int8,
    )
    calibration_path = training_root / "confidence_calibration.json"
    calibration_status: str
    calibration_loss = float("nan")
    if split == "val":
        slope, intercept, calibration_loss, fit_method = _fit_platt(
            raw_confidence,
            correct_labels,
        )
        calibration_status = "fit_on_validation"
        calibration_payload = {
            "slope": slope,
            "intercept": intercept,
            "method": fit_method,
            "fit_split": split,
            "fold": fold,
            "n": int(correct_labels.size),
            "positives": int(np.sum(correct_labels)),
            "negative_log_likelihood": calibration_loss,
            "checkpoint_sha256": _sha256(checkpoint),
        }
        calibration_path.write_text(
            json.dumps(calibration_payload, indent=2),
            encoding="utf-8",
        )
        (eval_dir / "confidence_calibration.json").write_text(
            json.dumps(calibration_payload, indent=2),
            encoding="utf-8",
        )
    elif calibration_path.exists():
        payload = _load_bound_calibration(
            calibration_path,
            checkpoint_path=checkpoint,
            fold=fold,
        )
        slope = float(payload["slope"])
        intercept = float(payload["intercept"])
        fit_method = str(payload.get("method", "monotone_platt"))
        calibration_status = "loaded_validation_calibration"
    else:
        slope = 1.0
        intercept = 0.0
        fit_method = "identity_missing_validation_calibration"
        calibration_status = "missing_validation_calibration_identity_used"

    for row in rows:
        raw = float(row["sqi_confidence_raw"])
        if not np.isfinite(raw):
            continue
        calibrated = float(
            _apply_platt(np.asarray([raw]), slope, intercept)[0]
        )
        row["sqi_confidence_calibrated"] = calibrated
        row["sqi_confidence_brier"] = (
            calibrated - int(row["sqi_correct_label"])
        ) ** 2

    summary_rows: list[dict[str, Any]] = []
    memberships = sorted(
        {
            membership
            for row in rows
            for membership in _group_memberships(row)
        }
    )
    for variant in variants:
        variant_rows = [row for row in rows if row["variant"] == variant]
        for group_type, group_value in memberships:
            selected = [
                row
                for row in variant_rows
                if (group_type, group_value) in _group_memberships(row)
            ]
            if not selected:
                continue
            summary_rows.append(
                {
                    "variant": variant,
                    "group_type": group_type,
                    "group_value": group_value,
                    "n": len(selected),
                    **{
                        measure: _finite_mean(
                            float(row[measure]) for row in selected
                        )
                        for measure in SUMMARY_MEASURES
                    },
                }
            )

    classification_rows: list[dict[str, Any]] = []
    artifact_target_source = str(
        get_nested(
            config,
            "training.targets.coupled_artifact_source",
            "imu_envelope",
        )
    )
    artifact_frame_definition = {
        "imu_envelope": "motion-coupled IMU artifact target >= 0.5",
        "local_noise_fraction": (
            "frame-local noise-energy fraction for artifact-mode windows >= 0.5"
        ),
    }.get(
        artifact_target_source,
        f"{artifact_target_source} artifact target >= 0.5",
    )
    task_definitions = {
        "motion_frame_ge_0.5": "motion envelope target >= 0.5",
        "artifact_frame_ge_0.5": artifact_frame_definition,
        "reliability_frame_ge_0.5": "IMU valid/present target >= 0.5",
    }
    for (variant, task), values in sorted(frame_collectors.items()):
        labels = np.concatenate(values["labels"]).astype(np.int8)
        scores = np.concatenate(values["scores"]).astype(np.float64)
        classification_rows.append(
            {
                "variant": variant,
                "task": task,
                "score_kind": "raw_probability",
                "target_definition": task_definitions[task],
                "decision_threshold": 0.5,
                **_binary_metrics(labels, scores),
            }
        )
    for variant in auxiliary_variants:
        selected = [row for row in rows if row["variant"] == variant]
        artifact_labels = np.asarray(
            [int(row["artifact_window_label"]) for row in selected],
            dtype=np.int8,
        )
        artifact_scores = np.asarray(
            [float(row["artifact_prediction_mean"]) for row in selected],
            dtype=np.float64,
        )
        classification_rows.append(
            {
                "variant": variant,
                "task": "artifact_window",
                "score_kind": "raw_probability",
                "target_definition": (
                    "artifact_mode is motion_artifact or combined"
                ),
                "decision_threshold": 0.5,
                **_binary_metrics(artifact_labels, artifact_scores),
            }
        )
        if artifact_threshold_enabled:
            classification_rows.append(
                {
                    "variant": variant,
                    "task": "artifact_window",
                    "score_kind": "validation_selected_threshold",
                    "target_definition": (
                        "artifact_mode is motion_artifact or combined"
                    ),
                    "decision_threshold": artifact_threshold,
                    **_binary_metrics(
                        artifact_labels,
                        artifact_scores,
                        threshold=artifact_threshold,
                    ),
                }
            )
        sqi_labels = np.asarray(
            [int(row["sqi_correct_label"]) for row in selected],
            dtype=np.int8,
        )
        for field, score_kind in (
            ("sqi_confidence_raw", "raw_probability"),
            ("sqi_confidence_calibrated", "monotone_platt"),
        ):
            sqi_scores = np.asarray(
                [float(row[field]) for row in selected],
                dtype=np.float64,
            )
            classification_rows.append(
                {
                    "variant": variant,
                    "task": "sqi_correctness",
                    "score_kind": score_kind,
                    "target_definition": (
                        f"|SQI prediction - target| <= {confidence_tolerance:g}"
                    ),
                    "decision_threshold": 0.5,
                    **_binary_metrics(sqi_labels, sqi_scores),
                }
            )

    calibration_rows = _calibration_rows(
        rows,
        "sqi_confidence_raw",
        "raw_probability",
    )
    calibration_rows.extend(
        _calibration_rows(
            rows,
            "sqi_confidence_calibrated",
            "monotone_platt",
        )
    )
    selective_rows = _selective_risk_rows(rows)

    window_path = eval_dir / "window_metrics.csv"
    summary_path = eval_dir / "summary_by_group.csv"
    classification_path = eval_dir / "classification_metrics.csv"
    calibration_bins_path = eval_dir / "calibration_bins.csv"
    selective_path = eval_dir / "selective_risk.csv"
    _write_csv(window_path, rows)
    _write_csv(summary_path, summary_rows)
    _write_csv(classification_path, classification_rows)
    _write_csv(calibration_bins_path, calibration_rows)
    _write_csv(selective_path, selective_rows)

    safety = {
        name: max(float(row[name]) for row in rows)
        for name in (
            "mask_max_abs_vs_m7",
            "waveform_max_abs_vs_m7",
            "s1s2_max_abs_vs_m7",
        )
    }
    audit = {
        "schema_version": 1,
        "experiment": experiment_label,
        "config": str(config_path.resolve()),
        "split": split,
        "fold": fold,
        "device": str(device),
        "evaluation_amp": False,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_epoch": m142_checkpoint.get("epoch"),
        "checkpoint_val_loss": m142_checkpoint.get("val_loss"),
        "base_m7_checkpoint": str(base_path.resolve()),
        "base_m7_checkpoint_sha256": _sha256(base_path),
        "windows": len(dataset),
        "rows": len(rows),
        "bsslab_subject_filter": (
            sorted(bsslab_subjects) if bsslab_subjects else None
        ),
        "max_windows": max_windows,
        "variants": list(variants),
        "counterfactuals": {
            "shuffle": (
                "deterministic cross-window reassignment; training prefers a "
                "different participant, while LOSO test uses a different "
                "record/window from the same held-out participant"
            ),
            "singleton_shuffle_batches": singleton_shuffle_batches,
            "shift": "right shift with zero padding; no circular wrap",
            "requested_shift_seconds": requested_shift_sec,
            "shift_frames": shift_frames,
            "actual_shift_seconds": actual_shift_sec,
        },
        "confidence_calibration": {
            "status": calibration_status,
            "method": fit_method,
            "slope": slope,
            "intercept": intercept,
            "validation_negative_log_likelihood": _json_number(calibration_loss),
            "path": str(calibration_path.resolve()),
            "correctness_tolerance": confidence_tolerance,
        },
        "artifact_threshold_calibration": {
            "enabled": artifact_threshold_enabled,
            "status": artifact_threshold_status,
            "method": "max_balanced_accuracy_f1_tiebreak",
            "threshold": artifact_threshold,
            "fit_split": "val" if artifact_threshold_enabled else None,
            "validation_metrics": artifact_threshold_validation_metrics,
            "path": str(artifact_threshold_path.resolve()),
        },
        "strict_m7_identity_max_abs": safety,
        "strict_m7_identity_pass": all(value == 0.0 for value in safety.values()),
        "outputs": {
            "window_metrics": str(window_path.resolve()),
            "summary_by_group": str(summary_path.resolve()),
            "classification_metrics": str(classification_path.resolve()),
            "calibration_bins": str(calibration_bins_path.resolve()),
            "selective_risk": str(selective_path.resolve()),
        },
    }
    audit_path = eval_dir / "evaluation_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"Wrote {experiment_label} to {eval_dir}")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate auxiliary-only IMU SQI/confidence heads "
            "(the final V2 YAML selects M14.3-v2)."
        )
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Exact evaluation output directory (defaults to output/eval_SPLIT).",
    )
    parser.add_argument(
        "--bsslab-subjects",
        nargs="+",
        default=None,
        help="Optional subject IDs, space- or comma-separated.",
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help="Optional deterministic prefix limit, intended for smoke tests.",
    )
    parser.add_argument(
        "--imu-shift-seconds",
        type=float,
        default=None,
        help="Optional non-circular IMU shift override for lag sweeps.",
    )
    args = parser.parse_args()
    evaluate_m142(
        args.config,
        checkpoint_path=args.checkpoint,
        split=args.split,
        output_dir=args.output_dir,
        bsslab_subjects=_parse_subjects(args.bsslab_subjects),
        max_windows=args.max_windows,
        imu_shift_seconds=args.imu_shift_seconds,
    )


if __name__ == "__main__":
    main()
