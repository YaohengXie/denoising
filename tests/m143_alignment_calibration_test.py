from __future__ import annotations

import numpy as np
import torch

from ecg_pcg_denoise.train.eval_m142_imu import _select_binary_threshold
from ecg_pcg_denoise.train.m142_imu_runtime import (
    m142_fallback_loss,
    prepare_imu_aux,
)


def _imu_batch() -> dict[str, object]:
    features = torch.arange(48, dtype=torch.float32).reshape(1, 6, 8)
    return {
        "imu_feat": features,
        "imu_valid_mask": torch.ones(1, 8),
        "imu_present": torch.ones(1),
        "imu_subject_id": ["subject"],
    }


def test_m143_explicit_shift_is_non_circular_and_bidirectional() -> None:
    batch = _imu_batch()
    original = batch["imu_feat"]
    if not isinstance(original, torch.Tensor):
        raise TypeError("Invalid test IMU tensor.")
    right, _ = prepare_imu_aux(
        batch,
        torch.device("cpu"),
        "shift",
        shift_frames=2,
        shift_direction=1,
    )
    left, _ = prepare_imu_aux(
        batch,
        torch.device("cpu"),
        "shift",
        shift_frames=2,
        shift_direction=-1,
    )
    torch.testing.assert_close(right[..., :2], torch.zeros_like(right[..., :2]))
    torch.testing.assert_close(right[..., 2:], original[..., :-2])
    torch.testing.assert_close(left[..., :-2], original[..., 2:])
    torch.testing.assert_close(left[..., -2:], torch.zeros_like(left[..., -2:]))


def test_m143_validation_threshold_recovers_separable_scores() -> None:
    labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int8)
    scores = np.asarray([0.04, 0.08, 0.12, 0.22, 0.30, 0.41])
    threshold, metrics = _select_binary_threshold(labels, scores)
    assert 0.12 < threshold <= 0.22
    assert float(metrics["balanced_accuracy"]) == 1.0
    assert float(metrics["f1"]) == 1.0
    assert float(metrics["sensitivity"]) == 1.0
    assert float(metrics["specificity"]) == 1.0


def _fallback_output(artifact_probability: float) -> dict[str, torch.Tensor]:
    return {
        "sqi_score": torch.tensor([0.8]),
        "base_sqi_score": torch.tensor([0.8]),
        "mask": torch.ones(1, 2, 3),
        "s1s2_prob": torch.ones(1, 2, 3),
        "imu_reliability": torch.ones(1, 3),
        "imu_artifact_probability": torch.full((1, 3), artifact_probability),
        "imu_sqi_logit_delta": torch.zeros(1),
        "sqi_confidence": torch.full((1,), 0.5),
        "s1s2_confidence": torch.full((1, 2, 3), 0.5),
    }


def test_m143_alignment_margin_penalizes_mismatched_artifact_score() -> None:
    aligned = _fallback_output(0.2)
    shifted = _fallback_output(0.8)
    targets = {
        "sqi": torch.tensor([0.8]),
        "coupled_artifact": torch.ones(1, 3),
        "reliability": torch.ones(1, 3),
        "s1s2_confidence": torch.full((1, 2, 3), 0.5),
        "s1s2_confidence_weight": torch.ones(1, 2, 3),
    }
    config = {
        "training": {
            "loss": {"sqi_confidence_tolerance": 0.1},
            "fallback": {
                "sqi_to_base": 0.0,
                "mask_to_base": 0.0,
                "s1s2_to_base": 0.0,
                "reliability_zero": 0.0,
                "coupled_artifact_zero": 0.0,
                "sqi_delta_zero": 0.0,
                "sqi_confidence": 0.0,
                "s1s2_confidence": 0.0,
                "alignment_margin_weight": 1.0,
                "alignment_margin_value": 0.05,
            },
        }
    }
    loss, components = m142_fallback_loss(
        shifted,
        "shift",
        config,
        reference_output=aligned,
        targets=targets,
        return_components=True,
    )
    assert float(loss) > 0.6
    assert float(components["shift_fallback_alignment_margin"]) > 0.6
