from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ecg_pcg_denoise.train.eval_m142_imu import _load_bound_calibration


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_bound_calibration_accepts_matching_provenance(tmp_path: Path) -> None:
    checkpoint = tmp_path / "best_safe.pt"
    checkpoint.write_bytes(b"model-state")
    calibration = tmp_path / "artifact_threshold_calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "fit_split": "val",
                "fold": "M001_S01",
                "variant": "m143v2_correct",
                "checkpoint_sha256": _sha256(checkpoint),
                "threshold": 0.17,
            }
        ),
        encoding="utf-8",
    )

    payload = _load_bound_calibration(
        calibration,
        checkpoint_path=checkpoint,
        fold="M001_S01",
        variant="m143v2_correct",
    )
    assert payload["threshold"] == 0.17


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("fit_split", "test"),
        ("fold", "M001_S02"),
        ("variant", "m143v2_shift"),
        ("checkpoint_sha256", "0" * 64),
    ],
)
def test_bound_calibration_rejects_mismatched_provenance(
    tmp_path: Path,
    field: str,
    replacement: str,
) -> None:
    checkpoint = tmp_path / "best_safe.pt"
    checkpoint.write_bytes(b"model-state")
    payload = {
        "fit_split": "val",
        "fold": "M001_S01",
        "variant": "m143v2_correct",
        "checkpoint_sha256": _sha256(checkpoint),
    }
    payload[field] = replacement
    calibration = tmp_path / "artifact_threshold_calibration.json"
    calibration.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match=field):
        _load_bound_calibration(
            calibration,
            checkpoint_path=checkpoint,
            fold="M001_S01",
            variant="m143v2_correct",
        )
