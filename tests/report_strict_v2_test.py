from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ecg_pcg_denoise.train.report_strict_v2 import (
    _find_variant,
    _paired_difference,
    _require_columns,
    build_report,
)


def _rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for variant, values in {
        "m143v2_correct": (0.8, 0.4),
        "m143v2_shift": (0.5, 0.3),
        "m143v2_shuffle": (0.4, 0.2),
        "m7": (float("nan"), float("nan")),
    }.items():
        for index, value in enumerate(values):
            rows.append(
                {
                    "sample_index": str(index),
                    "variant": variant,
                    "artifact_window_label": str(1 - index),
                    "artifact_prediction_mean": str(value),
                }
            )
    return rows


def test_variant_and_positive_paired_difference() -> None:
    rows = _rows()
    correct = _find_variant(rows, "correct")
    shift = _find_variant(rows, "shift")
    difference = _paired_difference(
        rows,
        correct,
        shift,
        "artifact_prediction_mean",
        positive_only=True,
    )
    np.testing.assert_allclose(difference, np.asarray([0.3]))


def test_pairing_population_mismatch_is_rejected() -> None:
    rows = _rows()
    rows = [
        row
        for row in rows
        if not (
            row["variant"] == "m143v2_shift"
            and row["sample_index"] == "1"
        )
    ]
    with pytest.raises(ValueError, match="populations do not match"):
        _paired_difference(
            rows,
            "m143v2_correct",
            "m143v2_shift",
            "artifact_prediction_mean",
        )


def test_required_columns_report_the_missing_field() -> None:
    with pytest.raises(ValueError, match="missing required columns: value"):
        _require_columns([{"sample_index": "0"}], ("sample_index", "value"))


def test_missing_inputs_do_not_create_report() -> None:
    missing_root = Path("__strict_v2_test_inputs_do_not_exist__")
    report_root = missing_root / "report"
    with pytest.raises(
        FileNotFoundError,
        match="Strict v2 report was not generated",
    ):
        build_report(
            m7_root=missing_root / "missing_m7",
            outputs_root=missing_root / "missing_outputs",
            report_root=report_root,
        )
    assert not report_root.exists()
