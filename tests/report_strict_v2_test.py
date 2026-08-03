from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ecg_pcg_denoise.train.report_strict_v2 import (
    _find_variant,
    _paired_difference,
    _require_columns,
    _write_report,
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


def test_examiner_report_is_written_in_english(tmp_path: Path) -> None:
    summary = {
        "m7_strict_esc50": {
            "delta_snr": 13.7,
            "delta_si_sdr": 12.8,
            "corr_estimate": 0.94,
            "log_spectral_distance": 0.013,
        },
        "m143_synthetic_loso": {
            "identity_max_abs_across_folds": 0.0,
            "sqi_improvement_vs_m7_fold_macro": 0.001,
            "artifact_auroc_fold_macro": 0.85,
            "artifact_auprc_fold_macro": 0.94,
            "correct_minus_shift_artifact_probability_fold_macro": 0.09,
            "correct_minus_shuffle_artifact_probability_fold_macro": 0.11,
        },
    }
    folds = [
        {
            "fold": "M001_S01",
            "test_windows": 1082,
            "m143_delta_snr": 3.2,
            "m143_delta_si_sdr": 3.6,
            "m7_sqi_mae": 0.24,
            "m143_sqi_mae": 0.23,
            "sqi_improvement_vs_m7": 0.01,
            "identity_max_abs_vs_m7": 0.0,
        }
    ]

    _write_report(tmp_path, summary, folds)

    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "Executive summary" in report
    assert "Interpretation boundaries" in report
    assert "Four-fold M14.3-v2 results" in report
    assert not (tmp_path / "report_zh.md").exists()
