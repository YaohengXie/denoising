"""Build the canonical actual-result document used by the golden checker."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from ecg_pcg_denoise.repro.common import load_json_object


FOLDS = ("M001_S01", "M001_S02", "M001_S03", "M001_S04")


def _integer(row: dict[str, str], field: str) -> int:
    return int(float(row[field]))


def _number(row: dict[str, str], field: str) -> float:
    return float(row[field])


def _read_fold_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    indexed = {row["fold"]: row for row in rows}
    if len(rows) != len(FOLDS) or len(indexed) != len(FOLDS) or set(indexed) != set(FOLDS):
        raise ValueError(f"Expected exactly four fold rows in {path}, found {sorted(indexed)}")
    return indexed


def _ordinary_evaluation_counts(outputs_root: Path) -> tuple[int, int]:
    row_counts: set[int] = set()
    variant_counts: set[int] = set()
    for fold in FOLDS:
        path = (
            outputs_root
            / f"bsslab_m143_imu_fold_{fold}_aligned_calibrated_v2"
            / "eval_test"
            / "window_metrics.csv"
        )
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        row_counts.add(len(rows))
        variant_counts.add(len({row["variant"] for row in rows}))
    if len(row_counts) != 1 or len(variant_counts) != 1:
        raise ValueError(
            "Ordinary M14.3-v2 evaluation counts differ across folds: "
            f"rows={sorted(row_counts)}, variants={sorted(variant_counts)}"
        )
    return next(iter(row_counts)), next(iter(variant_counts))


def build_canonical_actual(
    analysis_summary_path: str | Path,
    fold_summary_path: str | Path,
    outputs_root: str | Path,
) -> dict[str, Any]:
    """Map generated report artifacts to the public reference-value schema."""

    summary = load_json_object(Path(analysis_summary_path))
    folds = _read_fold_rows(Path(fold_summary_path))
    ordinary_rows, ordinary_variants = _ordinary_evaluation_counts(Path(outputs_root))
    m7 = summary["m7_strict_esc50"]
    synthetic = summary["m143_synthetic_loso"]
    synthetic_m7 = synthetic["m7"]
    synthetic_m143 = synthetic["m143"]
    noisy_counts = {
        fold: _integer(folds[fold], "noisy_test_windows") for fold in FOLDS
    }
    test_counts = {_integer(folds[fold], "test_windows") for fold in FOLDS}
    if len(test_counts) != 1:
        raise ValueError(f"Fixed test-window counts differ across folds: {sorted(test_counts)}")

    fold_results: dict[str, dict[str, Any]] = {}
    for fold in FOLDS:
        row = folds[fold]
        fold_results[fold] = {
            "selected_epoch_in_archived_run": _integer(row, "selected_epoch"),
            "artifact_threshold": _number(row, "artifact_threshold"),
            "delta_snr_db": _number(row, "m143_delta_snr"),
            "delta_si_sdr_db": _number(row, "m143_delta_si_sdr"),
            "corr_estimate": _number(row, "m143_corr_estimate"),
            "log_spectral_distance": _number(row, "m143_log_spectral_distance"),
            "m7_sqi_mae": _number(row, "m7_sqi_mae"),
            "m143_sqi_mae": _number(row, "m143_sqi_mae"),
            "artifact_auroc": _number(row, "artifact_auroc"),
            "artifact_auprc": _number(row, "artifact_auprc"),
            "protected_output_max_abs": _number(row, "identity_max_abs_vs_m7"),
        }

    leakage = summary["leakage_audit"]
    return {
        "schema_version": 1,
        "experiment": summary["experiment"],
        "population_counts": {
            "strict_esc50_test_windows": int(m7["n"]),
            "synthetic_imu_loso_noisy_fold_windows": sum(noisy_counts.values()),
            "fixed_test_windows_per_fold": next(iter(test_counts)),
            "ordinary_test_variants_per_window": ordinary_variants,
            "ordinary_test_rows_per_fold": ordinary_rows,
            "noisy_test_windows_by_fold": noisy_counts,
        },
        "leakage_audit": {
            "status": leakage["status"],
            "source_group_overlap_count": int(leakage["source_group_overlap_count"]),
            "exact_file_hash_overlap_count": int(leakage["exact_file_hash_overlap_count"]),
            "scale_normalised_fingerprint_overlap_count": int(
                leakage["scale_normalised_fingerprint_overlap_count"]
            ),
            "mixed_windows_status": leakage["mixed_windows_status"],
        },
        "m7_strict_esc50": {
            "evaluation_scope": m7["evaluation_scope"],
            "aggregation": m7["aggregation"],
            "n": int(m7["n"]),
            "delta_snr_db": float(m7["delta_snr"]),
            "delta_si_sdr_db": float(m7["delta_si_sdr"]),
            "corr_estimate": float(m7["corr_estimate"]),
            "log_spectral_distance": float(m7["log_spectral_distance"]),
            "sqi_mae": float(m7["sqi_mae"]),
            "s1s2_mae": float(m7["s1s2_mae"]),
        },
        "m143_synthetic_imu_loso": {
            "evaluation_scope": synthetic_m143["evaluation_scope"],
            "aggregation": synthetic_m143["aggregation"],
            "n_noisy_fold_windows": sum(noisy_counts.values()),
            "m7_reference": {
                "delta_snr_db": float(synthetic_m7["delta_snr"]),
                "delta_si_sdr_db": float(synthetic_m7["delta_si_sdr"]),
                "corr_estimate": float(synthetic_m7["corr_estimate"]),
                "log_spectral_distance": float(
                    synthetic_m7["log_spectral_distance"]
                ),
                "sqi_mae": float(synthetic_m7["sqi_mae"]),
                "s1s2_mae": float(synthetic_m7["s1s2_mae"]),
            },
            "m143_v2": {
                "delta_snr_db": float(synthetic_m143["delta_snr"]),
                "delta_si_sdr_db": float(synthetic_m143["delta_si_sdr"]),
                "corr_estimate": float(synthetic_m143["corr_estimate"]),
                "log_spectral_distance": float(
                    synthetic_m143["log_spectral_distance"]
                ),
                "sqi_mae": float(synthetic_m143["sqi_mae"]),
                "s1s2_mae": float(synthetic_m143["s1s2_mae"]),
            },
            "sqi_improvement_vs_m7_fold_macro": float(
                synthetic["sqi_improvement_vs_m7_fold_macro"]
            ),
            "protected_output_max_abs_across_folds": float(
                synthetic["identity_max_abs_across_folds"]
            ),
            "artifact_auroc_fold_macro": float(
                synthetic["artifact_auroc_fold_macro"]
            ),
            "artifact_auprc_fold_macro": float(
                synthetic["artifact_auprc_fold_macro"]
            ),
            "artifact_balanced_accuracy_fold_macro": float(
                synthetic["artifact_balanced_accuracy_fold_macro"]
            ),
            "artifact_f1_fold_macro": float(synthetic["artifact_f1_fold_macro"]),
            "correct_minus_shift_artifact_probability_fold_macro": float(
                synthetic["correct_minus_shift_artifact_probability_fold_macro"]
            ),
            "correct_minus_shuffle_artifact_probability_fold_macro": float(
                synthetic["correct_minus_shuffle_artifact_probability_fold_macro"]
            ),
        },
        "folds": fold_results,
    }


def write_canonical_actual(
    analysis_summary_path: str | Path,
    fold_summary_path: str | Path,
    outputs_root: str | Path,
    output_path: str | Path,
) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = build_canonical_actual(
        analysis_summary_path,
        fold_summary_path,
        outputs_root,
    )
    destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return destination
