from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np

from ecg_pcg_denoise.utils.files import ensure_dir


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_M7_ROOT = ROOT / "outputs" / "bsslab_esc50_v2_m7_robust_sqi_eval"
DEFAULT_REPORT_ROOT = ROOT / "reports" / "strict_esc50_m143_v2"
DEFAULT_MANIFESTS_ROOT = ROOT / "data" / "manifests"
FOLDS = ("M001_S01", "M001_S02", "M001_S03", "M001_S04")
VARIANT_PREFIX = "m143v2"
LAG_DIRECTORIES = {
    0.128: "eval_lag_0128",
    0.256: "eval_lag_0256",
    0.512: "eval_test",
    1.024: "eval_lag_1024",
}
NOISY_MODES = {
    "combined",
    "independent_artifact",
    "motion_artifact",
}
METRICS = (
    "delta_snr",
    "delta_si_sdr",
    "corr_estimate",
    "log_spectral_distance",
)
COLORS = {
    "blue": "#3568A8",
    "blue_light": "#9CB7D7",
    "orange": "#C95D3A",
    "gold": "#D39B2A",
    "olive": "#718355",
    "ink": "#30343B",
    "grey": "#9AA1A8",
    "light": "#E6E9ED",
    "open": "#F6F7F8",
}


def _m143_root(fold: str, outputs_root: Path) -> Path:
    return outputs_root / f"bsslab_m143_imu_fold_{fold}_aligned_calibrated_v2"


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV contains no rows: {path}")
    return rows


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    ensure_dir(path.parent)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _finite_mean(values: Iterable[float]) -> float:
    selected = np.asarray(
        [float(value) for value in values if math.isfinite(float(value))],
        dtype=np.float64,
    )
    return float(np.mean(selected)) if selected.size else float("nan")


def _require_columns(
    rows: list[dict[str, str]],
    columns: Iterable[str],
    source: str = "rows",
) -> None:
    available = set(rows[0]) if rows else set()
    missing = sorted(set(columns) - available)
    if missing:
        raise ValueError(
            f"{source} is missing required columns: {', '.join(missing)}"
        )


def _find_variant(
    rows: list[dict[str, str]],
    suffix: str,
    prefix: str = VARIANT_PREFIX,
) -> str:
    expected = f"{prefix}_{suffix}"
    variants = {str(row.get("variant", "")) for row in rows}
    if expected not in variants:
        raise ValueError(
            f"Expected variant '{expected}', found: {sorted(variants)}"
        )
    return expected


def _variant_rows(
    rows: list[dict[str, str]],
    variant: str,
) -> list[dict[str, str]]:
    selected = [row for row in rows if row.get("variant") == variant]
    if not selected:
        raise ValueError(f"No rows found for variant={variant}")
    return selected


def _mean(rows: list[dict[str, str]], field: str) -> float:
    if not rows:
        return float("nan")
    return _finite_mean(float(row[field]) for row in rows)


def _paired_difference(
    rows: list[dict[str, str]],
    left_variant: str,
    right_variant: str,
    field: str,
    positive_only: bool = False,
) -> np.ndarray:
    def selected(variant: str) -> dict[str, dict[str, str]]:
        output: dict[str, dict[str, str]] = {}
        for row in rows:
            if row.get("variant") != variant:
                continue
            if positive_only and int(row["artifact_window_label"]) != 1:
                continue
            output[str(row["sample_index"])] = row
        return output

    left = selected(left_variant)
    right = selected(right_variant)
    if set(left) != set(right):
        left_only = sorted(set(left) - set(right))[:5]
        right_only = sorted(set(right) - set(left))[:5]
        raise ValueError(
            "Counterfactual populations do not match. "
            f"left_only={left_only}, right_only={right_only}"
        )
    if not left:
        raise ValueError(
            f"No paired rows for {left_variant} versus {right_variant}"
        )
    return np.asarray(
        [
            float(left[index][field]) - float(right[index][field])
            for index in sorted(left, key=lambda value: int(value))
        ],
        dtype=np.float64,
    )


def _classification_row(
    rows: list[dict[str, str]],
    variant: str,
) -> dict[str, str]:
    selected = [
        row
        for row in rows
        if row.get("variant") == variant
        and row.get("task") == "artifact_window"
        and row.get("score_kind") == "validation_selected_threshold"
    ]
    if len(selected) != 1:
        raise ValueError(
            "Expected one validation-selected artifact_window row for "
            f"{variant}; found {len(selected)}."
        )
    return selected[0]


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _required_paths(
    m7_root: Path,
    outputs_root: Path,
    manifests_root: Path = DEFAULT_MANIFESTS_ROOT,
) -> list[Path]:
    paths = [
        manifests_root / "esc50_strict_audit.json",
        manifests_root / "bsslab_esc50_v2_mixed_audit.json",
        m7_root / "eval" / "window_metrics.csv",
    ]
    for fold in FOLDS:
        root = _m143_root(fold, outputs_root)
        paths.extend(
            [
                root / "eval_test" / "window_metrics.csv",
                root / "eval_test" / "classification_metrics.csv",
                root / "artifact_threshold_calibration.json",
                root / "checkpoint_selection_summary.json",
            ]
        )
        paths.extend(
            root / directory / "window_metrics.csv"
            for lag, directory in LAG_DIRECTORIES.items()
            if lag != 0.512
        )
    return paths


def _check_inputs(
    m7_root: Path,
    outputs_root: Path,
    manifests_root: Path = DEFAULT_MANIFESTS_ROOT,
) -> tuple[dict[str, Any], dict[str, Any]]:
    missing = [
        path
        for path in _required_paths(m7_root, outputs_root, manifests_root)
        if not path.exists()
    ]
    if missing:
        formatted = "\n".join(f"  - {_relative(path)}" for path in missing)
        raise FileNotFoundError(
            "Strict v2 report was not generated because required evaluation "
            f"artifacts are missing ({len(missing)}):\n{formatted}\n"
            "Complete the strict M7 ESC-50 evaluation, four M14.3 test "
            "evaluations, validation-only threshold calibration, and "
            "0.128/0.256/0.512/1.024 s lag evaluations first. No report "
            "files have been written."
        )
    strict_audit = _read_json(
        manifests_root / "esc50_strict_audit.json"
    )
    mixed_audit = _read_json(
        manifests_root / "bsslab_esc50_v2_mixed_audit.json"
    )
    failures: list[str] = []
    if str(strict_audit.get("status", "")).lower() != "pass":
        failures.append("esc50_strict_audit.status is not pass")
    if str(mixed_audit.get("status", "")).lower() != "pass":
        failures.append("bsslab_esc50_v2_mixed_audit.status is not pass")
    for split_name in ("standard", "enhanced"):
        split = mixed_audit.get(split_name, {})
        if not isinstance(split, dict) or split.get("status") != "pass":
            failures.append(f"mixed audit {split_name}.status is not pass")
    if failures:
        raise RuntimeError(
            "Strict v2 leakage audit failed; report generation stopped: "
            + "; ".join(failures)
        )
    return strict_audit, mixed_audit


def _aggregate_m7(
    m7_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source = m7_root / "eval" / "window_metrics.csv"
    rows = _read_csv(source)
    _require_columns(
        rows,
        (
            "snr_db",
            "delta_snr",
            "delta_si_sdr",
            "corr_estimate",
            "log_spectral_distance",
            "sqi_abs_error",
            "s1s2_mae",
        ),
        _relative(source),
    )
    overall = [
        {
            "evaluation_scope": "strict_esc50_test",
            "model": "M7-robust-SQI-v2",
            "aggregation": "window_mean",
            "n": len(rows),
            **{field: _mean(rows, field) for field in METRICS},
            "sqi_mae": _mean(rows, "sqi_abs_error"),
            "s1s2_mae": _mean(rows, "s1s2_mae"),
        }
    ]
    grouped: dict[float, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[float(row["snr_db"])].append(row)
    by_snr = [
        {
            "evaluation_scope": "strict_esc50_test",
            "model": "M7-robust-SQI-v2",
            "snr_db": snr,
            "n": len(selected),
            **{field: _mean(selected, field) for field in METRICS},
            "sqi_mae": _mean(selected, "sqi_abs_error"),
            "s1s2_mae": _mean(selected, "s1s2_mae"),
        }
        for snr, selected in sorted(grouped.items())
    ]
    return overall, by_snr


def _aggregate_m143(
    outputs_root: Path,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    folds: list[dict[str, Any]] = []
    by_snr: list[dict[str, Any]] = []
    identity: list[dict[str, Any]] = []
    counterfactual: list[dict[str, Any]] = []
    lag_rows: list[dict[str, Any]] = []

    for fold in FOLDS:
        root = _m143_root(fold, outputs_root)
        source = root / "eval_test" / "window_metrics.csv"
        rows = _read_csv(source)
        _require_columns(
            rows,
            (
                "sample_index",
                "variant",
                "artifact_mode",
                "artifact_window_label",
                "snr_db",
                "mask_max_abs_vs_m7",
                "waveform_max_abs_vs_m7",
                "s1s2_max_abs_vs_m7",
                "s1s2_target_mae",
                "sqi_abs_error",
                "sqi_m7_abs_error",
                "artifact_prediction_mean",
                *METRICS,
            ),
            _relative(source),
        )
        correct_name = _find_variant(rows, "correct")
        shift_name = _find_variant(rows, "shift")
        shuffle_name = _find_variant(rows, "shuffle")
        correct = _variant_rows(rows, correct_name)
        m7 = _variant_rows(rows, "m7")
        shifted = _variant_rows(rows, shift_name)
        shuffled = _variant_rows(rows, shuffle_name)
        correct_ids = {row["sample_index"] for row in correct}
        m7_ids = {row["sample_index"] for row in m7}
        if correct_ids != m7_ids:
            raise ValueError(
                f"{fold}: M7 and M14.3 test populations are not identical."
            )
        noisy_correct = [
            row for row in correct if row["artifact_mode"] in NOISY_MODES
        ]
        noisy_m7 = [row for row in m7 if row["artifact_mode"] in NOISY_MODES]

        class_rows = _read_csv(
            root / "eval_test" / "classification_metrics.csv"
        )
        classification = _classification_row(class_rows, correct_name)
        threshold_payload = _read_json(
            root / "artifact_threshold_calibration.json"
        )
        if threshold_payload.get("fit_split") != "val":
            raise RuntimeError(
                f"{fold}: artifact threshold was not fitted on validation."
            )
        selection = _read_json(root / "checkpoint_selection_summary.json")
        max_mask = max(float(row["mask_max_abs_vs_m7"]) for row in correct)
        max_wave = max(
            float(row["waveform_max_abs_vs_m7"]) for row in correct
        )
        max_s1s2 = max(
            float(row["s1s2_max_abs_vs_m7"]) for row in correct
        )
        fold_row = {
            "fold": fold,
            "test_windows": len(correct),
            "noisy_test_windows": len(noisy_correct),
            "m7_delta_snr": _mean(noisy_m7, "delta_snr"),
            "m143_delta_snr": _mean(noisy_correct, "delta_snr"),
            "m7_delta_si_sdr": _mean(noisy_m7, "delta_si_sdr"),
            "m143_delta_si_sdr": _mean(noisy_correct, "delta_si_sdr"),
            "m7_corr_estimate": _mean(noisy_m7, "corr_estimate"),
            "m143_corr_estimate": _mean(noisy_correct, "corr_estimate"),
            "m7_log_spectral_distance": _mean(
                noisy_m7, "log_spectral_distance"
            ),
            "m143_log_spectral_distance": _mean(
                noisy_correct, "log_spectral_distance"
            ),
            "m7_sqi_mae": _mean(m7, "sqi_abs_error"),
            "m143_sqi_mae": _mean(correct, "sqi_abs_error"),
            "sqi_improvement_vs_m7": (
                _mean(m7, "sqi_abs_error")
                - _mean(correct, "sqi_abs_error")
            ),
            "s1s2_target_mae": _mean(correct, "s1s2_target_mae"),
            "mask_max_abs_vs_m7": max_mask,
            "waveform_max_abs_vs_m7": max_wave,
            "s1s2_max_abs_vs_m7": max_s1s2,
            "identity_max_abs_vs_m7": max(max_mask, max_wave, max_s1s2),
            "artifact_threshold": float(threshold_payload["threshold"]),
            "artifact_auroc": float(classification["auroc"]),
            "artifact_auprc": float(classification["auprc"]),
            "artifact_balanced_accuracy": float(
                classification["balanced_accuracy"]
            ),
            "artifact_f1": float(classification["f1"]),
            "artifact_sensitivity": float(classification["sensitivity"]),
            "artifact_specificity": float(classification["specificity"]),
            "selected_epoch": int(selection["selected_epoch"]),
        }
        folds.append(fold_row)
        identity.extend(
            {
                "fold": fold,
                "output": output,
                "max_abs_vs_m7": value,
            }
            for output, value in (
                ("Mask", max_mask),
                ("Waveform", max_wave),
                ("S1/S2 location", max_s1s2),
            )
        )

        for variant_name, variant_rows in (
            ("M7", m7),
            ("M14.3-v2", correct),
        ):
            snr_groups: dict[float, list[dict[str, str]]] = defaultdict(list)
            for row in variant_rows:
                if row["artifact_mode"] in NOISY_MODES:
                    snr_groups[float(row["snr_db"])].append(row)
            by_snr.extend(
                {
                    "fold": fold,
                    "model": variant_name,
                    "snr_db": snr,
                    "n": len(selected),
                    **{field: _mean(selected, field) for field in METRICS},
                }
                for snr, selected in sorted(snr_groups.items())
            )

        positive_correct = [
            row for row in correct if int(row["artifact_window_label"]) == 1
        ]
        positive_shifted = [
            row for row in shifted if int(row["artifact_window_label"]) == 1
        ]
        positive_shuffled = [
            row for row in shuffled if int(row["artifact_window_label"]) == 1
        ]
        counterfactual.append(
            {
                "fold": fold,
                "positive_windows": len(positive_correct),
                "correct_artifact_probability": _mean(
                    positive_correct, "artifact_prediction_mean"
                ),
                "shift_artifact_probability": _mean(
                    positive_shifted, "artifact_prediction_mean"
                ),
                "shuffle_artifact_probability": _mean(
                    positive_shuffled, "artifact_prediction_mean"
                ),
                "correct_minus_shift_artifact_probability": float(
                    np.mean(
                        _paired_difference(
                            rows,
                            correct_name,
                            shift_name,
                            "artifact_prediction_mean",
                            positive_only=True,
                        )
                    )
                ),
                "correct_minus_shuffle_artifact_probability": float(
                    np.mean(
                        _paired_difference(
                            rows,
                            correct_name,
                            shuffle_name,
                            "artifact_prediction_mean",
                            positive_only=True,
                        )
                    )
                ),
                "correct_sqi_mae": _mean(correct, "sqi_abs_error"),
                "shift_sqi_mae": _mean(shifted, "sqi_abs_error"),
                "shuffle_sqi_mae": _mean(shuffled, "sqi_abs_error"),
                "shift_minus_correct_sqi_mae": (
                    _mean(shifted, "sqi_abs_error")
                    - _mean(correct, "sqi_abs_error")
                ),
                "shuffle_minus_correct_sqi_mae": (
                    _mean(shuffled, "sqi_abs_error")
                    - _mean(correct, "sqi_abs_error")
                ),
            }
        )

        for lag_seconds, directory in LAG_DIRECTORIES.items():
            lag_source = root / directory / "window_metrics.csv"
            lag_data = _read_csv(lag_source)
            lag_correct = _find_variant(lag_data, "correct")
            lag_shift = _find_variant(lag_data, "shift")
            lag_correct_rows = _variant_rows(lag_data, lag_correct)
            lag_shift_rows = _variant_rows(lag_data, lag_shift)
            lag_rows.append(
                {
                    "fold": fold,
                    "lag_seconds": lag_seconds,
                    "positive_windows": sum(
                        int(row["artifact_window_label"]) == 1
                        for row in lag_correct_rows
                    ),
                    "correct_minus_shift_artifact_probability": float(
                        np.mean(
                            _paired_difference(
                                lag_data,
                                lag_correct,
                                lag_shift,
                                "artifact_prediction_mean",
                                positive_only=True,
                            )
                        )
                    ),
                    "shift_minus_correct_sqi_mae": (
                        _mean(lag_shift_rows, "sqi_abs_error")
                        - _mean(lag_correct_rows, "sqi_abs_error")
                    ),
                }
            )
    return folds, by_snr, identity, counterfactual, lag_rows


def _overall_rows(
    m7_overall: list[dict[str, Any]],
    folds: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = list(m7_overall)
    for model_key, model_label in (
        ("m7", "M7-robust-SQI-v2"),
        ("m143", "M14.3-v2"),
    ):
        rows.append(
            {
                "evaluation_scope": "synthetic_imu_loso_test",
                "model": model_label,
                "aggregation": "fold_macro",
                "n": sum(int(row["noisy_test_windows"]) for row in folds),
                "delta_snr": _finite_mean(
                    row[f"{model_key}_delta_snr"] for row in folds
                ),
                "delta_si_sdr": _finite_mean(
                    row[f"{model_key}_delta_si_sdr"] for row in folds
                ),
                "corr_estimate": _finite_mean(
                    row[f"{model_key}_corr_estimate"] for row in folds
                ),
                "log_spectral_distance": _finite_mean(
                    row[f"{model_key}_log_spectral_distance"] for row in folds
                ),
                "sqi_mae": _finite_mean(
                    row[f"{model_key}_sqi_mae"] for row in folds
                ),
                "s1s2_mae": _finite_mean(
                    row["s1s2_target_mae"] for row in folds
                ),
            }
        )
    return rows


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.edgecolor": COLORS["ink"],
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": "#D9DDE2",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.75,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_m7(
    overall: list[dict[str, Any]],
    by_snr: list[dict[str, Any]],
    path: Path,
) -> None:
    row = overall[0]
    ordered = sorted(by_snr, key=lambda value: float(value["snr_db"]))
    snr = np.asarray([float(item["snr_db"]) for item in ordered])
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2))

    values = [float(row["delta_snr"]), float(row["delta_si_sdr"])]
    bars = axes[0, 0].bar(
        ["ΔSNR", "ΔSI-SDR"],
        values,
        color=[COLORS["blue"], COLORS["orange"]],
        edgecolor=COLORS["ink"],
        linewidth=0.7,
    )
    axes[0, 0].bar_label(bars, fmt="%.3f dB", padding=4)
    axes[0, 0].set_ylim(0, max(values) * 1.22)
    axes[0, 0].set_ylabel("Improvement (dB)")
    axes[0, 0].set_title("Overall waveform improvement")

    for field, label, color, marker in (
        ("delta_snr", "ΔSNR", COLORS["blue"], "o"),
        ("delta_si_sdr", "ΔSI-SDR", COLORS["orange"], "s"),
    ):
        axes[0, 1].plot(
            snr,
            [float(item[field]) for item in ordered],
            color=color,
            marker=marker,
            linewidth=2.0,
            label=label,
        )
    axes[0, 1].set_title("Improvement by nominal input SNR")
    axes[0, 1].set_xlabel("Input SNR (dB)")
    axes[0, 1].set_ylabel("Improvement (dB)")
    axes[0, 1].legend()

    axes[1, 0].plot(
        snr,
        [float(item["corr_estimate"]) for item in ordered],
        color=COLORS["blue"],
        marker="o",
        linewidth=2.0,
    )
    axes[1, 0].set_ylim(0, 1.0)
    axes[1, 0].set_title("Output–clean waveform correlation")
    axes[1, 0].set_xlabel("Input SNR (dB)")
    axes[1, 0].set_ylabel("Correlation")

    axes[1, 1].plot(
        snr,
        [float(item["log_spectral_distance"]) for item in ordered],
        color=COLORS["orange"],
        marker="s",
        linewidth=2.0,
    )
    axes[1, 1].set_ylim(
        0,
        max(float(item["log_spectral_distance"]) for item in ordered) * 1.20,
    )
    axes[1, 1].set_title("Log-spectral distance")
    axes[1, 1].set_xlabel("Input SNR (dB)")
    axes[1, 1].set_ylabel("LSD (lower is better)")

    fig.suptitle("M7-robust-SQI-v2 on the strict ESC-50 test partition", y=1.01)
    fig.text(
        0.5,
        -0.01,
        f"Window-level means; n={int(row['n']):,}. ESC-50 test clips and "
        "source groups are sealed from training and validation.",
        ha="center",
        color=COLORS["ink"],
    )
    fig.tight_layout()
    _save(fig, path)


def _plot_identity(
    identity: list[dict[str, Any]],
    path: Path,
) -> None:
    outputs = ("Mask", "Waveform", "S1/S2 location")
    matrix = np.asarray(
        [
            [
                next(
                    float(row["max_abs_vs_m7"])
                    for row in identity
                    if row["fold"] == fold and row["output"] == output
                )
                for output in outputs
            ]
            for fold in FOLDS
        ],
        dtype=np.float64,
    )
    vmax = max(float(np.max(matrix)), 1e-12)
    fig, ax = plt.subplots(figsize=(8.8, 5.3))
    image = ax.imshow(matrix, cmap="Blues", vmin=0.0, vmax=vmax, aspect="auto")
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            ax.text(
                column_index,
                row_index,
                f"{matrix[row_index, column_index]:.2e}",
                ha="center",
                va="center",
                color=COLORS["ink"],
                fontweight="bold",
            )
    ax.set_xticks(np.arange(len(outputs)), outputs)
    ax.set_yticks(np.arange(len(FOLDS)), FOLDS)
    ax.set_title("Maximum absolute difference from the frozen M7 output")
    ax.set_xlabel("Audited output")
    ax.set_ylabel("Held-out Motema fold")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.04)
    colorbar.set_label("Max absolute difference")
    status = (
        "Exact identity in every audited output"
        if float(np.max(matrix)) == 0.0
        else "Non-zero identity difference detected"
    )
    fig.suptitle("M14.3-v2 denoising and S1/S2 identity audit", y=1.01)
    fig.text(
        0.5,
        0.01,
        status + "; exact values are printed in every cell.",
        ha="center",
        color=COLORS["ink"],
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _save(fig, path)


def _plot_sqi(
    folds: list[dict[str, Any]],
    path: Path,
) -> None:
    x = np.arange(len(folds))
    m7 = np.asarray([float(row["m7_sqi_mae"]) for row in folds])
    m143 = np.asarray([float(row["m143_sqi_mae"]) for row in folds])
    gain = m7 - m143
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.2))
    width = 0.34
    bars_m7 = axes[0].bar(
        x - width / 2,
        m7,
        width,
        label="M7",
        color=COLORS["blue_light"],
        edgecolor=COLORS["ink"],
        linewidth=0.6,
    )
    bars_m143 = axes[0].bar(
        x + width / 2,
        m143,
        width,
        label="M14.3-v2",
        color=COLORS["orange"],
        edgecolor=COLORS["ink"],
        linewidth=0.6,
    )
    axes[0].bar_label(bars_m7, fmt="%.4f", padding=3, fontsize=8)
    axes[0].bar_label(bars_m143, fmt="%.4f", padding=3, fontsize=8)
    axes[0].set_ylim(0, max(float(np.max(m7)), float(np.max(m143))) * 1.25)
    axes[0].set_xticks(x, [row["fold"].removeprefix("M001_") for row in folds])
    axes[0].set_ylabel("SQI MAE (lower is better)")
    axes[0].set_title("Absolute SQI error")
    axes[0].legend()

    gain_bars = axes[1].bar(
        x,
        gain,
        color=COLORS["gold"],
        edgecolor=COLORS["ink"],
        linewidth=0.6,
    )
    axes[1].axhline(0, color=COLORS["ink"], linewidth=1.0)
    axes[1].bar_label(gain_bars, fmt="%+.6f", padding=3, fontsize=8)
    limit = max(float(np.max(np.abs(gain))) * 1.35, 1e-5)
    axes[1].set_ylim(-limit, limit)
    axes[1].set_xticks(x, [row["fold"].removeprefix("M001_") for row in folds])
    axes[1].set_ylabel("M7 MAE − M14.3 MAE")
    axes[1].set_title("Paired SQI improvement")

    fig.suptitle("SQI comparison on four M14.3-v2 LOSO folds", y=1.02)
    fig.text(
        0.5,
        -0.01,
        "Same fixed test windows within each fold; positive improvement favours "
        "M14.3-v2. Bars start at zero in the absolute-error panel.",
        ha="center",
        color=COLORS["ink"],
    )
    fig.tight_layout()
    _save(fig, path)


def _plot_artifact(
    folds: list[dict[str, Any]],
    counterfactual: list[dict[str, Any]],
    lags: list[dict[str, Any]],
    path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 9.2))
    for fold in FOLDS:
        selected = sorted(
            [row for row in lags if row["fold"] == fold],
            key=lambda row: float(row["lag_seconds"]),
        )
        axes[0, 0].plot(
            [float(row["lag_seconds"]) * 1000 for row in selected],
            [
                float(row["correct_minus_shift_artifact_probability"])
                for row in selected
            ],
            marker="o",
            linewidth=1.8,
            label=fold.removeprefix("M001_"),
        )
    axes[0, 0].axhline(0, color=COLORS["ink"], linewidth=1.0)
    axes[0, 0].set_title("Aligned-minus-shifted artifact score")
    axes[0, 0].set_xlabel("Absolute IMU shift (ms)")
    axes[0, 0].set_ylabel("Probability difference on positive windows")
    axes[0, 0].legend(title="Fold", ncol=2)

    x = np.arange(len(counterfactual))
    width = 0.24
    for offset, field, label, color in (
        (-width, "correct_artifact_probability", "Correct", COLORS["orange"]),
        (0.0, "shift_artifact_probability", "Shifted", COLORS["gold"]),
        (width, "shuffle_artifact_probability", "Shuffled", COLORS["blue_light"]),
    ):
        axes[0, 1].bar(
            x + offset,
            [float(row[field]) for row in counterfactual],
            width,
            label=label,
            color=color,
            edgecolor=COLORS["ink"],
            linewidth=0.5,
        )
    axes[0, 1].set_xticks(
        x, [row["fold"].removeprefix("M001_") for row in counterfactual]
    )
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].set_ylabel("Mean artifact probability")
    axes[0, 1].set_title("IMU counterfactual scores on positive windows")
    axes[0, 1].legend(ncol=3)

    metric_fields = (
        ("artifact_auroc", "AUROC"),
        ("artifact_auprc", "AUPRC"),
        ("artifact_balanced_accuracy", "Balanced accuracy"),
        ("artifact_f1", "F1"),
    )
    metric_x = np.arange(len(metric_fields))
    fold_width = 0.18
    palette = (
        COLORS["blue"],
        COLORS["orange"],
        COLORS["gold"],
        COLORS["olive"],
    )
    for index, (fold, color) in enumerate(zip(FOLDS, palette, strict=True)):
        row = next(value for value in folds if value["fold"] == fold)
        axes[1, 0].bar(
            metric_x + (index - 1.5) * fold_width,
            [float(row[field]) for field, _ in metric_fields],
            fold_width,
            label=fold.removeprefix("M001_"),
            color=color,
            edgecolor=COLORS["ink"],
            linewidth=0.45,
        )
    axes[1, 0].set_xticks(
        metric_x, [label for _, label in metric_fields], rotation=12
    )
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].set_ylabel("Score")
    axes[1, 0].set_title("Validation-threshold artifact detection")
    axes[1, 0].legend(title="Fold", ncol=2)

    width = 0.34
    shift_gap = [
        float(row["shift_minus_correct_sqi_mae"]) for row in counterfactual
    ]
    shuffle_gap = [
        float(row["shuffle_minus_correct_sqi_mae"]) for row in counterfactual
    ]
    axes[1, 1].bar(
        x - width / 2,
        shift_gap,
        width,
        label="Shift − correct",
        color=COLORS["gold"],
        edgecolor=COLORS["ink"],
        linewidth=0.5,
    )
    axes[1, 1].bar(
        x + width / 2,
        shuffle_gap,
        width,
        label="Shuffle − correct",
        color=COLORS["blue_light"],
        edgecolor=COLORS["ink"],
        linewidth=0.5,
    )
    axes[1, 1].axhline(0, color=COLORS["ink"], linewidth=1.0)
    axes[1, 1].set_xticks(
        x, [row["fold"].removeprefix("M001_") for row in counterfactual]
    )
    axes[1, 1].set_ylabel("SQI MAE difference")
    axes[1, 1].set_title("SQI sensitivity to mismatched IMU")
    axes[1, 1].legend()

    fig.suptitle(
        "M14.3-v2 artifact alignment and IMU counterfactual evaluation",
        y=1.01,
    )
    fig.text(
        0.5,
        -0.005,
        "Artifact positives are motion_artifact or combined windows. Detection "
        "thresholds were selected on validation only.",
        ha="center",
        color=COLORS["ink"],
    )
    fig.tight_layout()
    _save(fig, path)


def _summary(
    strict_audit: dict[str, Any],
    mixed_audit: dict[str, Any],
    overall: list[dict[str, Any]],
    folds: list[dict[str, Any]],
    counterfactual: list[dict[str, Any]],
) -> dict[str, Any]:
    m7_strict = next(
        row for row in overall if row["evaluation_scope"] == "strict_esc50_test"
    )
    m7_synthetic = next(
        row
        for row in overall
        if row["evaluation_scope"] == "synthetic_imu_loso_test"
        and row["model"] == "M7-robust-SQI-v2"
    )
    m143 = next(
        row
        for row in overall
        if row["evaluation_scope"] == "synthetic_imu_loso_test"
        and row["model"] == "M14.3-v2"
    )
    identity_max = max(float(row["identity_max_abs_vs_m7"]) for row in folds)
    return {
        "schema_version": 1,
        "experiment": "strict ESC-50 v2 M7 and four-fold M14.3-v2",
        "leakage_audit": {
            "status": "pass",
            "esc50_manifest_status": strict_audit.get("status"),
            "source_group_overlap_count": strict_audit.get(
                "source_group_overlap_count"
            ),
            "exact_file_hash_overlap_count": strict_audit.get(
                "exact_file_hash_overlap_count"
            ),
            "scale_normalised_fingerprint_overlap_count": strict_audit.get(
                "scale_normalised_fingerprint_overlap_count"
            ),
            "mixed_windows_status": mixed_audit.get("status"),
        },
        "m7_strict_esc50": m7_strict,
        "m143_synthetic_loso": {
            "m7": m7_synthetic,
            "m143": m143,
            "sqi_improvement_vs_m7_fold_macro": _finite_mean(
                row["sqi_improvement_vs_m7"] for row in folds
            ),
            "identity_max_abs_across_folds": identity_max,
            "exact_m7_denoising_and_s1s2_identity": identity_max == 0.0,
            "artifact_auroc_fold_macro": _finite_mean(
                row["artifact_auroc"] for row in folds
            ),
            "artifact_auprc_fold_macro": _finite_mean(
                row["artifact_auprc"] for row in folds
            ),
            "artifact_balanced_accuracy_fold_macro": _finite_mean(
                row["artifact_balanced_accuracy"] for row in folds
            ),
            "artifact_f1_fold_macro": _finite_mean(
                row["artifact_f1"] for row in folds
            ),
            "correct_minus_shift_artifact_probability_fold_macro": _finite_mean(
                row["correct_minus_shift_artifact_probability"]
                for row in counterfactual
            ),
            "correct_minus_shuffle_artifact_probability_fold_macro": _finite_mean(
                row["correct_minus_shuffle_artifact_probability"]
                for row in counterfactual
            ),
        },
        "scope_warning": (
            "The strict ESC-50 M7 test and synthetic IMU LOSO test are "
            "different populations; their absolute metrics must not be "
            "interpreted as a paired model comparison."
        ),
    }


def _write_report(
    report_root: Path,
    summary: dict[str, Any],
    folds: list[dict[str, Any]],
) -> None:
    m7 = summary["m7_strict_esc50"]
    m143 = summary["m143_synthetic_loso"]
    identity = float(m143["identity_max_abs_across_folds"])
    sqi_gain = float(m143["sqi_improvement_vs_m7_fold_macro"])
    fold_lines = "\n".join(
        "| {fold} | {n} | {dsn:.4f} | {dsi:.4f} | {m7sqi:.6f} | "
        "{m143sqi:.6f} | {gain:+.6f} | {identity:.2e} |".format(
            fold=row["fold"],
            n=int(row["test_windows"]),
            dsn=float(row["m143_delta_snr"]),
            dsi=float(row["m143_delta_si_sdr"]),
            m7sqi=float(row["m7_sqi_mae"]),
            m143sqi=float(row["m143_sqi_mae"]),
            gain=float(row["sqi_improvement_vs_m7"]),
            identity=float(row["identity_max_abs_vs_m7"]),
        )
        for row in folds
    )
    report = f"""# Strict ESC-50 v2 与 M14.3-v2 结果汇总

## 结论

本报告仅在 ESC-50 严格分割和四折 M14.3-v2 结果全部存在、且两份
leakage audit 均为 `pass` 时生成。ESC-50 的 source group、文件哈希和
scale-normalised fingerprint 跨集合重叠均为 0。

M7-robust-SQI-v2 在严格 ESC-50 测试集上的窗口均值为：
ΔSNR `{float(m7['delta_snr']):.4f}` dB、ΔSI-SDR
`{float(m7['delta_si_sdr']):.4f}` dB、Corr
`{float(m7['corr_estimate']):.5f}`、LSD
`{float(m7['log_spectral_distance']):.5f}`。

在合成 IMU LOSO 测试上，M14.3-v2 相对 M7 的 mask、去噪波形及
S1/S2 location 最大绝对差为 `{identity:.2e}`。SQI MAE 的四折宏平均
改善为 `{sqi_gain:+.6f}`。artifact AUROC/AUPRC 的四折宏平均为
`{float(m143['artifact_auroc_fold_macro']):.4f}` /
`{float(m143['artifact_auprc_fold_macro']):.4f}`；正确 IMU 相对
512 ms shift 与 shuffle 的 artifact score 优势分别为
`{float(m143['correct_minus_shift_artifact_probability_fold_macro']):+.4f}` 和
`{float(m143['correct_minus_shuffle_artifact_probability_fold_macro']):+.4f}`。

## M7 严格 ESC-50 去噪

![M7 strict ESC-50](figures/fig01_m7_esc50_denoising.png)

## M14.3-v2 四折结果

| Fold | 窗口数 | ΔSNR | ΔSI-SDR | M7 SQI MAE | M14.3 SQI MAE | SQI改善 | 最大identity差 |
|---|---:|---:|---:|---:|---:|---:|---:|
{fold_lines}

![Identity audit](figures/fig02_m143_identity.png)

![SQI comparison](figures/fig03_sqi_comparison.png)

![Artifact alignment](figures/fig04_artifact_alignment_counterfactual.png)

## 解释边界

- ESC-50 strict test 与 M14.3 synthetic IMU LOSO test 是不同测试总体，
  不能把两组绝对数值作配对比较。
- 四个 LOSO fold 改变的是 held-out Motema IMU ID；BSSLAB 测试窗口会在
  各 fold 中重复，因此不能将四折行数当作独立 BSSLAB 样本数。
- M14.3-v2 的 IMU 不改变 mask、waveform 或 S1/S2 location；它只影响
  SQI residual 与辅助置信度/运动/伪影/IMU有效性输出。
- 本报告没有把真实 Motema task condition 当作人工 artifact ground truth。

## 可复核表格

- `data/overall_summary.csv`
- `data/m7_by_snr.csv`
- `data/m143_fold_summary.csv`
- `data/m143_by_snr.csv`
- `data/m143_identity_audit.csv`
- `data/m143_artifact_counterfactual.csv`
- `data/m143_artifact_lag.csv`
- `data/analysis_summary.json`
"""
    (report_root / "report_zh.md").write_text(report, encoding="utf-8")


def build_report(
    m7_root: Path = DEFAULT_M7_ROOT,
    outputs_root: Path = ROOT / "outputs",
    report_root: Path = DEFAULT_REPORT_ROOT,
    manifests_root: Path = DEFAULT_MANIFESTS_ROOT,
) -> Path:
    strict_audit, mixed_audit = _check_inputs(
        m7_root,
        outputs_root,
        manifests_root,
    )
    m7_overall, m7_by_snr = _aggregate_m7(m7_root)
    folds, m143_by_snr, identity, counterfactual, lags = _aggregate_m143(
        outputs_root
    )
    overall = _overall_rows(m7_overall, folds)
    summary = _summary(
        strict_audit,
        mixed_audit,
        overall,
        folds,
        counterfactual,
    )

    data_root = ensure_dir(report_root / "data")
    figure_root = ensure_dir(report_root / "figures")
    _write_csv(data_root / "overall_summary.csv", overall)
    _write_csv(data_root / "m7_by_snr.csv", m7_by_snr)
    _write_csv(data_root / "m143_fold_summary.csv", folds)
    _write_csv(data_root / "m143_by_snr.csv", m143_by_snr)
    _write_csv(data_root / "m143_identity_audit.csv", identity)
    _write_csv(
        data_root / "m143_artifact_counterfactual.csv",
        counterfactual,
    )
    _write_csv(data_root / "m143_artifact_lag.csv", lags)
    (data_root / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    chart_map = [
        {
            "figure": "fig01_m7_esc50_denoising.png",
            "question": "How does M7-v2 denoise the sealed ESC-50 test set overall and by SNR?",
            "family": "comparison and ordered trend",
            "palette": "hard two-root cap",
        },
        {
            "figure": "fig02_m143_identity.png",
            "question": "Does M14.3-v2 preserve the frozen M7 mask, waveform and S1/S2 locations?",
            "family": "matrix audit",
            "palette": "single-root preferred",
        },
        {
            "figure": "fig03_sqi_comparison.png",
            "question": "Does correct IMU reduce SQI MAE within each LOSO fold?",
            "family": "paired comparison",
            "palette": "hard two-root cap",
        },
        {
            "figure": "fig04_artifact_alignment_counterfactual.png",
            "question": "Is artifact scoring sensitive to IMU timing and correspondence?",
            "family": "ordered trend and grouped comparison",
            "palette": "relaxed multi-category",
        },
    ]
    (data_root / "chart_map.json").write_text(
        json.dumps(chart_map, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    _style()
    _plot_m7(
        m7_overall,
        m7_by_snr,
        figure_root / "fig01_m7_esc50_denoising.png",
    )
    _plot_identity(
        identity,
        figure_root / "fig02_m143_identity.png",
    )
    _plot_sqi(
        folds,
        figure_root / "fig03_sqi_comparison.png",
    )
    _plot_artifact(
        folds,
        counterfactual,
        lags,
        figure_root / "fig04_artifact_alignment_counterfactual.png",
    )
    _write_report(report_root, summary, folds)
    return report_root


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Summarise the leakage-free strict ESC-50 M7-v2 and four-fold "
            "M14.3-v2 evaluations without touching legacy reports."
        )
    )
    parser.add_argument("--m7-root", type=Path, default=DEFAULT_M7_ROOT)
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=ROOT / "outputs",
    )
    parser.add_argument(
        "--report-root",
        type=Path,
        default=DEFAULT_REPORT_ROOT,
    )
    parser.add_argument(
        "--manifests-root",
        type=Path,
        default=DEFAULT_MANIFESTS_ROOT,
        help="Directory containing the strict ESC-50 leakage-audit JSON files.",
    )
    args = parser.parse_args()
    output = build_report(
        m7_root=args.m7_root,
        outputs_root=args.outputs_root,
        report_root=args.report_root,
        manifests_root=args.manifests_root,
    )
    print(f"Wrote strict v2 report to {output}")


if __name__ == "__main__":
    main()
