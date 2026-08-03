from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any


def _as_float(row: dict[str, str], name: str) -> float:
    value = row.get(name, "")
    return float(value) if value not in {"", None} else float("nan")


def select_m142_checkpoint(
    output_dir: str | Path,
    sqi_tolerance: float = 0.0,
) -> dict[str, Any]:
    """Select the lowest validation loss under hard M7 safety gates."""

    root = Path(output_dir)
    history_path = root / "training_history.csv"
    with history_path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty M14.2 history: {history_path}")

    selection_rows: list[dict[str, Any]] = []
    candidates: list[tuple[float, int, Path]] = []
    for row in rows:
        epoch = int(row["epoch"])
        checkpoint = root / "checkpoints" / f"epoch_{epoch:03d}.pt"
        sqi_mae = _as_float(row, "val_metric_sqi_mae")
        base_sqi_mae = _as_float(row, "val_metric_base_sqi_mae")
        mask_delta = _as_float(row, "val_metric_mask_delta")
        s1s2_delta = _as_float(row, "val_metric_s1s2_location_delta")
        val_loss = _as_float(row, "val_loss")
        reasons: list[str] = []
        if not checkpoint.exists():
            reasons.append("checkpoint_missing")
        required_metrics = {
            "val_loss": val_loss,
            "sqi_mae": sqi_mae,
            "base_sqi_mae": base_sqi_mae,
            "mask_delta": mask_delta,
            "s1s2_delta": s1s2_delta,
        }
        nonfinite = [
            name for name, value in required_metrics.items() if not math.isfinite(value)
        ]
        if nonfinite:
            reasons.append("nonfinite_metrics:" + ",".join(nonfinite))
        if mask_delta != 0.0:
            reasons.append("mask_not_identical")
        if s1s2_delta != 0.0:
            reasons.append("s1s2_location_not_identical")
        if sqi_mae > base_sqi_mae + float(sqi_tolerance):
            reasons.append("sqi_inferior_to_m7")
        eligible = not reasons
        if eligible:
            candidates.append((val_loss, epoch, checkpoint))
        selection_rows.append(
            {
                "epoch": epoch,
                "val_loss": val_loss,
                "val_sqi_mae": sqi_mae,
                "val_m7_sqi_mae": base_sqi_mae,
                "val_sqi_improvement": base_sqi_mae - sqi_mae,
                "mask_max_abs_vs_m7": mask_delta,
                "s1s2_max_abs_vs_m7": s1s2_delta,
                "eligible": eligible,
                "failed_constraints": ";".join(reasons),
                "selected": False,
            }
        )
    if not candidates:
        raise RuntimeError("No M14.2 checkpoint passed the M7 safety gates.")
    _, selected_epoch, selected_path = min(candidates)
    for row in selection_rows:
        row["selected"] = int(row["epoch"]) == selected_epoch

    best_safe = root / "checkpoints" / "best_safe.pt"
    best = root / "checkpoints" / "best.pt"
    shutil.copy2(selected_path, best_safe)
    shutil.copy2(selected_path, best)
    csv_path = root / "checkpoint_selection.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selection_rows[0]))
        writer.writeheader()
        writer.writerows(selection_rows)
    selected_row = next(row for row in selection_rows if row["selected"])
    summary = {
        "selection_rule": (
            "minimum validation total loss among checkpoints with exact mask "
            "and S1/S2 identity and SQI MAE no worse than frozen M7"
        ),
        "sqi_noninferiority_tolerance": float(sqi_tolerance),
        "selected_epoch": selected_epoch,
        "selected_checkpoint": str(best.resolve()),
        "selected_metrics": selected_row,
        "eligible_epochs": [
            int(row["epoch"]) for row in selection_rows if row["eligible"]
        ],
    }
    (root / "checkpoint_selection_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Select an auxiliary IMU checkpoint under hard M7 safety gates "
            "(used by M14.3-v2)."
        )
    )
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--sqi-tolerance", type=float, default=0.0)
    args = parser.parse_args()
    result = select_m142_checkpoint(args.output_dir, args.sqi_tolerance)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
