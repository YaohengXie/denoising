from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ecg_pcg_denoise.models import DenoisingModel
from ecg_pcg_denoise.train.dataset import get_mixed_window_dataset
from ecg_pcg_denoise.train.torch_audio import (
    apply_polar_complex_mask,
    complex_stft_features,
    istft_waveform,
    reconstruct_from_mag,
    stft_waveform,
)
from ecg_pcg_denoise.train.train_denoise import beat_to_frames, choose_device, s1s2_to_frames, stft_config
from ecg_pcg_denoise.utils.config import get_nested, load_config, require_nested
from ecg_pcg_denoise.utils.files import ensure_dir
from ecg_pcg_denoise.utils.metrics import evaluate_pair


def load_model(config: dict[str, Any], checkpoint_path: Path, device: torch.device) -> DenoisingModel:
    model = DenoisingModel.from_config(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def predict_batch(
    model: DenoisingModel,
    batch: dict[str, Any],
    config: dict[str, Any],
    device: torch.device,
    ecg_mode: str = "normal",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    n_fft, hop, win = stft_config(config)
    noisy = batch["noisy_pcg"].to(device)
    beat = batch["ecg_beat"].to(device)
    noisy_complex = stft_waveform(noisy, n_fft, hop, win)
    noisy_mag = noisy_complex.abs()
    beat_frames = beat_to_frames(beat, noisy_mag.shape[-1])
    if ecg_mode == "zero":
        beat_frames = torch.zeros_like(beat_frames)
    elif ecg_mode == "shift":
        beat_frames = torch.roll(beat_frames, shifts=max(1, beat_frames.shape[-1] // 4), dims=-1)
    model_input = complex_stft_features(noisy_complex) if model.use_complex_mask else noisy_mag
    output = model(model_input, beat_frames, imu_feat=None, modality_mask={"pcg": 1, "ecg": 1, "imu": 0})
    if model.use_complex_mask:
        estimate_complex = apply_polar_complex_mask(noisy_complex, output["mask"], output["phase_residual"])
        estimate = istft_waveform(estimate_complex, noisy.shape[-1], n_fft, hop, win)
    else:
        estimate_mag = output["mask"] * noisy_mag
        estimate = reconstruct_from_mag(noisy_complex, estimate_mag, noisy.shape[-1], n_fft, hop, win)
    phase_abs = output["phase_residual"].detach().abs().float().cpu().flatten(1)
    complex_imag_abs = output["complex_mask_imag"].detach().abs().float().cpu().flatten(1)
    phase_stats = {
        "phase_residual_abs_mean": phase_abs.mean(dim=1),
        "phase_residual_abs_p95": torch.quantile(phase_abs, 0.95, dim=1),
        "complex_mask_imag_abs_mean": complex_imag_abs.mean(dim=1),
    }
    return (
        estimate.detach().cpu(),
        output["sqi_score"].detach().cpu(),
        output["s1s2_prob"].detach().cpu(),
        phase_stats,
    )


def tensor_correlation(x: torch.Tensor, y: torch.Tensor) -> float:
    x_flat = x.flatten().float()
    y_flat = y.flatten().float()
    x_centered = x_flat - x_flat.mean()
    y_centered = y_flat - y_flat.mean()
    denom = torch.linalg.vector_norm(x_centered) * torch.linalg.vector_norm(y_centered)
    if float(denom) <= 1e-12:
        return 0.0
    return float(torch.sum(x_centered * y_centered) / denom)


def auxiliary_metrics(
    batch: dict[str, Any],
    s1s2_prob: torch.Tensor,
    sqi_score: torch.Tensor,
    device: torch.device,
) -> list[dict[str, float]]:
    s1s2_target = s1s2_to_frames(batch["s1s2_weak"].to(device), s1s2_prob.shape[-1]).detach().cpu()
    sqi_target = batch["sqi_target"].detach().cpu()
    prob = torch.clamp(s1s2_prob.float(), 1e-6, 1.0 - 1e-6)
    target = s1s2_target.float()
    sqi_error = sqi_score.float() - sqi_target.float()
    absolute_error = torch.abs(prob - target)
    bce = F.binary_cross_entropy(prob, target, reduction="none")
    return [
        {
            "sqi_target": float(sqi_target[index]),
            "sqi_abs_error": float(torch.abs(sqi_error)[index]),
            "sqi_sq_error": float(torch.square(sqi_error)[index]),
            "s1s2_mae": float(absolute_error[index].mean()),
            "s1_mae": float(absolute_error[index, 0].mean()),
            "s2_mae": float(absolute_error[index, 1].mean()),
            "s1s2_bce": float(bce[index].mean()),
            "s1s2_corr": tensor_correlation(prob[index], target[index]),
        }
        for index in range(prob.shape[0])
    ]


def summarize_coverage(rows: list[dict[str, float]], output_path: Path) -> None:
    sorted_rows = sorted(rows, key=lambda row: row["sqi_score"], reverse=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "coverage",
                "n",
                "mean_delta_snr",
                "mean_corr_estimate",
                "mean_sqi_score",
                "mean_sqi_target",
                "mean_sqi_abs_error",
                "mean_s1s2_mae",
            ],
        )
        writer.writeheader()
        for coverage in (1.0, 0.8, 0.6, 0.4):
            n = max(1, int(round(len(sorted_rows) * coverage)))
            kept = sorted_rows[:n]
            optional = {
                "mean_sqi_target": float(np.mean([r["sqi_target"] for r in kept])) if "sqi_target" in kept[0] else float("nan"),
                "mean_sqi_abs_error": float(np.mean([r["sqi_abs_error"] for r in kept]))
                if "sqi_abs_error" in kept[0]
                else float("nan"),
                "mean_s1s2_mae": float(np.mean([r["s1s2_mae"] for r in kept])) if "s1s2_mae" in kept[0] else float("nan"),
            }
            writer.writerow(
                {
                    "coverage": coverage,
                    "n": n,
                    "mean_delta_snr": float(np.mean([r["delta_snr"] for r in kept])),
                    "mean_corr_estimate": float(np.mean([r["corr_estimate"] for r in kept])),
                    "mean_sqi_score": float(np.mean([r["sqi_score"] for r in kept])),
                    **optional,
                }
            )


def evaluate_from_config(
    config_path: str | Path,
    checkpoint: str | Path | None = None,
    ecg_mode: str = "normal",
    disable_tf_conformer: bool = False,
    disable_axial_position: bool = False,
) -> None:
    config = load_config(config_path)
    device = choose_device(config)
    active_model_ablations = int(disable_tf_conformer) + int(disable_axial_position)
    if active_model_ablations > 1 or (active_model_ablations and ecg_mode != "normal"):
        raise ValueError("Run only one model or ECG ablation at a time.")
    if disable_tf_conformer:
        eval_dir_name = "eval_tfconformer_off"
    elif disable_axial_position:
        eval_dir_name = "eval_axial_position_off"
    else:
        eval_dir_name = "eval" if ecg_mode == "normal" else f"eval_ecg_{ecg_mode}"
    output_dir = ensure_dir(Path(require_nested(config, "paths.output_dir")) / eval_dir_name)
    checkpoint_path = Path(checkpoint) if checkpoint else Path(require_nested(config, "paths.output_dir")) / "checkpoints" / "best.pt"
    model = load_model(config, checkpoint_path, device)
    if disable_tf_conformer:
        if isinstance(model.tf_conformer, torch.nn.Identity):
            raise ValueError("--disable-tf-conformer requires a model with TF-Conformer enabled.")
        model.tf_conformer = torch.nn.Identity()
    if disable_axial_position:
        scale = getattr(model.transformer, "axial_position_scale", None)
        if scale is None:
            raise ValueError("--disable-axial-position requires axial position encoding.")
        with torch.no_grad():
            scale.zero_()
    if ecg_mode == "cross_off":
        projection = getattr(model.ecg_cross_attention, "output_projection", None)
        if projection is None:
            raise ValueError("ecg_mode=cross_off requires a model with ECGCrossAttention.")
        with torch.no_grad():
            projection.weight.zero_()
            projection.bias.zero_()
    batch_size = int(get_nested(config, "training.batch_size", 1))
    cache_in_memory = bool(get_nested(config, "training.cache_in_memory", False))
    loader = DataLoader(
        get_mixed_window_dataset(require_nested(config, "paths.windows_dir"), "test", cache_in_memory),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )

    rows: list[dict[str, float]] = []
    detail_path = output_dir / "window_metrics.csv"
    with detail_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "path",
            "snr_db",
            "sqi_score",
            "sqi_target",
            "sqi_abs_error",
            "sqi_sq_error",
            "s1s2_mae",
            "s1_mae",
            "s2_mae",
            "s1s2_bce",
            "s1s2_corr",
            "phase_residual_abs_mean",
            "phase_residual_abs_p95",
            "complex_mask_imag_abs_mean",
            "snr_noisy",
            "snr_estimate",
            "delta_snr",
            "si_sdr_noisy",
            "si_sdr_estimate",
            "delta_si_sdr",
            "corr_noisy",
            "corr_estimate",
            "log_spectral_distance",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        with torch.no_grad():
            for batch in loader:
                estimate, sqi_score, s1s2_prob, phase_stats = predict_batch(
                    model,
                    batch,
                    config,
                    device,
                    ecg_mode=ecg_mode,
                )
                auxiliary_rows = auxiliary_metrics(batch, s1s2_prob, sqi_score, device)
                for index in range(estimate.shape[0]):
                    noisy = batch["noisy_pcg"][index].numpy()
                    clean = batch["clean_pcg"][index].numpy()
                    metrics = evaluate_pair(noisy, estimate[index].numpy(), clean, int(batch["fs"][index]))
                    row = {
                        "path": batch["path"][index],
                        "snr_db": float(batch["snr_db"][index]),
                        "sqi_score": float(sqi_score[index]),
                        **auxiliary_rows[index],
                        **{name: float(values[index]) for name, values in phase_stats.items()},
                        **metrics,
                    }
                    writer.writerow(row)
                    rows.append({k: float(v) for k, v in row.items() if k != "path"})
    summarize_coverage(rows, output_dir / "sqi_coverage.csv")
    print(f"Wrote evaluation metrics to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained denoising model.")
    parser.add_argument("config", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--ecg-mode",
        choices=("normal", "cross_off", "zero", "shift"),
        default="normal",
        help="ECG ablation applied only during evaluation.",
    )
    parser.add_argument(
        "--disable-tf-conformer",
        action="store_true",
        help="Replace the trained TF-Conformer with an identity mapping during evaluation.",
    )
    parser.add_argument(
        "--disable-axial-position",
        action="store_true",
        help="Set the learned axial-position scale to zero during evaluation.",
    )
    args = parser.parse_args()
    evaluate_from_config(
        args.config,
        args.checkpoint,
        ecg_mode=args.ecg_mode,
        disable_tf_conformer=args.disable_tf_conformer,
        disable_axial_position=args.disable_axial_position,
    )


if __name__ == "__main__":
    main()
