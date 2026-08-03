from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
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
from ecg_pcg_denoise.utils.config import get_nested, load_config, require_nested
from ecg_pcg_denoise.utils.files import ensure_dir


class NonFiniteGradientError(RuntimeError):
    """Raised when a finite loss produces a non-finite gradient norm."""

    def __init__(self, batch_index: int, gradient_norm: float) -> None:
        self.batch_index = int(batch_index)
        self.gradient_norm = float(gradient_norm)
        super().__init__(
            "Non-finite gradient norm detected "
            f"at zero-based batch {self.batch_index}: {self.gradient_norm}"
        )


def configure_random_seed(config: dict[str, Any]) -> int:
    """Apply the configured project seed to every RNG used by training."""
    seed = int(get_nested(config, "project.seed", 1337))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    deterministic = bool(get_nested(config, "training.deterministic", False))
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic
    return seed


def choose_device(config: dict[str, Any]) -> torch.device:
    requested = str(get_nested(config, "training.device", "auto"))
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def stft_config(config: dict[str, Any]) -> tuple[int, int, int]:
    n_fft = int(require_nested(config, "stft.n_fft"))
    hop = int(require_nested(config, "stft.hop_length"))
    win = int(require_nested(config, "stft.win_length"))
    return n_fft, hop, win


def beat_to_frames(beat: Tensor, frames: int) -> Tensor:
    return F.interpolate(beat[:, None, :], size=frames, mode="linear", align_corners=False).squeeze(1)


def s1s2_to_frames(labels: Tensor, frames: int) -> Tensor:
    return F.interpolate(labels, size=frames, mode="linear", align_corners=False)


def matches_prefix(name: str, prefixes: list[str] | tuple[str, ...]) -> bool:
    return any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes)


def load_initial_checkpoint(model: DenoisingModel, config: dict[str, Any], device: torch.device) -> None:
    checkpoint_value = get_nested(config, "training.init_checkpoint", None)
    if not checkpoint_value:
        return
    checkpoint_path = Path(str(checkpoint_value))
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    strict = bool(get_nested(config, "training.init_strict", True))
    if strict:
        model.load_state_dict(checkpoint["model"], strict=True)
        print(f"Loaded initial checkpoint from {checkpoint_path}")
        return

    current_state = model.state_dict()
    compatible_state = {
        key: value
        for key, value in checkpoint["model"].items()
        if key in current_state and current_state[key].shape == value.shape
    }
    current_state.update(compatible_state)
    model.load_state_dict(current_state, strict=True)
    skipped = len(checkpoint["model"]) - len(compatible_state)
    print(f"Loaded {len(compatible_state)} tensors from {checkpoint_path}; skipped {skipped} incompatible tensors")


def load_teacher_model(config: dict[str, Any], device: torch.device) -> DenoisingModel | None:
    checkpoint_value = get_nested(config, "training.teacher_checkpoint", None)
    if not checkpoint_value:
        return None
    checkpoint_path = Path(str(checkpoint_value))
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    teacher_config = checkpoint.get("config", config)
    teacher = DenoisingModel.from_config(teacher_config).to(device)
    teacher.load_state_dict(checkpoint["model"], strict=True)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    print(f"Loaded teacher checkpoint from {checkpoint_path}")
    return teacher


def configure_trainable_parameters(model: DenoisingModel, config: dict[str, Any]) -> list[torch.nn.Parameter]:
    trainable_modules = [str(name) for name in get_nested(config, "training.trainable_modules", [])]
    freeze_modules = [str(name) for name in get_nested(config, "training.freeze_modules", [])]

    if trainable_modules and freeze_modules:
        raise ValueError("Use either training.trainable_modules or training.freeze_modules, not both.")

    if trainable_modules:
        for name, parameter in model.named_parameters():
            parameter.requires_grad = matches_prefix(name, trainable_modules)
        frozen_prefixes = [name for name, _ in model.named_children() if name not in trainable_modules]
    elif freeze_modules:
        for name, parameter in model.named_parameters():
            parameter.requires_grad = not matches_prefix(name, freeze_modules)
        frozen_prefixes = freeze_modules
    else:
        frozen_prefixes = []

    model._frozen_module_prefixes = tuple(frozen_prefixes)  # type: ignore[attr-defined]
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("No trainable parameters remain after applying freeze configuration.")

    trainable_count = sum(parameter.numel() for parameter in trainable)
    total_count = sum(parameter.numel() for parameter in model.parameters())
    if trainable_modules or freeze_modules:
        print(f"Trainable parameters: {trainable_count}/{total_count}")
    return trainable


def keep_frozen_modules_eval(model: DenoisingModel) -> None:
    frozen_prefixes = getattr(model, "_frozen_module_prefixes", ())
    if not frozen_prefixes:
        return
    for name, module in model.named_modules():
        if name and matches_prefix(name, frozen_prefixes):
            module.eval()


def forward_batch(
    model: DenoisingModel,
    batch: dict[str, Tensor],
    config: dict[str, Any],
    device: torch.device,
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    n_fft, hop, win = stft_config(config)
    noisy = batch["noisy_pcg"].to(device)
    clean = batch["clean_pcg"].to(device)
    beat = batch["ecg_beat"].to(device)
    noisy_complex = stft_waveform(noisy, n_fft, hop, win)
    clean_complex = stft_waveform(clean, n_fft, hop, win)
    noisy_mag = noisy_complex.abs()
    clean_mag = clean_complex.abs()
    beat_frames = beat_to_frames(beat, noisy_mag.shape[-1])
    model_input = complex_stft_features(noisy_complex) if model.use_complex_mask else noisy_mag
    output = model(
        pcg_stft=model_input,
        ecg_beat=beat_frames,
        imu_feat=None,
        modality_mask={"pcg": 1, "ecg": 1, "imu": 0},
    )
    if model.use_complex_mask:
        estimate_complex = apply_polar_complex_mask(noisy_complex, output["mask"], output["phase_residual"])
        estimate_mag = estimate_complex.abs()
        estimate = istft_waveform(estimate_complex, noisy.shape[-1], n_fft, hop, win)
    else:
        estimate_mag = output["mask"] * noisy_mag
        estimate_complex = torch.polar(estimate_mag.clamp_min(0.0), torch.angle(noisy_complex))
        estimate = reconstruct_from_mag(noisy_complex, estimate_mag, noisy.shape[-1], n_fft, hop, win)
    tensors = {
        "noisy": noisy,
        "clean": clean,
        "estimate": estimate,
        "noisy_mag": noisy_mag,
        "clean_mag": clean_mag,
        "estimate_mag": estimate_mag,
        "noisy_complex": noisy_complex,
        "clean_complex": clean_complex,
        "estimate_complex": estimate_complex,
        "s1s2_target": s1s2_to_frames(batch["s1s2_weak"].to(device), noisy_mag.shape[-1]),
        "sqi_target": batch["sqi_target"].to(device),
    }
    if float(get_nested(config, "training.loss.clean_identity_smooth_l1", 0.0)) > 0:
        clean_model_input = complex_stft_features(clean_complex) if model.use_complex_mask else clean_mag
        clean_output = model(
            pcg_stft=clean_model_input,
            ecg_beat=beat_frames,
            imu_feat=None,
            modality_mask={"pcg": 1, "ecg": 1, "imu": 0},
        )
        if model.use_complex_mask:
            identity_complex = apply_polar_complex_mask(
                clean_complex,
                clean_output["mask"],
                clean_output["phase_residual"],
            )
            identity_estimate = istft_waveform(identity_complex, clean.shape[-1], n_fft, hop, win)
        else:
            identity_mag = clean_output["mask"] * clean_mag
            identity_estimate = reconstruct_from_mag(
                clean_complex,
                identity_mag,
                clean.shape[-1],
                n_fft,
                hop,
                win,
            )
        tensors["identity_estimate"] = identity_estimate
    return output, tensors


def compute_loss(
    output: dict[str, Tensor],
    tensors: dict[str, Tensor],
    config: dict[str, Any],
    teacher_output: dict[str, Tensor] | None = None,
    teacher_tensors: dict[str, Tensor] | None = None,
) -> Tensor:
    weights = get_nested(config, "training.loss", {})
    loss = torch.zeros((), device=tensors["clean"].device)
    s1s2_alpha = float(weights.get("s1s2_weight_alpha", 3.0))
    smooth_l1_beta = float(weights.get("smooth_l1_beta", 1.0))
    if float(weights.get("waveform_l1", 0.0)) > 0:
        loss = loss + float(weights["waveform_l1"]) * F.l1_loss(tensors["estimate"], tensors["clean"])
    if float(weights.get("waveform_smooth_l1", 0.0)) > 0:
        loss = loss + float(weights["waveform_smooth_l1"]) * F.smooth_l1_loss(
            tensors["estimate"],
            tensors["clean"],
            beta=smooth_l1_beta,
        )
    if float(weights.get("s1s2_weighted_waveform_l1", 0.0)) > 0:
        heart = tensors["s1s2_target"].amax(dim=1, keepdim=True)
        heart = F.interpolate(heart, size=tensors["estimate"].shape[-1], mode="linear", align_corners=False).squeeze(1)
        weight = 1.0 + s1s2_alpha * heart
        weighted_error = weight * torch.abs(tensors["estimate"] - tensors["clean"])
        loss = loss + float(weights["s1s2_weighted_waveform_l1"]) * (
            weighted_error.sum() / (weight.sum() + 1e-8)
        )
    if float(weights.get("s1s2_weighted_waveform_smooth_l1", 0.0)) > 0:
        heart = tensors["s1s2_target"].amax(dim=1, keepdim=True)
        heart = F.interpolate(heart, size=tensors["estimate"].shape[-1], mode="linear", align_corners=False).squeeze(1)
        weight = 1.0 + s1s2_alpha * heart
        weighted_error = weight * F.smooth_l1_loss(
            tensors["estimate"],
            tensors["clean"],
            beta=smooth_l1_beta,
            reduction="none",
        )
        loss = loss + float(weights["s1s2_weighted_waveform_smooth_l1"]) * (
            weighted_error.sum() / (weight.sum() + 1e-8)
        )
    if float(weights.get("mag_l1", 0.0)) > 0:
        loss = loss + float(weights["mag_l1"]) * F.l1_loss(
            torch.log1p(tensors["estimate_mag"]), torch.log1p(tensors["clean_mag"])
        )
    if float(weights.get("s1s2_weighted_mag_l1", 0.0)) > 0:
        heart = tensors["s1s2_target"].amax(dim=1)[:, None, :]
        weight = 1.0 + s1s2_alpha * heart
        mag_error = torch.abs(torch.log1p(tensors["estimate_mag"]) - torch.log1p(tensors["clean_mag"]))
        loss = loss + float(weights["s1s2_weighted_mag_l1"]) * (
            (weight * mag_error).sum() / (weight.sum() * mag_error.shape[1] + 1e-8)
        )
    if float(weights.get("fft_mag_mse", 0.0)) > 0:
        estimate_fft_mag = torch.fft.rfft(tensors["estimate"], dim=-1, norm="ortho").abs()
        clean_fft_mag = torch.fft.rfft(tensors["clean"], dim=-1, norm="ortho").abs()
        loss = loss + float(weights["fft_mag_mse"]) * F.mse_loss(estimate_fft_mag, clean_fft_mag)
    if float(weights.get("clean_identity_smooth_l1", 0.0)) > 0:
        if "identity_estimate" not in tensors:
            raise RuntimeError("clean_identity_smooth_l1 requires an identity forward pass.")
        loss = loss + float(weights["clean_identity_smooth_l1"]) * F.smooth_l1_loss(
            tensors["identity_estimate"],
            tensors["clean"],
            beta=smooth_l1_beta,
        )
    if float(weights.get("complex_l1", 0.0)) > 0:
        estimate_ri = torch.view_as_real(tensors["estimate_complex"])
        clean_ri = torch.view_as_real(tensors["clean_complex"])
        loss = loss + float(weights["complex_l1"]) * F.l1_loss(estimate_ri, clean_ri)
    if float(weights.get("phase_cosine", 0.0)) > 0:
        target_residual = torch.angle(tensors["clean_complex"] * tensors["noisy_complex"].conj()).detach()
        phase_error = 1.0 - torch.cos(output["phase_residual"] - target_residual)
        valid_noisy = (tensors["noisy_mag"] > 1e-6).to(tensors["clean_mag"].dtype)
        phase_weight = (tensors["clean_mag"] * valid_noisy).detach()
        weighted_phase_error = (phase_weight * phase_error).sum() / (phase_weight.sum() + 1e-8)
        loss = loss + float(weights["phase_cosine"]) * weighted_phase_error
    if float(weights.get("s1s2_bce", 0.0)) > 0:
        loss = loss + float(weights["s1s2_bce"]) * F.binary_cross_entropy(
            output["s1s2_prob"], tensors["s1s2_target"]
        )
    if float(weights.get("sqi_mse", 0.0)) > 0:
        loss = loss + float(weights["sqi_mse"]) * F.mse_loss(
            output["sqi_score"], tensors["sqi_target"]
        )
    if teacher_output is not None and teacher_tensors is not None:
        if float(weights.get("teacher_mask_l1", 0.0)) > 0:
            loss = loss + float(weights["teacher_mask_l1"]) * F.l1_loss(
                output["mask"], teacher_output["mask"]
            )
        if float(weights.get("teacher_mag_l1", 0.0)) > 0:
            loss = loss + float(weights["teacher_mag_l1"]) * F.l1_loss(
                torch.log1p(tensors["estimate_mag"]), torch.log1p(teacher_tensors["estimate_mag"])
            )
        if float(weights.get("teacher_waveform_l1", 0.0)) > 0:
            loss = loss + float(weights["teacher_waveform_l1"]) * F.l1_loss(
                tensors["estimate"], teacher_tensors["estimate"]
            )
    return loss


def run_epoch(
    model: DenoisingModel,
    loader: DataLoader,
    config: dict[str, Any],
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    teacher_model: DenoisingModel | None = None,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    if is_train:
        keep_frozen_modules_eval(model)
    total = 0.0
    count = 0
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch_index, batch in enumerate(loader):
            output, tensors = forward_batch(model, batch, config, device)
            teacher_output = None
            teacher_tensors = None
            if teacher_model is not None:
                with torch.no_grad():
                    teacher_output, teacher_tensors = forward_batch(teacher_model, batch, config, device)
            loss = compute_loss(output, tensors, config, teacher_output, teacher_tensors)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss detected; check the active loss terms and input window.")
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient_norm = nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=5.0,
                    error_if_nonfinite=False,
                )
                if not torch.isfinite(gradient_norm):
                    optimizer.zero_grad(set_to_none=True)
                    raise NonFiniteGradientError(
                        batch_index=batch_index,
                        gradient_norm=float(gradient_norm.detach().cpu()),
                    )
                optimizer.step()
            total += float(loss.detach().cpu()) * tensors["clean"].shape[0]
            count += tensors["clean"].shape[0]
    return total / max(1, count)


def build_optimizer(model: DenoisingModel, config: dict[str, Any], default_lr: float) -> torch.optim.Optimizer:
    module_lrs = get_nested(config, "training.module_lrs", {})
    if not isinstance(module_lrs, dict):
        raise ValueError("training.module_lrs must be a mapping of module prefixes to learning rates.")
    ordered_lrs = sorted(
        ((str(prefix), float(value)) for prefix, value in module_lrs.items()),
        key=lambda item: len(item[0]),
        reverse=True,
    )
    grouped: dict[float, list[torch.nn.Parameter]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        parameter_lr = default_lr
        for prefix, configured_lr in ordered_lrs:
            if matches_prefix(name, [prefix]):
                parameter_lr = configured_lr
                break
        grouped.setdefault(parameter_lr, []).append(parameter)
    weight_decay = float(get_nested(config, "training.weight_decay", 0.01))
    parameter_groups = [
        {"params": parameters, "lr": group_lr, "initial_lr": group_lr}
        for group_lr, parameters in grouped.items()
    ]
    if ordered_lrs:
        summary = {group_lr: sum(parameter.numel() for parameter in parameters) for group_lr, parameters in grouped.items()}
        print(f"Optimizer parameter groups: {summary}")
    return torch.optim.AdamW(parameter_groups, lr=default_lr, weight_decay=weight_decay)


def set_epoch_learning_rates(
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    epoch: int,
    epochs: int,
) -> float:
    scheduler_name = str(get_nested(config, "training.scheduler.name", "none"))
    multiplier = 1.0
    if scheduler_name == "warmup_cosine":
        warmup_epochs = int(get_nested(config, "training.scheduler.warmup_epochs", 0))
        min_lr_ratio = float(get_nested(config, "training.scheduler.min_lr_ratio", 0.0))
        if warmup_epochs > 0 and epoch <= warmup_epochs:
            multiplier = epoch / warmup_epochs
        else:
            cosine_epochs = max(1, epochs - warmup_epochs)
            progress = min(1.0, max(0.0, (epoch - warmup_epochs) / cosine_epochs))
            multiplier = min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress))
    elif scheduler_name != "none":
        raise ValueError(f"Unsupported training.scheduler.name: {scheduler_name}")
    for group in optimizer.param_groups:
        group["lr"] = float(group["initial_lr"]) * multiplier
    return multiplier


def train_from_config(config_path: str | Path) -> None:
    config = load_config(config_path)
    seed = configure_random_seed(config)
    device = choose_device(config)
    output_dir = ensure_dir(require_nested(config, "paths.output_dir"))
    checkpoint_dir = ensure_dir(output_dir / "checkpoints")
    windows_dir = require_nested(config, "paths.windows_dir")
    batch_size = int(require_nested(config, "training.batch_size"))
    num_workers = int(get_nested(config, "training.num_workers", 0))
    epochs = int(require_nested(config, "training.epochs"))
    lr = float(require_nested(config, "training.lr"))
    cache_in_memory = bool(get_nested(config, "training.cache_in_memory", False))
    train_generator = torch.Generator()
    train_generator.manual_seed(seed)

    train_loader = DataLoader(
        get_mixed_window_dataset(windows_dir, "train", cache_in_memory),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        generator=train_generator,
    )
    val_loader = DataLoader(
        get_mixed_window_dataset(windows_dir, "val", cache_in_memory),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    model = DenoisingModel.from_config(config).to(device)
    load_initial_checkpoint(model, config, device)
    trainable_parameters = configure_trainable_parameters(model, config)
    teacher_model = load_teacher_model(config, device)
    del trainable_parameters
    optimizer = build_optimizer(model, config, lr)
    history_path = output_dir / "training_history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "epoch",
                "train_loss",
                "val_loss",
                "learning_rates",
                "lr_multiplier",
                "improved",
            ),
        )
        writer.writeheader()

    best_val = float("inf")
    if bool(get_nested(config, "training.include_initial_in_best", False)):
        initial_val = run_epoch(model, val_loader, config, device, optimizer=None, teacher_model=teacher_model)
        best_val = initial_val
        print(f"epoch=000 train_loss=nan val_loss={initial_val:.6f}")
        torch.save(
            {
                "model": model.state_dict(),
                "config": config,
                "epoch": 0,
                "val_loss": initial_val,
                "seed": seed,
            },
            checkpoint_dir / "best.pt",
        )
        with history_path.open("a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(
                [0, "nan", initial_val, "", "", True]
            )
    early_stopping_patience = int(get_nested(config, "training.early_stopping_patience", 0))
    early_stopping_min_delta = float(get_nested(config, "training.early_stopping_min_delta", 0.0))
    epochs_without_improvement = 0
    last_epoch = 0
    for epoch in range(1, epochs + 1):
        last_epoch = epoch
        lr_multiplier = set_epoch_learning_rates(optimizer, config, epoch, epochs)
        train_loss = run_epoch(model, train_loader, config, device, optimizer, teacher_model)
        val_loss = run_epoch(model, val_loader, config, device, optimizer=None, teacher_model=teacher_model)
        current_lrs = sorted({float(group["lr"]) for group in optimizer.param_groups})
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
            f"lr={current_lrs} lr_multiplier={lr_multiplier:.6f}"
        )
        improved = val_loss < best_val - early_stopping_min_delta
        if improved:
            best_val = val_loss
            epochs_without_improvement = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "seed": seed,
                },
                checkpoint_dir / "best.pt",
            )
        else:
            epochs_without_improvement += 1
        with history_path.open("a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(
                [
                    epoch,
                    train_loss,
                    val_loss,
                    "|".join(f"{value:.12g}" for value in current_lrs),
                    lr_multiplier,
                    improved,
                ]
            )
        if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
            print(f"Early stopping after {early_stopping_patience} epochs without validation improvement.")
            break
    torch.save(
        {
            "model": model.state_dict(),
            "config": config,
            "epoch": last_epoch,
            "seed": seed,
        },
        checkpoint_dir / "last.pt",
    )
    print(f"Saved checkpoints to {checkpoint_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ECG-guided PCG denoising model.")
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    train_from_config(args.config)


if __name__ == "__main__":
    main()
