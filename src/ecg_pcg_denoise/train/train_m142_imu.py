from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from ecg_pcg_denoise.models import DenoisingModel
from ecg_pcg_denoise.train.m14_imu_dataset import M14IMUDataset
from ecg_pcg_denoise.train.m142_imu_runtime import (
    forward_m142_batch,
    m142_auxiliary_loss,
    m142_fallback_loss,
    m142_metrics,
    reforward_m142_auxiliary,
)
from ecg_pcg_denoise.train.train_denoise import choose_device
from ecg_pcg_denoise.utils.config import get_nested, load_config, require_nested
from ecg_pcg_denoise.utils.files import ensure_dir


AUXILIARY_PREFIX = "imu_aux_adapter."


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_m7_into_m142(
    model: DenoisingModel,
    checkpoint_path: str | Path,
    device: torch.device,
) -> dict[str, Any]:
    """Load M7 strictly except for the newly introduced auxiliary adapter."""

    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=True
    )
    if "model" not in checkpoint:
        raise KeyError(f"Checkpoint has no model state: {checkpoint_path}")
    incompatible = model.load_state_dict(checkpoint["model"], strict=False)
    unexpected = sorted(incompatible.unexpected_keys)
    missing = sorted(incompatible.missing_keys)
    expected_missing = sorted(
        name for name in model.state_dict() if name.startswith(AUXILIARY_PREFIX)
    )
    if unexpected:
        raise RuntimeError(f"Unexpected M7 checkpoint keys: {unexpected}")
    if not expected_missing:
        raise RuntimeError("M14.2 config did not construct imu_aux_adapter.")
    if missing != expected_missing:
        raise RuntimeError(
            "M7 to M14.2 load was not strict outside imu_aux_adapter. "
            f"missing={missing}, expected={expected_missing}"
        )
    return checkpoint


def freeze_m7_train_aux_only(
    model: DenoisingModel,
) -> tuple[list[nn.Parameter], list[str]]:
    for parameter in model.parameters():
        parameter.requires_grad = False
    adapter = getattr(model, "imu_aux_adapter", None)
    if adapter is None:
        raise RuntimeError(
            "M14.2 requires model.use_imu_aux=true and model.imu_aux_adapter."
        )
    for parameter in adapter.parameters():
        parameter.requires_grad = True
    names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    if not names or any(not name.startswith(AUXILIARY_PREFIX) for name in names):
        raise RuntimeError(f"Invalid M14.2 trainable parameter scope: {names}")
    return [parameter for parameter in adapter.parameters()], names


def _frozen_state_snapshot(model: DenoisingModel) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if not name.startswith(AUXILIARY_PREFIX)
    }


def _audit_frozen_state(
    model: DenoisingModel,
    reference: dict[str, torch.Tensor],
) -> None:
    changed = [
        name
        for name, value in model.state_dict().items()
        if name in reference
        and not torch.equal(value.detach().cpu(), reference[name])
    ]
    if changed:
        raise RuntimeError(f"Frozen M7 state changed during M14.2 training: {changed}")


def _fallback_modes(config: dict[str, Any]) -> tuple[str, ...]:
    configured = get_nested(
        config,
        "training.fallback.modes",
        ["missing", "shuffle"],
    )
    modes = tuple(str(mode) for mode in configured)
    invalid = set(modes) - {"missing", "shuffle", "shift"}
    if invalid:
        raise ValueError(f"Unsupported M14.2 fallback modes: {sorted(invalid)}")
    return modes


def run_epoch(
    model: DenoisingModel,
    loader: DataLoader,
    config: dict[str, Any],
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
) -> tuple[float, dict[str, float]]:
    is_train = optimizer is not None
    model.eval()
    adapter = getattr(model, "imu_aux_adapter", None)
    if adapter is None:
        raise RuntimeError("M14.2 auxiliary adapter is unavailable.")
    adapter.train(is_train)
    amp_enabled = device.type == "cuda" and bool(
        get_nested(config, "training.amp", True)
    )
    fallback_modes = _fallback_modes(config)
    fallback_config = get_nested(config, "training.fallback", {})

    loss_total = 0.0
    sample_count = 0
    skipped_nonfinite_samples = 0
    aggregate: dict[str, float] = {}
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch_index, batch in enumerate(loader):
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                output, tensors, targets = forward_m142_batch(
                    model,
                    batch,
                    config,
                    device,
                    imu_mode="correct",
                )
                loss, components = m142_auxiliary_loss(
                    output,
                    targets,
                    config,
                    return_components=True,
                )
                fallback_outputs: dict[str, dict[str, torch.Tensor]] = {}
                for mode in fallback_modes:
                    shift_frames: int | None = None
                    shift_direction = 1
                    if mode == "shift":
                        configured_shifts = get_nested(
                            config,
                            "training.fallback.shift_seconds",
                            [0.25],
                        )
                        if isinstance(configured_shifts, (int, float)):
                            shift_seconds = [float(configured_shifts)]
                        else:
                            shift_seconds = [
                                float(value) for value in configured_shifts
                            ]
                        if not shift_seconds:
                            raise ValueError(
                                "training.fallback.shift_seconds is empty."
                            )
                        selected_shift = shift_seconds[
                            batch_index % len(shift_seconds)
                        ]
                        window_seconds = float(
                            require_nested(config, "data.window_sec")
                        )
                        shift_frames = max(
                            1,
                            int(
                                round(
                                    selected_shift
                                    * int(batch["imu_feat"].shape[-1])
                                    / window_seconds
                                )
                            ),
                        )
                        shift_direction = (
                            1
                            if (batch_index // len(shift_seconds)) % 2 == 0
                            else -1
                        )
                    fallback_output = reforward_m142_auxiliary(
                        model,
                        output,
                        batch,
                        device,
                        mode,
                        shift_frames=shift_frames,
                        shift_direction=shift_direction,
                    )
                    fallback_outputs[mode] = fallback_output
                    fallback_loss, fallback_components = m142_fallback_loss(
                        fallback_output,
                        mode,
                        config,
                        reference_output=output,
                        targets=targets,
                        return_components=True,
                    )
                    loss = loss + float(
                        fallback_config.get(f"{mode}_weight", 1.0)
                    ) * fallback_loss
                    components.update(fallback_components)

            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite M14.2 loss detected.")
            if is_train:
                if scaler is None:
                    raise RuntimeError("M14.2 training requires a gradient scaler.")
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                gradient_norm = nn.utils.clip_grad_norm_(
                    [
                        parameter
                        for parameter in model.parameters()
                        if parameter.requires_grad
                    ],
                    max_norm=float(
                        get_nested(config, "training.gradient_clip", 5.0)
                    ),
                    error_if_nonfinite=False,
                )
                if torch.isfinite(gradient_norm):
                    scaler.step(optimizer)
                else:
                    # AMP overflow is recoverable: GradScaler has already
                    # recorded the non-finite gradients in ``unscale_``.
                    # Calling step lets it skip the optimizer update, and
                    # update lowers the scale for the next batch.
                    if scaler.is_enabled():
                        scaler.step(optimizer)
                    skipped_nonfinite_samples += int(tensors["clean"].shape[0])
                scaler.update()

            batch_metrics = m142_metrics(output, targets, fallback_outputs)
            batch_size = int(tensors["clean"].shape[0])
            loss_total += float(loss.detach().cpu()) * batch_size
            sample_count += batch_size
            values = {
                **components,
                **{f"metric_{name}": value for name, value in batch_metrics.items()},
            }
            for name, value in values.items():
                aggregate[name] = aggregate.get(name, 0.0) + (
                    float(value.detach().cpu()) * batch_size
                )

    denominator = max(1, sample_count)
    aggregate["skipped_nonfinite_fraction"] = float(skipped_nonfinite_samples)
    return loss_total / denominator, {
        name: value / denominator for name, value in aggregate.items()
    }


def _checkpoint_payload(
    model: DenoisingModel,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    epoch: int,
    val_loss: float,
    base_checkpoint: str,
    base_checkpoint_sha256: str,
    fold: str,
    trainable_names: list[str],
) -> dict[str, Any]:
    adapter = getattr(model, "imu_aux_adapter", None)
    if adapter is None:
        raise RuntimeError("Cannot checkpoint an absent M14.2 adapter.")
    return {
        "artifact_type": "m143_v2_full_training_checkpoint",
        "fold": fold,
        "model": model.state_dict(),
        "imu_aux_adapter": adapter.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config,
        "epoch": int(epoch),
        "val_loss": float(val_loss),
        "base_m7_checkpoint": base_checkpoint,
        "base_checkpoint_sha256": base_checkpoint_sha256,
        "trainable_scope": "imu_aux_adapter_only",
        "trainable_parameter_names": trainable_names,
    }


def _write_history(path: Path, history: list[dict[str, Any]]) -> None:
    fieldnames = list(
        dict.fromkeys(name for row in history for name in row)
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def train_m142_from_config(
    config_path: str | Path,
    fold_override: str | None = None,
    output_dir_override: str | Path | None = None,
) -> None:
    config = load_config(config_path)
    if fold_override is not None:
        config.setdefault("m14_imu", {})["fold"] = str(fold_override)
    if output_dir_override is not None:
        config.setdefault("paths", {})["output_dir"] = str(output_dir_override)

    device = choose_device(config)
    output_dir = ensure_dir(require_nested(config, "paths.output_dir"))
    checkpoint_dir = ensure_dir(output_dir / "checkpoints")
    data_config = require_nested(config, "paths.data_config")
    fold = str(require_nested(config, "m14_imu.fold"))
    base_checkpoint = str(require_nested(config, "paths.base_m7_checkpoint"))
    base_checkpoint_sha256 = _sha256(Path(base_checkpoint))
    seed = int(get_nested(config, "project.seed", 1337))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_dataset = M14IMUDataset.from_config(data_config, "train", fold)
    val_dataset = M14IMUDataset.from_config(data_config, "val", fold)
    validation_subjects = {
        str(value)
        for value in get_nested(
            config,
            "training.validation_bsslab_subjects",
            [],
        )
    }
    if validation_subjects:
        if val_dataset.fixed_rows is None:
            raise RuntimeError("M14.2 validation rows are unavailable.")
        val_dataset.fixed_rows = [
            row
            for row in val_dataset.fixed_rows
            if str(row["bsslab_subject_id"]) in validation_subjects
        ]
        if not val_dataset.fixed_rows:
            raise RuntimeError(
                "No validation windows remain for BSSLAB subjects "
                f"{sorted(validation_subjects)}."
            )
    batch_size = int(require_nested(config, "training.batch_size"))
    num_workers = int(get_nested(config, "training.num_workers", 0))
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )

    model = DenoisingModel.from_config(config)
    if not isinstance(model, DenoisingModel):
        raise TypeError("M14.2 requires the U-Net DenoisingModel.")
    model = model.to(device)
    base = load_m7_into_m142(model, base_checkpoint, device)
    trainable, trainable_names = freeze_m7_train_aux_only(model)
    frozen_reference = _frozen_state_snapshot(model)

    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(parameter.numel() for parameter in trainable)
    print(
        f"device={device} fold={fold} train={len(train_dataset)} "
        f"val={len(val_dataset)} trainable={trainable_params}/{total_params}"
    )
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(require_nested(config, "training.lr")),
        weight_decay=float(get_nested(config, "training.weight_decay", 0.01)),
    )
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=device.type == "cuda"
        and bool(get_nested(config, "training.amp", True)),
    )

    initial_val, initial_metrics = run_epoch(
        model,
        val_loader,
        config,
        device,
        optimizer=None,
        scaler=None,
    )
    best_val = initial_val
    best_epoch = 0
    history: list[dict[str, Any]] = [
        {
            "epoch": 0,
            "train_loss": float("nan"),
            "val_loss": initial_val,
            **{f"val_{name}": value for name, value in initial_metrics.items()},
        }
    ]
    initial_payload = _checkpoint_payload(
        model,
        optimizer,
        config,
        0,
        initial_val,
        base_checkpoint,
        base_checkpoint_sha256,
        fold,
        trainable_names,
    )
    torch.save(initial_payload, checkpoint_dir / "best.pt")
    torch.save(initial_payload, checkpoint_dir / "epoch_000.pt")
    print(f"epoch=000 train_loss=nan val_loss={initial_val:.8f}")

    epochs = int(require_nested(config, "training.epochs"))
    patience = int(get_nested(config, "training.early_stopping_patience", 0))
    min_delta = float(
        get_nested(config, "training.early_stopping_min_delta", 0.0)
    )
    without_improvement = 0
    last_epoch = 0
    last_val = initial_val
    for epoch in range(1, epochs + 1):
        last_epoch = epoch
        train_dataset.set_epoch(epoch)
        train_loss, train_metrics = run_epoch(
            model,
            train_loader,
            config,
            device,
            optimizer=optimizer,
            scaler=scaler,
        )
        val_loss, val_metrics = run_epoch(
            model,
            val_loader,
            config,
            device,
            optimizer=None,
            scaler=None,
        )
        last_val = val_loss
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                **{
                    f"train_{name}": value
                    for name, value in train_metrics.items()
                },
                **{f"val_{name}": value for name, value in val_metrics.items()},
            }
        )
        _write_history(output_dir / "training_history.csv", history)
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.8f} "
            f"val_loss={val_loss:.8f} "
            f"val_sqi_mae={val_metrics.get('metric_sqi_mae', float('nan')):.6f}"
        )
        if val_loss < best_val - min_delta:
            best_val = val_loss
            best_epoch = epoch
            without_improvement = 0
            torch.save(
                _checkpoint_payload(
                    model,
                    optimizer,
                    config,
                    epoch,
                    val_loss,
                    base_checkpoint,
                    base_checkpoint_sha256,
                    fold,
                    trainable_names,
                ),
                checkpoint_dir / "best.pt",
            )
        else:
            without_improvement += 1
        if bool(get_nested(config, "training.save_every_epoch", True)):
            torch.save(
                _checkpoint_payload(
                    model,
                    optimizer,
                    config,
                    epoch,
                    val_loss,
                    base_checkpoint,
                    base_checkpoint_sha256,
                    fold,
                    trainable_names,
                ),
                checkpoint_dir / f"epoch_{epoch:03d}.pt",
            )
        if patience > 0 and without_improvement >= patience:
            print(
                "Early stopping: no validation improvement for "
                f"{patience} epochs."
            )
            break

    _audit_frozen_state(model, frozen_reference)
    torch.save(
        _checkpoint_payload(
            model,
            optimizer,
            config,
            last_epoch,
            last_val,
            base_checkpoint,
            base_checkpoint_sha256,
            fold,
            trainable_names,
        ),
        checkpoint_dir / "last.pt",
    )
    _write_history(output_dir / "training_history.csv", history)

    summary = {
        "fold": fold,
        "device": str(device),
        "base_checkpoint": base_checkpoint,
        "base_checkpoint_epoch": base.get("epoch"),
        "trainable_scope": "imu_aux_adapter_only",
        "trainable_parameters": trainable_params,
        "total_parameters": total_params,
        "initial_val_loss": initial_val,
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        "last_epoch": last_epoch,
        "validation_bsslab_subjects": sorted(validation_subjects),
        "frozen_m7_audit": "exact",
        "best_validation_metrics": next(
            (
                {
                    key.removeprefix("val_"): value
                    for key, value in row.items()
                    if key.startswith("val_")
                }
                for row in history
                if int(row["epoch"]) == best_epoch
            ),
            {},
        ),
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(f"best_epoch={best_epoch} best_val_loss={best_val:.8f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train auxiliary-only IMU SQI/confidence heads on frozen M7 "
            "(the final V2 YAML selects M14.3-v2)."
        )
    )
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--fold",
        choices=("M001_S01", "M001_S02", "M001_S03", "M001_S04"),
        help="Optional LOSO fold override.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional output directory override.",
    )
    args = parser.parse_args()
    train_m142_from_config(args.config, args.fold, args.output_dir)


if __name__ == "__main__":
    main()
