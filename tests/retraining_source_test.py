from __future__ import annotations

import csv
import sys
from copy import deepcopy
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ecg_pcg_denoise.models import DenoisingModel
from ecg_pcg_denoise.train import train_m142_imu, train_m143_v2
from ecg_pcg_denoise.train.select_m142_checkpoint import (
    select_m142_checkpoint,
)
from ecg_pcg_denoise.utils.config import load_config


def _model_configs() -> tuple[dict[str, object], dict[str, object]]:
    base: dict[str, object] = {
        "model": {
            "input_channels": 2,
            "use_ecg": True,
            "use_transformer": False,
            "base_channels": 8,
            "dropout": 0.0,
        }
    }
    auxiliary = deepcopy(base)
    model_config = auxiliary["model"]
    if not isinstance(model_config, dict):  # pragma: no cover - test invariant
        raise TypeError("Invalid test model configuration.")
    model_config.update(
        {
            "use_imu": False,
            "use_imu_aux": True,
            "imu_input_channels": 6,
            "imu_hidden_channels": 8,
            "imu_joint_gate_hidden_channels": 8,
        }
    )
    return base, auxiliary


def test_retraining_entrypoints_and_full_configs_are_importable() -> None:
    assert train_m143_v2.main is train_m142_imu.main

    config_names = (
        "bsslab_esc50_v2_m5.yaml",
        "bsslab_esc50_v2_m6_multitask.yaml",
        "bsslab_esc50_v2_m7_distill.yaml",
        "bsslab_esc50_enhanced_v2_m7_robust.yaml",
        "bsslab_esc50_enhanced_v2_m7_robust_sqi.yaml",
    )
    configs = [load_config(ROOT / "configs" / name) for name in config_names]
    assert all(config["training"]["epochs"] > 0 for config in configs)
    assert configs[2]["training"]["init_checkpoint"].endswith(
        "bsslab_esc50_v2_m6_multitask/checkpoints/best.pt"
    )
    assert configs[2]["training"]["teacher_checkpoint"].endswith(
        "bsslab_esc50_v2_m5/checkpoints/best.pt"
    )
    assert configs[4]["training"]["trainable_modules"] == ["sqi_head"]


def test_m7_load_is_weights_only_and_training_scope_is_adapter_only(
    tmp_path: Path,
    monkeypatch,
) -> None:
    base_config, auxiliary_config = _model_configs()
    base = DenoisingModel.from_config(base_config)
    auxiliary = DenoisingModel.from_config(auxiliary_config)
    if not isinstance(base, DenoisingModel) or not isinstance(
        auxiliary, DenoisingModel
    ):
        raise TypeError("Retraining source test requires U-Net models.")

    checkpoint_path = tmp_path / "m7.pt"
    torch.save(
        {"model": base.state_dict(), "config": base_config, "epoch": 7},
        checkpoint_path,
    )
    original_load = torch.load
    observed: dict[str, object] = {}

    def load_spy(*args, **kwargs):
        observed.update(kwargs)
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", load_spy)
    checkpoint = train_m142_imu.load_m7_into_m142(
        auxiliary,
        checkpoint_path,
        torch.device("cpu"),
    )
    trainable, names = train_m142_imu.freeze_m7_train_aux_only(auxiliary)

    assert checkpoint["epoch"] == 7
    assert observed["weights_only"] is True
    assert trainable
    assert names
    assert all(name.startswith(train_m142_imu.AUXILIARY_PREFIX) for name in names)
    assert {
        id(parameter)
        for parameter in auxiliary.parameters()
        if parameter.requires_grad
    } == {id(parameter) for parameter in trainable}


def test_safe_checkpoint_selection_uses_lowest_eligible_validation_loss(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    rows = [
        {
            "epoch": 0,
            "val_loss": 0.8,
            "val_metric_sqi_mae": 0.30,
            "val_metric_base_sqi_mae": 0.31,
            "val_metric_mask_delta": 0.0,
            "val_metric_s1s2_location_delta": 0.0,
        },
        {
            "epoch": 1,
            "val_loss": 0.1,
            "val_metric_sqi_mae": 0.32,
            "val_metric_base_sqi_mae": 0.31,
            "val_metric_mask_delta": 0.0,
            "val_metric_s1s2_location_delta": 0.0,
        },
        {
            "epoch": 2,
            "val_loss": 0.4,
            "val_metric_sqi_mae": 0.29,
            "val_metric_base_sqi_mae": 0.31,
            "val_metric_mask_delta": 0.0,
            "val_metric_s1s2_location_delta": 0.0,
        },
        {
            "epoch": 3,
            "val_loss": 0.2,
            "val_metric_sqi_mae": 0.28,
            "val_metric_base_sqi_mae": 0.31,
            "val_metric_mask_delta": 1e-8,
            "val_metric_s1s2_location_delta": 0.0,
        },
        {
            "epoch": 4,
            "val_loss": float("nan"),
            "val_metric_sqi_mae": 0.27,
            "val_metric_base_sqi_mae": 0.31,
            "val_metric_mask_delta": 0.0,
            "val_metric_s1s2_location_delta": 0.0,
        },
    ]
    history_path = tmp_path / "training_history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        epoch = int(row["epoch"])
        (checkpoint_dir / f"epoch_{epoch:03d}.pt").write_bytes(
            f"epoch={epoch}".encode("ascii")
        )

    summary = select_m142_checkpoint(tmp_path)

    assert summary["selected_epoch"] == 2
    assert summary["eligible_epochs"] == [0, 2]
    expected = (checkpoint_dir / "epoch_002.pt").read_bytes()
    assert (checkpoint_dir / "best.pt").read_bytes() == expected
    assert (checkpoint_dir / "best_safe.pt").read_bytes() == expected


def test_training_checkpoint_binds_fold_and_base_digest(tmp_path: Path) -> None:
    _, auxiliary_config = _model_configs()
    model = DenoisingModel.from_config(auxiliary_config)
    if not isinstance(model, DenoisingModel):  # pragma: no cover - test invariant
        raise TypeError("Retraining source test requires a U-Net model.")
    trainable, names = train_m142_imu.freeze_m7_train_aux_only(model)
    optimizer = torch.optim.AdamW(trainable, lr=1e-3)
    base_checkpoint = tmp_path / "m7.pt"
    base_checkpoint.write_bytes(b"sealed-m7-checkpoint")

    payload = train_m142_imu._checkpoint_payload(
        model,
        optimizer,
        auxiliary_config,
        3,
        0.25,
        str(base_checkpoint),
        train_m142_imu._sha256(base_checkpoint),
        "M001_S01",
        names,
    )

    assert payload["artifact_type"] == "m143_v2_full_training_checkpoint"
    assert payload["fold"] == "M001_S01"
    assert payload["base_checkpoint_sha256"] == train_m142_imu._sha256(
        base_checkpoint
    )
    assert payload["trainable_scope"] == "imu_aux_adapter_only"
