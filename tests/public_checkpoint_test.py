from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import torch

from ecg_pcg_denoise.models import DenoisingModel
from ecg_pcg_denoise.repro.integrity import verify_checkpoints
from ecg_pcg_denoise.train.eval_m142_imu import _load_models
from ecg_pcg_denoise.utils.config import load_config


ROOT = Path(__file__).resolve().parents[1]
FOLDS = ("M001_S01", "M001_S02", "M001_S03", "M001_S04")


def _strings(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [text for nested in value.values() for text in _strings(nested)]
    if isinstance(value, (list, tuple)):
        return [text for nested in value for text in _strings(nested)]
    return []


def _assert_no_private_path_or_filename(value: object) -> None:
    forbidden = ("c:\\users", "/users/", "onedrive", "@", ".npz", ".mat", ".bin", ".wav")
    for text in _strings(value):
        lowered = text.lower()
        assert not any(token in lowered for token in forbidden), text


def _state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        key_bytes = key.encode("utf-8")
        dtype_bytes = str(tensor.dtype).encode("ascii")
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(struct.pack(">I", len(key_bytes)))
        digest.update(key_bytes)
        digest.update(struct.pack(">I", len(dtype_bytes)))
        digest.update(dtype_bytes)
        digest.update(struct.pack(">I", tensor.ndim))
        for dimension in tensor.shape:
            digest.update(struct.pack(">Q", int(dimension)))
        digest.update(struct.pack(">Q", len(raw)))
        digest.update(raw)
    return digest.hexdigest()


def test_public_checkpoint_ledger_and_factorised_states() -> None:
    report = verify_checkpoints(ROOT)
    assert report["status"] == "pass", report
    ledger = json.loads((ROOT / "checkpoints" / "checksums.json").read_text())
    base_checkpoint = torch.load(
        ROOT / ledger["files"]["m7_v2_base"]["path"],
        map_location="cpu",
        weights_only=True,
    )
    assert set(base_checkpoint) == {
        "format_version",
        "model",
        "config",
        "epoch",
        "val_loss",
        "source_checkpoint_sha256",
    }
    _assert_no_private_path_or_filename(base_checkpoint)
    base_state = base_checkpoint["model"]
    assert _state_sha256(base_state) == ledger["files"]["m7_v2_base"]["canonical_state_sha256"]

    for fold in FOLDS:
        item = ledger["files"][fold]
        checkpoint = torch.load(ROOT / item["path"], map_location="cpu", weights_only=True)
        assert set(checkpoint) == {
            "format_version",
            "artifact_type",
            "adapter",
            "fold",
            "base_checkpoint_sha256",
            "epoch",
            "val_loss",
            "source_checkpoint_sha256",
        }
        _assert_no_private_path_or_filename(checkpoint)
        selection = json.loads((ROOT / item["selection_summary"]["path"]).read_text())
        _assert_no_private_path_or_filename(selection)
        assert checkpoint["artifact_type"] == "m143_v2_imu_aux_adapter"
        assert checkpoint["fold"] == fold
        assert _state_sha256(checkpoint["adapter"]) == item["canonical_adapter_state_sha256"]
        composed = dict(base_state)
        composed.update(
            {f"imu_aux_adapter.{key}": value for key, value in checkpoint["adapter"].items()}
        )
        assert _state_sha256(composed) == item["canonical_composed_model_state_sha256"]


def test_factorised_checkpoints_load_and_preserve_protected_outputs() -> None:
    generator = torch.Generator().manual_seed(1432)
    magnitude = torch.rand(1, 129, 251, generator=generator)
    beat = torch.rand(1, 251, generator=generator)
    imu = torch.randn(1, 6, 251, generator=generator)
    base_path = ROOT / "checkpoints" / "m7_v2" / "best.pt"

    for index, fold in enumerate(FOLDS, start=1):
        config = load_config(ROOT / "configs" / f"bsslab_m143_imu_fold{index}_aligned_calibrated_v2.yaml")
        config["paths"]["base_m7_checkpoint"] = str(base_path)
        m7, m143, _, _ = _load_models(
            config,
            ROOT / "checkpoints" / "m143_v2" / fold / "best_safe.pt",
            torch.device("cpu"),
        )
        assert isinstance(m7, DenoisingModel) and isinstance(m143, DenoisingModel)
        with torch.inference_mode():
            expected = m7(magnitude, beat)
            actual = m143(
                magnitude,
                beat,
                imu_feat=imu,
                modality_mask={"pcg": 1, "ecg": 1, "imu": torch.ones(1)},
            )
        for name in (
            "mask",
            "denoised_mag",
            "phase_residual",
            "complex_mask_real",
            "complex_mask_imag",
            "s1s2_prob",
        ):
            torch.testing.assert_close(actual[name], expected[name], rtol=0.0, atol=0.0)


def test_adapter_fold_mismatch_is_rejected() -> None:
    config = load_config(ROOT / "configs" / "bsslab_m143_imu_fold1_aligned_calibrated_v2.yaml")
    config["paths"]["base_m7_checkpoint"] = str(ROOT / "checkpoints" / "m7_v2" / "best.pt")
    try:
        _load_models(
            config,
            ROOT / "checkpoints" / "m143_v2" / "M001_S02" / "best_safe.pt",
            torch.device("cpu"),
        )
    except ValueError as error:
        assert "does not match" in str(error)
    else:  # pragma: no cover
        raise AssertionError("A fold-mismatched adapter was accepted.")
