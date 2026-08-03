from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ecg_pcg_denoise.models import DenoisingModel  # noqa: E402


def check_model(model_config: dict[str, object]) -> set[str]:
    model = DenoisingModel.from_config({"model": model_config}).eval()
    pcg = torch.rand(2, 129, 251)
    beat = torch.rand(2, 251)
    with torch.no_grad():
        output = model(
            pcg_stft=pcg,
            ecg_beat=beat,
            imu_feat=None,
            modality_mask={"pcg": 1, "ecg": int(bool(model_config.get("use_ecg"))), "imu": 0},
        )
    assert output["mask"].shape == pcg.shape
    assert output["denoised_mag"].shape == pcg.shape
    assert output["s1s2_prob"].shape == (2, 2, 251)
    assert output["sqi_score"].shape == (2,)
    assert torch.isfinite(output["mask"]).all()
    return set(output)


def test_model_architectures() -> None:
    """Exercise both released neural architecture families without dataset files."""
    unet_keys = check_model(
        {
            "architecture": "unet",
            "input_channels": 1,
            "use_ecg": False,
            "use_transformer": False,
            "base_channels": 16,
        }
    )
    transformer_keys = check_model(
        {
            "architecture": "pure_transformer",
            "input_channels": 1,
            "use_ecg": False,
            "base_channels": 16,
            "pure_transformer_dim": 128,
            "pure_transformer_layers": 4,
            "pure_transformer_heads": 4,
            "pure_transformer_ff_multiplier": 2,
            "pure_transformer_patch_frequency": 8,
            "pure_transformer_patch_time": 8,
        }
    )
    assert transformer_keys == unet_keys


def main() -> None:
    """Retain the original direct-execution smoke-check interface."""
    test_model_architectures()
    print("model_architecture_test passed")


if __name__ == "__main__":
    main()
