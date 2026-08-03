from __future__ import annotations

from copy import deepcopy

import torch

from ecg_pcg_denoise.models import DenoisingModel
from ecg_pcg_denoise.train.m142_imu_runtime import reforward_m142_auxiliary


def _configs() -> tuple[dict[str, object], dict[str, object]]:
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
    auxiliary_model = auxiliary["model"]
    if not isinstance(auxiliary_model, dict):
        raise TypeError("Invalid test model config.")
    auxiliary_model.update(
        {
            "use_imu": False,
            "use_imu_aux": True,
            "imu_input_channels": 6,
            "imu_hidden_channels": 8,
            "imu_joint_gate_hidden_channels": 8,
            "imu_artifact_gate_bias": -3.0,
            "imu_aux_max_sqi_logit_delta": 0.75,
        }
    )
    return base, auxiliary


def _models() -> tuple[DenoisingModel, DenoisingModel]:
    torch.manual_seed(142)
    base_config, auxiliary_config = _configs()
    base = DenoisingModel.from_config(base_config)
    auxiliary = DenoisingModel.from_config(auxiliary_config)
    if not isinstance(base, DenoisingModel) or not isinstance(auxiliary, DenoisingModel):
        raise TypeError("M14.2 tests require U-Net models.")
    incompatible = auxiliary.load_state_dict(base.state_dict(), strict=False)
    assert not incompatible.unexpected_keys
    assert incompatible.missing_keys
    assert all(key.startswith("imu_aux_adapter.") for key in incompatible.missing_keys)
    assert auxiliary.imu_adapter is None
    assert auxiliary.imu_aux_adapter is not None
    return base.eval(), auxiliary.eval()


def _inputs(batch: int = 2) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(143)
    magnitude = torch.rand(batch, 33, 41, generator=generator)
    beat = torch.rand(batch, 41, generator=generator)
    imu = torch.randn(batch, 6, 41, generator=generator)
    return magnitude, beat, imu


def _assert_base_outputs_equal(
    expected: dict[str, torch.Tensor],
    actual: dict[str, torch.Tensor],
) -> None:
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=0.0, atol=0.0)


def test_m142_zero_initialized_auxiliary_path_is_exact_m7() -> None:
    base, auxiliary = _models()
    magnitude, beat, imu = _inputs()
    with torch.no_grad():
        expected = base(magnitude, beat)
        actual = auxiliary(
            magnitude,
            beat,
            imu_feat=imu,
            modality_mask={"pcg": 1, "ecg": 1, "imu": torch.ones(2)},
        )

    _assert_base_outputs_equal(expected, actual)
    torch.testing.assert_close(
        actual["base_sqi_score"],
        expected["sqi_score"],
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        actual["base_s1s2_prob"],
        expected["s1s2_prob"],
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        actual["base_mask"],
        expected["mask"],
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        actual["imu_mask_delta"],
        torch.zeros_like(actual["imu_mask_delta"]),
        rtol=0.0,
        atol=0.0,
    )
    assert actual["imu_motion_probability"].shape == (2, 41)
    assert actual["imu_artifact_probability"].shape == (2, 41)
    assert actual["imu_reliability"].shape == (2, 41)
    assert actual["sqi_confidence"].shape == (2,)
    assert actual["s1s2_confidence"].shape == (2, 2, 41)
    assert actual["imu_sqi_logit_delta"].shape == (2,)
    assert torch.count_nonzero(actual["imu_sqi_logit_delta"]) == 0
    torch.testing.assert_close(
        actual["sqi_confidence"],
        torch.full_like(actual["sqi_confidence"], 0.5),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        actual["s1s2_confidence"],
        torch.full_like(actual["s1s2_confidence"], 0.5),
        rtol=0.0,
        atol=0.0,
    )


def test_m142_absent_imu_is_exact_fallback_after_auxiliary_changes() -> None:
    base, auxiliary = _models()
    if auxiliary.imu_aux_adapter is None:
        raise AssertionError("Missing M14.2 auxiliary adapter.")
    with torch.no_grad():
        for parameter in auxiliary.imu_aux_adapter.parameters():
            parameter.uniform_(-0.2, 0.2)
    magnitude, beat, imu = _inputs()
    with torch.no_grad():
        expected = base(magnitude, beat)
        actual = auxiliary(
            magnitude,
            beat,
            imu_feat=imu,
            modality_mask={"pcg": 1, "ecg": 1, "imu": torch.zeros(2)},
        )
        absent_tensor = auxiliary(
            magnitude,
            beat,
            imu_feat=None,
            modality_mask={"pcg": 1, "ecg": 1, "imu": torch.zeros(2)},
        )

    _assert_base_outputs_equal(expected, actual)
    _assert_base_outputs_equal(expected, absent_tensor)
    for name in (
        "imu_motion_probability",
        "imu_artifact_probability",
        "imu_reliability",
        "imu_sqi_logit_delta",
    ):
        torch.testing.assert_close(
            actual[name],
            torch.zeros_like(actual[name]),
            rtol=0.0,
            atol=0.0,
        )
    assert torch.all(actual["sqi_confidence"] > 0.0)
    assert torch.all(actual["sqi_confidence"] < 1.0)
    assert torch.all(actual["s1s2_confidence"] > 0.0)
    assert torch.all(actual["s1s2_confidence"] < 1.0)
    assert not torch.equal(
        actual["sqi_confidence"],
        torch.ones_like(actual["sqi_confidence"]),
    )
    assert not torch.equal(
        actual["s1s2_confidence"],
        torch.ones_like(actual["s1s2_confidence"]),
    )
    torch.testing.assert_close(
        absent_tensor["sqi_confidence"],
        actual["sqi_confidence"],
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        absent_tensor["s1s2_confidence"],
        actual["s1s2_confidence"],
        rtol=0.0,
        atol=0.0,
    )


def test_m142_auxiliary_outputs_never_change_mask_or_s1s2_locations() -> None:
    base, auxiliary = _models()
    if auxiliary.imu_aux_adapter is None:
        raise AssertionError("Missing M14.2 auxiliary adapter.")
    with torch.no_grad():
        for parameter in auxiliary.imu_aux_adapter.parameters():
            parameter.uniform_(-0.15, 0.15)
    magnitude, beat, imu = _inputs()
    with torch.no_grad():
        expected = base(magnitude, beat)
        first = auxiliary(
            magnitude,
            beat,
            imu_feat=imu,
            modality_mask={"pcg": 1, "ecg": 1, "imu": torch.ones(2)},
        )
        second = auxiliary(
            magnitude,
            beat,
            imu_feat=torch.flip(imu, dims=(-1,)),
            modality_mask={"pcg": 1, "ecg": 1, "imu": torch.ones(2)},
        )

    for name in (
        "mask",
        "denoised_mag",
        "phase_residual",
        "complex_mask_real",
        "complex_mask_imag",
        "s1s2_prob",
    ):
        torch.testing.assert_close(first[name], expected[name], rtol=0.0, atol=0.0)
        torch.testing.assert_close(second[name], expected[name], rtol=0.0, atol=0.0)
    assert torch.all(first["imu_sqi_logit_delta"].abs() <= 0.75)
    assert torch.all(second["imu_sqi_logit_delta"].abs() <= 0.75)
    for name in (
        "imu_motion_probability",
        "imu_artifact_probability",
        "imu_reliability",
        "sqi_confidence",
        "s1s2_confidence",
        "sqi_score",
    ):
        assert torch.all(first[name] >= 0.0)
        assert torch.all(first[name] <= 1.0)


def test_m142_auxiliary_heads_receive_gradients_at_identity_start() -> None:
    _, auxiliary = _models()
    if auxiliary.imu_aux_adapter is None:
        raise AssertionError("Missing M14.2 auxiliary adapter.")
    auxiliary.train()
    magnitude, beat, imu = _inputs()
    output = auxiliary(
        magnitude,
        beat,
        imu_feat=imu,
        modality_mask={"pcg": 1, "ecg": 1, "imu": torch.ones(2)},
    )
    loss = (
        output["sqi_score"].mean()
        + output["imu_motion_probability"].mean()
        + output["imu_artifact_probability"].mean()
        + output["imu_reliability"].mean()
        + output["sqi_confidence"].mean()
        + output["s1s2_confidence"].mean()
    )
    loss.backward()

    heads = (
        auxiliary.imu_aux_adapter.motion_head,
        auxiliary.imu_aux_adapter.artifact_head,
        auxiliary.imu_aux_adapter.reliability_head,
        auxiliary.imu_aux_adapter.sqi_confidence_head,
        auxiliary.imu_aux_adapter.s1s2_confidence_head,
        auxiliary.imu_aux_adapter.sqi_residual_head,
    )
    for head in heads:
        gradient = head.weight.grad
        assert gradient is not None
        assert torch.all(torch.isfinite(gradient))
        assert torch.count_nonzero(gradient) > 0


def test_m142_cached_counterfactual_matches_full_backbone_forward() -> None:
    _, auxiliary = _models()
    auxiliary.eval()
    magnitude, beat, imu = _inputs()
    batch = {
        "imu_feat": imu,
        "imu_valid_mask": torch.ones(imu.shape[0], imu.shape[-1]),
        "imu_present": torch.ones(imu.shape[0]),
        "imu_subject_id": ["participant_a", "participant_b"],
    }
    shift = max(1, imu.shape[-1] // 16)
    shifted = torch.nn.functional.pad(imu[..., :-shift], (shift, 0))
    with torch.no_grad():
        reference = auxiliary(
            magnitude,
            beat,
            imu_feat=imu,
            modality_mask={"pcg": 1, "ecg": 1, "imu": torch.ones(2)},
            return_aux_context=True,
        )
        cached = reforward_m142_auxiliary(
            auxiliary,
            reference,
            batch,
            torch.device("cpu"),
            "shift",
        )
        complete = auxiliary(
            magnitude,
            beat,
            imu_feat=shifted,
            modality_mask={"pcg": 1, "ecg": 1, "imu": torch.ones(2)},
        )
    for name in (
        "mask",
        "s1s2_prob",
        "sqi_score",
        "sqi_confidence",
        "s1s2_confidence",
        "imu_motion_probability",
        "imu_artifact_probability",
        "imu_reliability",
    ):
        torch.testing.assert_close(cached[name], complete[name], rtol=0.0, atol=0.0)
