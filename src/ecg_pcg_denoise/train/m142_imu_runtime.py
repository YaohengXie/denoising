from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
import torch.nn.functional as F

from ecg_pcg_denoise.models import DenoisingModel
from ecg_pcg_denoise.train.m14_imu_runtime import artifact_gate_target
from ecg_pcg_denoise.train.torch_audio import stft_waveform
from ecg_pcg_denoise.train.train_denoise import (
    beat_to_frames,
    s1s2_to_frames,
    stft_config,
)
from ecg_pcg_denoise.utils.config import get_nested, require_nested


ARTIFACT_TYPE_NAMES = (
    "clean_identity",
    "motion_clean",
    "motion_artifact",
    "independent_artifact",
    "combined",
)


def prepare_imu_aux(
    batch: dict[str, Any],
    device: torch.device,
    mode: str = "correct",
    shift_frames: int | None = None,
    shift_direction: int = 1,
) -> tuple[Tensor, Tensor]:
    """Prepare the IMU branch without changing the PCG/ECG inputs.

    ``shuffle`` prefers a different Motema participant for every sample. If a
    batch contains only one participant, a non-circular temporal shift is used
    instead. ``shift`` is also non-circular, so the counterfactual does not
    create an artificial wrap-around discontinuity.
    """

    features = batch["imu_feat"].to(device, non_blocking=True).float()
    valid = batch["imu_valid_mask"].to(device, non_blocking=True).float()
    features = features * valid[:, None, :]
    present = batch["imu_present"].to(device, non_blocking=True).float()

    if mode == "correct":
        return features, present
    if mode == "missing":
        return torch.zeros_like(features), torch.zeros_like(present)
    if mode == "shuffle":
        subject_ids = [str(value) for value in batch.get("imu_subject_id", [])]
        if len(subject_ids) == features.shape[0] and len(set(subject_ids)) > 1:
            indices: list[int] = []
            for index, subject_id in enumerate(subject_ids):
                candidates = [
                    candidate
                    for candidate, other_id in enumerate(subject_ids)
                    if candidate != index and other_id != subject_id
                ]
                indices.append(candidates[index % len(candidates)])
            permutation = torch.tensor(indices, device=device, dtype=torch.long)
            return (
                features.index_select(0, permutation),
                present.index_select(0, permutation),
            )
        shift = max(
            1,
            min(
                int(shift_frames or features.shape[-1] // 16),
                features.shape[-1] - 1,
            ),
        )
        shifted = F.pad(features[..., :-shift], (shift, 0))
        return shifted, present
    if mode == "shift":
        shift = max(
            1,
            min(
                int(shift_frames or features.shape[-1] // 16),
                features.shape[-1] - 1,
            ),
        )
        if int(shift_direction) < 0:
            shifted = F.pad(features[..., shift:], (0, shift))
        else:
            shifted = F.pad(features[..., :-shift], (shift, 0))
        return shifted, present
    raise ValueError(f"Unsupported M14.2 IMU mode: {mode}")


def _resize_frames(values: Tensor, frames: int) -> Tensor:
    values = values.float()
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim == 2:
        values = values[:, None, :]
    if values.ndim != 3:
        raise ValueError(f"Expected [B,T] or [B,1,T], got {values.shape}")
    if values.shape[-1] != frames:
        values = F.interpolate(values, size=frames, mode="linear", align_corners=False)
    return values.squeeze(1)


def _local_quality_target(
    clean_magnitude: Tensor,
    noise_magnitude: Tensor,
    frames: int,
) -> Tensor:
    """Map frame-local clean/noise energy to the established M7 SQI scale."""

    clean_power = clean_magnitude.float().square().mean(dim=-2)
    noise_power = noise_magnitude.float().square().mean(dim=-2)
    local_snr = 10.0 * torch.log10(
        (clean_power + 1e-8) / (noise_power + 1e-8)
    )
    quality = torch.zeros_like(local_snr)
    quality = torch.where(local_snr >= 0.0, torch.full_like(quality, 0.33), quality)
    quality = torch.where(local_snr >= 5.0, torch.full_like(quality, 0.66), quality)
    quality = torch.where(local_snr >= 10.0, torch.ones_like(quality), quality)
    return _resize_frames(quality, frames).clamp(0.0, 1.0)


def _local_artifact_strength(
    clean_magnitude: Tensor,
    noise_magnitude: Tensor,
    frames: int,
) -> Tensor:
    """Return the frame-local fraction of energy attributable to noise."""

    clean_power = clean_magnitude.float().square().mean(dim=-2)
    noise_power = noise_magnitude.float().square().mean(dim=-2)
    fraction = noise_power / (clean_power + noise_power + 1e-8)
    return _resize_frames(fraction, frames).clamp(0.0, 1.0)


def _artifact_type_target(
    modes: list[str] | tuple[str, ...],
    device: torch.device,
) -> Tensor:
    indices = {name: index for index, name in enumerate(ARTIFACT_TYPE_NAMES)}
    try:
        values = [indices[str(mode)] for mode in modes]
    except KeyError as error:
        raise ValueError(f"Unsupported M14.2 artifact mode: {error.args[0]}") from error
    return torch.tensor(values, device=device, dtype=torch.long)


def build_m142_targets(
    output: dict[str, Tensor],
    tensors: dict[str, Tensor],
    batch: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Tensor]:
    if "imu_motion_probability" not in output:
        raise KeyError("M14.2 output is missing imu_motion_probability.")
    frames = int(output["imu_motion_probability"].shape[-1])
    device = output["imu_motion_probability"].device
    dtype = output["imu_motion_probability"].dtype

    motion_target = _resize_frames(
        batch["motion_envelope"].to(device, non_blocking=True),
        frames,
    ).to(dtype=dtype)
    coupled_target_source = str(
        get_nested(
            config,
            "training.targets.coupled_artifact_source",
            "imu_envelope",
        )
    )
    if coupled_target_source == "imu_envelope":
        coupled_target = artifact_gate_target(
            batch,
            frames,
            device,
            dtype,
            positive_floor=float(
                get_nested(
                    config,
                    "training.targets.artifact_gate_positive_floor",
                    0.15,
                )
            ),
        )
    elif coupled_target_source == "local_noise_fraction":
        artifact_strength = _local_artifact_strength(
            tensors["clean_mag"],
            tensors["noise_mag"],
            frames,
        ).to(device=device, dtype=dtype)
        positive = torch.tensor(
            [
                str(mode) in {"motion_artifact", "combined"}
                for mode in batch["artifact_mode"]
            ],
            device=device,
            dtype=dtype,
        )
        coupled_target = positive[:, None] * artifact_strength
    else:
        raise ValueError(
            "Unsupported training.targets.coupled_artifact_source: "
            f"{coupled_target_source}"
        )
    valid_target = _resize_frames(
        batch["imu_valid_mask"].to(device, non_blocking=True),
        frames,
    ).to(dtype=dtype)
    present = batch["imu_present"].to(device, non_blocking=True).to(dtype=dtype)
    reliability_target = valid_target * present[:, None]

    s1s2_target = s1s2_to_frames(
        batch["s1s2_weak"].to(device, non_blocking=True),
        frames,
    ).to(dtype=dtype)
    local_quality = _local_quality_target(
        tensors["clean_mag"],
        tensors["noise_mag"],
        frames,
    ).to(dtype=dtype)
    # Confidence is trained as P(the frozen M7 event location is consistent
    # with the weak ECG-timed target within 40 ms). It is not a second
    # location predictor: the M7 S1/S2 probability map remains unchanged.
    tolerance_ms = float(
        get_nested(config, "training.targets.s1s2_confidence_tolerance_ms", 40.0)
    )
    frame_ms = 1000.0 * float(
        require_nested(config, "stft.hop_length")
    ) / float(require_nested(config, "data.fs_model"))
    tolerance_frames = max(1, int(round(tolerance_ms / frame_ms)))
    target_peak = (
        (s1s2_target >= 0.50)
        & (s1s2_target >= F.pad(s1s2_target[..., :-1], (1, 0)))
        & (s1s2_target >= F.pad(s1s2_target[..., 1:], (0, 1)))
    ).float()
    s1s2_confidence_target = F.max_pool1d(
        target_peak,
        kernel_size=2 * tolerance_frames + 1,
        stride=1,
        padding=tolerance_frames,
    ).clamp(0.0, 1.0)
    base_s1s2 = output.get("base_s1s2_prob", output["s1s2_prob"]).detach()
    s1s2_confidence_weight = torch.maximum(
        base_s1s2.float(),
        target_peak,
    ).clamp(0.0, 1.0)

    sqi_target = batch["sqi_target"].to(device, non_blocking=True).to(dtype=dtype)
    return {
        "sqi": sqi_target,
        "motion": motion_target.clamp(0.0, 1.0),
        "coupled_artifact": coupled_target.clamp(0.0, 1.0),
        "reliability": reliability_target.clamp(0.0, 1.0),
        "s1s2": s1s2_target,
        "s1s2_confidence": s1s2_confidence_target.clamp(0.0, 1.0),
        "s1s2_confidence_weight": s1s2_confidence_weight,
        "local_quality": local_quality,
        "artifact_type": _artifact_type_target(batch["artifact_mode"], device),
        "imu_present": present,
    }


def forward_m142_batch(
    model: DenoisingModel,
    batch: dict[str, Any],
    config: dict[str, Any],
    device: torch.device,
    imu_mode: str = "correct",
    build_targets: bool = True,
) -> tuple[dict[str, Tensor], dict[str, Tensor], dict[str, Tensor]]:
    """Run frozen M7 plus the M14.2 auxiliary adapter.

    Waveform reconstruction is intentionally omitted: M14.2 is an SQI and
    confidence experiment and its IMU branch is not allowed to alter the mask.
    """

    n_fft, hop, win = stft_config(config)
    noisy = batch["noisy_pcg"].to(device, non_blocking=True).float()
    clean = batch["clean_pcg"].to(device, non_blocking=True).float()
    noise = batch["noise_signal"].to(device, non_blocking=True).float()
    beat = batch["ecg_beat"].to(device, non_blocking=True).float()
    imu, imu_present = prepare_imu_aux(batch, device, imu_mode)

    noisy_complex = stft_waveform(noisy, n_fft, hop, win)
    clean_complex = stft_waveform(clean, n_fft, hop, win)
    noise_complex = stft_waveform(noise, n_fft, hop, win)
    noisy_mag = noisy_complex.abs()
    clean_mag = clean_complex.abs()
    noise_mag = noise_complex.abs()
    beat_frames = beat_to_frames(beat, noisy_mag.shape[-1])
    output = model(
        pcg_stft=noisy_mag,
        ecg_beat=beat_frames,
        imu_feat=imu,
        modality_mask={"pcg": 1, "ecg": 1, "imu": imu_present},
        return_aux_context=True,
    )
    tensors = {
        "noisy": noisy,
        "clean": clean,
        "noise": noise,
        "noisy_mag": noisy_mag,
        "clean_mag": clean_mag,
        "noise_mag": noise_mag,
    }
    targets = (
        build_m142_targets(output, tensors, batch, config)
        if build_targets
        else {}
    )
    return output, tensors, targets


def reforward_m142_auxiliary(
    model: DenoisingModel,
    reference_output: dict[str, Tensor],
    batch: dict[str, Any],
    device: torch.device,
    imu_mode: str,
    shift_frames: int | None = None,
    shift_direction: int = 1,
) -> dict[str, Tensor]:
    """Re-run only the auxiliary adapter for an IMU counterfactual.

    M7 is frozen and its PCG/ECG path is identical across IMU variants. Reusing
    its decoder context avoids two redundant U-Net/Transformer passes per
    batch while preserving bit-exact mask and S1/S2 outputs.
    """

    adapter = getattr(model, "imu_aux_adapter", None)
    if adapter is None:
        raise RuntimeError("M14.2 auxiliary adapter is unavailable.")
    if "_imu_aux_decoder_features" not in reference_output:
        raise KeyError("Reference output has no reusable M14.2 context.")
    imu, imu_present = prepare_imu_aux(
        batch,
        device,
        imu_mode,
        shift_frames=shift_frames,
        shift_direction=shift_direction,
    )
    aux_output = adapter(
        reference_output["_imu_aux_decoder_features"],
        imu,
        {"pcg": 1, "ecg": 1, "imu": imu_present},
        reference_output["_imu_aux_beat_map"],
    )
    base_sqi = reference_output["base_sqi_score"]
    base_sqi_float = base_sqi.float()
    base_logit = torch.logit(base_sqi_float.clamp(1e-6, 1.0 - 1e-6))
    base_roundtrip = torch.sigmoid(base_logit)
    corrected_sqi = base_sqi_float + (
        torch.sigmoid(
            base_logit + aux_output["imu_sqi_logit_delta"].float()
        )
        - base_roundtrip
    )
    result = dict(reference_output)
    result["sqi_score"] = corrected_sqi.clamp(0.0, 1.0).to(base_sqi.dtype)
    result.update(aux_output)
    return result


def _probability_bce(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    # Probability-form BCE is intentionally evaluated in FP32 because PyTorch
    # rejects BCELoss inside an autocast region. The model heads remain
    # sigmoid probabilities so they can be inspected and calibrated directly.
    with torch.autocast(device_type=prediction.device.type, enabled=False):
        prediction = prediction.float().clamp(1e-6, 1.0 - 1e-6)
        target = target.float()
        loss = F.binary_cross_entropy(prediction, target, reduction="none")
    if mask is None:
        return loss.mean()
    mask = torch.broadcast_to(mask.float(), loss.shape)
    return torch.sum(loss * mask) / (torch.sum(mask) + 1e-8)


def _masked_l1(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    error = torch.abs(prediction.float() - target.float())
    if mask is None:
        return error.mean()
    mask = torch.broadcast_to(mask.float(), error.shape)
    return torch.sum(error * mask) / (torch.sum(mask) + 1e-8)


def m142_auxiliary_loss(
    output: dict[str, Tensor],
    targets: dict[str, Tensor],
    config: dict[str, Any],
    return_components: bool = False,
) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
    weights = get_nested(config, "training.loss", {})
    device = output["sqi_score"].device
    zero = torch.zeros((), device=device, dtype=torch.float32)

    sqi_kind = str(weights.get("sqi_kind", "huber")).lower()
    if sqi_kind == "huber":
        sqi_loss = F.huber_loss(
            output["sqi_score"].float(),
            targets["sqi"].float(),
            delta=float(weights.get("sqi_huber_delta", 0.10)),
        )
    elif sqi_kind == "mse":
        sqi_loss = F.mse_loss(
            output["sqi_score"].float(),
            targets["sqi"].float(),
        )
    else:
        raise ValueError(f"Unsupported training.loss.sqi_kind: {sqi_kind}")

    motion_loss = _probability_bce(
        output["imu_motion_probability"],
        targets["motion"],
    )
    coupled_loss = _probability_bce(
        output["imu_artifact_probability"],
        targets["coupled_artifact"],
    )
    reliability_loss = _probability_bce(
        output["imu_reliability"],
        targets["reliability"],
    )

    confidence_tolerance = float(weights.get("sqi_confidence_tolerance", 0.10))
    sqi_confidence_target = (
        torch.abs(
            output["sqi_score"].detach().float() - targets["sqi"].float()
        )
        <= confidence_tolerance
    ).float()
    present_mask = targets["imu_present"].float()
    sqi_confidence_loss = _probability_bce(
        output["sqi_confidence"],
        sqi_confidence_target,
        mask=present_mask,
    )
    s1s2_confidence_loss = _probability_bce(
        output["s1s2_confidence"],
        targets["s1s2_confidence"],
        mask=targets["s1s2_confidence_weight"],
    )

    artifact_type_loss = zero
    if "imu_artifact_type_logits" in output:
        logits = output["imu_artifact_type_logits"].float()
        if logits.ndim != 2:
            raise ValueError(
                "imu_artifact_type_logits must have shape [B,C], "
                f"got {logits.shape}"
            )
        if logits.shape[-1] != len(ARTIFACT_TYPE_NAMES):
            raise ValueError(
                "M14.2 artifact type head must use "
                f"{len(ARTIFACT_TYPE_NAMES)} classes, got {logits.shape[-1]}"
            )
        artifact_type_loss = F.cross_entropy(logits, targets["artifact_type"])

    mask_preservation = zero
    if "base_mask" in output:
        mask_preservation = F.l1_loss(
            output["mask"].float(),
            output["base_mask"].detach().float(),
        )
    s1s2_location_preservation = zero
    if "base_s1s2_prob" in output:
        s1s2_location_preservation = F.l1_loss(
            output["s1s2_prob"].float(),
            output["base_s1s2_prob"].detach().float(),
        )

    total = (
        float(weights.get("sqi", 1.0)) * sqi_loss
        + float(weights.get("motion", 0.5)) * motion_loss
        + float(weights.get("coupled_artifact", 0.75)) * coupled_loss
        + float(weights.get("reliability", 0.25)) * reliability_loss
        + float(weights.get("sqi_confidence", 0.20)) * sqi_confidence_loss
        + float(weights.get("s1s2_confidence", 0.20)) * s1s2_confidence_loss
        + float(weights.get("artifact_type", 0.0)) * artifact_type_loss
        + float(weights.get("mask_preservation", 10.0)) * mask_preservation
        + float(weights.get("s1s2_location_preservation", 10.0))
        * s1s2_location_preservation
    )
    if not return_components:
        return total
    return total, {
        "sqi_loss": sqi_loss.detach(),
        "motion_loss": motion_loss.detach(),
        "coupled_artifact_loss": coupled_loss.detach(),
        "reliability_loss": reliability_loss.detach(),
        "sqi_confidence_loss": sqi_confidence_loss.detach(),
        "s1s2_confidence_loss": s1s2_confidence_loss.detach(),
        "artifact_type_loss": artifact_type_loss.detach(),
        "mask_preservation_loss": mask_preservation.detach(),
        "s1s2_location_preservation_loss": (
            s1s2_location_preservation.detach()
        ),
    }


def m142_fallback_loss(
    output: dict[str, Tensor],
    mode: str,
    config: dict[str, Any],
    reference_output: dict[str, Tensor] | None = None,
    targets: dict[str, Tensor] | None = None,
    return_components: bool = False,
) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
    """Force missing or temporally mismatched IMU to fall back to frozen M7."""

    if mode not in {"missing", "shuffle", "shift"}:
        raise ValueError(f"Fallback loss does not support mode={mode}")
    weights = get_nested(config, "training.fallback", {})
    device = output["sqi_score"].device
    zero = torch.zeros((), device=device, dtype=torch.float32)

    sqi_fallback = zero
    if "base_sqi_score" in output:
        sqi_fallback = F.smooth_l1_loss(
            output["sqi_score"].float(),
            output["base_sqi_score"].detach().float(),
            beta=float(weights.get("huber_beta", 0.05)),
        )
    mask_fallback = zero
    if "base_mask" in output:
        mask_fallback = F.l1_loss(
            output["mask"].float(),
            output["base_mask"].detach().float(),
        )
    elif reference_output is not None:
        mask_fallback = F.l1_loss(
            output["mask"].float(),
            reference_output["mask"].detach().float(),
        )
    s1s2_fallback = zero
    if "base_s1s2_prob" in output:
        s1s2_fallback = F.l1_loss(
            output["s1s2_prob"].float(),
            output["base_s1s2_prob"].detach().float(),
        )
    elif reference_output is not None:
        s1s2_fallback = F.l1_loss(
            output["s1s2_prob"].float(),
            reference_output["s1s2_prob"].detach().float(),
        )
    reliability_zero = (
        torch.mean(output["imu_reliability"].float().square())
        if mode == "missing"
        else zero
    )
    coupled_zero = torch.mean(output["imu_artifact_probability"].float().square())
    delta_zero = zero
    if "imu_sqi_logit_delta" in output:
        delta_zero = torch.mean(output["imu_sqi_logit_delta"].float().square())
    sqi_confidence = zero
    s1s2_confidence = zero
    alignment_margin_loss = zero
    if targets is not None:
        confidence_tolerance = float(
            get_nested(config, "training.loss.sqi_confidence_tolerance", 0.10)
        )
        sqi_confidence_target = (
            torch.abs(
                output["sqi_score"].detach().float()
                - targets["sqi"].float()
            )
            <= confidence_tolerance
        ).float()
        sqi_confidence = _probability_bce(
            output["sqi_confidence"],
            sqi_confidence_target,
        )
        s1s2_confidence = _probability_bce(
            output["s1s2_confidence"],
            targets["s1s2_confidence"],
            mask=targets["s1s2_confidence_weight"],
        )
        if (
            mode in {"shuffle", "shift"}
            and reference_output is not None
            and "imu_artifact_probability" in reference_output
        ):
            aligned = reference_output["imu_artifact_probability"].float()
            mismatched = output["imu_artifact_probability"].float()
            active = (
                (targets["coupled_artifact"].float() > 0.0).float()
                * targets["reliability"].float()
            )
            margin = float(weights.get("alignment_margin_value", 0.05))
            pointwise = F.relu(margin - aligned + mismatched)
            alignment_margin_loss = torch.sum(pointwise * active) / (
                torch.sum(active) + 1e-8
            )

    total = (
        float(weights.get("sqi_to_base", 1.0)) * sqi_fallback
        + float(weights.get("mask_to_base", 10.0)) * mask_fallback
        + float(weights.get("s1s2_to_base", 10.0)) * s1s2_fallback
        + float(weights.get("reliability_zero", 0.5)) * reliability_zero
        + float(weights.get("coupled_artifact_zero", 0.5)) * coupled_zero
        + float(weights.get("sqi_delta_zero", 0.5)) * delta_zero
        + float(weights.get("sqi_confidence", 0.20)) * sqi_confidence
        + float(weights.get("s1s2_confidence", 0.20)) * s1s2_confidence
        + float(weights.get("alignment_margin_weight", 0.0))
        * alignment_margin_loss
    )
    if not return_components:
        return total
    prefix = f"{mode}_fallback"
    return total, {
        f"{prefix}_loss": total.detach(),
        f"{prefix}_sqi_to_base": sqi_fallback.detach(),
        f"{prefix}_mask_to_base": mask_fallback.detach(),
        f"{prefix}_s1s2_to_base": s1s2_fallback.detach(),
        f"{prefix}_reliability": reliability_zero.detach(),
        f"{prefix}_coupled_artifact": coupled_zero.detach(),
        f"{prefix}_sqi_delta": delta_zero.detach(),
        f"{prefix}_sqi_confidence": sqi_confidence.detach(),
        f"{prefix}_s1s2_confidence": s1s2_confidence.detach(),
        f"{prefix}_alignment_margin": alignment_margin_loss.detach(),
    }


def m142_metrics(
    output: dict[str, Tensor],
    targets: dict[str, Tensor],
    fallback_outputs: dict[str, dict[str, Tensor]] | None = None,
) -> dict[str, Tensor]:
    sqi_error = torch.abs(output["sqi_score"].float() - targets["sqi"].float())
    base_sqi_error = torch.abs(
        output.get("base_sqi_score", output["sqi_score"]).detach().float()
        - targets["sqi"].float()
    )
    sqi_confidence_target = (sqi_error.detach() <= 0.10).float()
    metrics = {
        "sqi_mae": sqi_error.mean(),
        "base_sqi_mae": base_sqi_error.mean(),
        "sqi_mae_improvement": base_sqi_error.mean() - sqi_error.mean(),
        "sqi_mse": torch.mean(sqi_error.square()),
        "motion_mae": _masked_l1(
            output["imu_motion_probability"],
            targets["motion"],
        ),
        "coupled_artifact_mae": _masked_l1(
            output["imu_artifact_probability"],
            targets["coupled_artifact"],
        ),
        "reliability_mae": _masked_l1(
            output["imu_reliability"],
            targets["reliability"],
        ),
        "sqi_confidence_mae": _masked_l1(
            output["sqi_confidence"],
            sqi_confidence_target,
            mask=targets["imu_present"],
        ),
        "s1s2_confidence_mae": _masked_l1(
            output["s1s2_confidence"],
            targets["s1s2_confidence"],
            mask=targets["s1s2_confidence_weight"],
        ),
    }
    motion_binary = (targets["motion"] >= 0.5).float()
    coupled_binary = (targets["coupled_artifact"] > 0.0).float()
    metrics["motion_frame_accuracy"] = torch.mean(
        (
            (output["imu_motion_probability"].float() >= 0.5)
            == (motion_binary > 0.5)
        ).float()
    )
    metrics["coupled_artifact_frame_accuracy"] = torch.mean(
        (
            (output["imu_artifact_probability"].float() >= 0.5)
            == (coupled_binary > 0.5)
        ).float()
    )
    if "base_s1s2_prob" in output:
        metrics["s1s2_location_delta"] = torch.mean(
            torch.abs(
                output["s1s2_prob"].float()
                - output["base_s1s2_prob"].detach().float()
            )
        )
    if "base_mask" in output:
        metrics["mask_delta"] = torch.mean(
            torch.abs(
                output["mask"].float() - output["base_mask"].detach().float()
            )
        )
    if "imu_artifact_type_logits" in output:
        metrics["artifact_type_accuracy"] = torch.mean(
            (
                output["imu_artifact_type_logits"].argmax(dim=-1)
                == targets["artifact_type"]
            ).float()
        )

    for mode, fallback in (fallback_outputs or {}).items():
        if "base_sqi_score" in fallback:
            metrics[f"{mode}_sqi_fallback_mae"] = torch.mean(
                torch.abs(
                    fallback["sqi_score"].float()
                    - fallback["base_sqi_score"].detach().float()
                )
            )
        metrics[f"{mode}_reliability_mean"] = torch.mean(
            fallback["imu_reliability"].float()
        )
        metrics[f"{mode}_coupled_artifact_mean"] = torch.mean(
            fallback["imu_artifact_probability"].float()
        )
        if "base_mask" in fallback:
            metrics[f"{mode}_mask_delta"] = torch.mean(
                torch.abs(
                    fallback["mask"].float()
                    - fallback["base_mask"].detach().float()
                )
            )
        else:
            metrics[f"{mode}_mask_delta"] = torch.mean(
                torch.abs(
                    fallback["mask"].float() - output["mask"].detach().float()
                )
            )
        if "base_s1s2_prob" in fallback:
            metrics[f"{mode}_s1s2_location_delta"] = torch.mean(
                torch.abs(
                    fallback["s1s2_prob"].float()
                    - fallback["base_s1s2_prob"].detach().float()
                )
            )
        else:
            metrics[f"{mode}_s1s2_location_delta"] = torch.mean(
                torch.abs(
                    fallback["s1s2_prob"].float()
                    - output["s1s2_prob"].detach().float()
                )
            )
        if "imu_artifact_probability" in fallback:
            active = (
                (targets["coupled_artifact"].float() > 0.0).float()
                * targets["reliability"].float()
            )
            separation = (
                output["imu_artifact_probability"].float()
                - fallback["imu_artifact_probability"].float()
            )
            metrics[f"{mode}_artifact_alignment_separation"] = (
                torch.sum(separation * active) / (torch.sum(active) + 1e-8)
            )
    return {name: value.detach() for name, value in metrics.items()}
