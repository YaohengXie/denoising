from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
import torch.nn.functional as F

from ecg_pcg_denoise.models import DenoisingModel
from ecg_pcg_denoise.train.torch_audio import (
    istft_waveform,
    reconstruct_from_mag,
    stft_waveform,
)
from ecg_pcg_denoise.train.train_denoise import beat_to_frames, stft_config
from ecg_pcg_denoise.utils.config import get_nested


def prepare_imu(
    batch: dict[str, Any],
    device: torch.device,
    mode: str = "correct",
) -> tuple[Tensor, Tensor]:
    features = batch["imu_feat"].to(device, non_blocking=True)
    valid = batch["imu_valid_mask"].to(device, non_blocking=True)
    features = features * valid[:, None, :]
    present = batch["imu_present"].to(device, non_blocking=True)

    if mode == "correct":
        return features, present
    if mode == "zero":
        return torch.zeros_like(features), torch.zeros_like(present)
    if mode == "shuffle":
        if features.shape[0] > 1:
            return torch.roll(features, shifts=1, dims=0), torch.roll(present, shifts=1, dims=0)
        return torch.flip(features, dims=(-1,)), present
    if mode == "shift":
        shift = max(1, features.shape[-1] // 4)
        return torch.roll(features, shifts=shift, dims=-1), present
    raise ValueError(f"Unsupported IMU mode: {mode}")


def forward_m14_batch(
    model: DenoisingModel,
    batch: dict[str, Any],
    config: dict[str, Any],
    device: torch.device,
    imu_mode: str = "correct",
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    n_fft, hop, win = stft_config(config)
    noisy = batch["noisy_pcg"].to(device, non_blocking=True)
    clean = batch["clean_pcg"].to(device, non_blocking=True)
    beat = batch["ecg_beat"].to(device, non_blocking=True)
    imu, imu_present = prepare_imu(batch, device, imu_mode)

    noisy_complex = stft_waveform(noisy, n_fft, hop, win)
    clean_complex = stft_waveform(clean, n_fft, hop, win)
    noisy_mag = noisy_complex.abs()
    clean_mag = clean_complex.abs()
    beat_frames = beat_to_frames(beat, noisy_mag.shape[-1])
    output = model(
        pcg_stft=noisy_mag,
        ecg_beat=beat_frames,
        imu_feat=imu,
        modality_mask={"pcg": 1, "ecg": 1, "imu": imu_present},
    )
    estimate_mag = output["mask"] * noisy_mag
    estimate = reconstruct_from_mag(
        noisy_complex,
        estimate_mag,
        noisy.shape[-1],
        n_fft,
        hop,
        win,
    )
    tensors = {
        "noisy": noisy,
        "clean": clean,
        "estimate": estimate,
        "noisy_mag": noisy_mag,
        "clean_mag": clean_mag,
        "estimate_mag": estimate_mag,
    }
    if "base_mask" in output:
        base_mask = output["base_mask"].detach()
        base_estimate_mag = base_mask * noisy_mag
        base_estimate = reconstruct_from_mag(
            noisy_complex,
            base_estimate_mag,
            noisy.shape[-1],
            n_fft,
            hop,
            win,
        )
        tensors["base_estimate"] = base_estimate.detach()
        tensors["base_estimate_mag"] = base_estimate_mag.detach()
    return output, tensors


def _si_sdr_db_torch(estimate: Tensor, reference: Tensor) -> Tensor:
    estimate = estimate.float()
    reference = reference.float()
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    reference = reference - reference.mean(dim=-1, keepdim=True)
    scale = torch.sum(estimate * reference, dim=-1, keepdim=True) / (
        torch.sum(reference.square(), dim=-1, keepdim=True) + 1e-8
    )
    target = scale * reference
    noise = estimate - target
    ratio = (torch.sum(target.square(), dim=-1) + 1e-8) / (
        torch.sum(noise.square(), dim=-1) + 1e-8
    )
    return 10.0 * torch.log10(ratio)


def _snr_db_torch(estimate: Tensor, reference: Tensor) -> Tensor:
    estimate = estimate.float()
    reference = reference.float()
    ratio = (torch.sum(reference.square(), dim=-1) + 1e-8) / (
        torch.sum((reference - estimate).square(), dim=-1) + 1e-8
    )
    return 10.0 * torch.log10(ratio)


def _multi_resolution_spectral_errors(
    estimate: Tensor,
    reference: Tensor,
    resolutions: tuple[tuple[int, int], ...],
) -> tuple[Tensor, Tensor]:
    estimate = estimate.float()
    reference = reference.float()
    log_errors: list[Tensor] = []
    convergence_errors: list[Tensor] = []
    for n_fft, hop_length in resolutions:
        window = torch.hann_window(
            n_fft,
            device=estimate.device,
            dtype=estimate.dtype,
        )
        estimate_mag = torch.stft(
            estimate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window=window,
            return_complex=True,
        ).abs()
        reference_mag = torch.stft(
            reference,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window=window,
            return_complex=True,
        ).abs()
        log_errors.append(
            torch.mean(
                torch.abs(torch.log1p(estimate_mag) - torch.log1p(reference_mag)),
                dim=(-2, -1),
            )
        )
        convergence_errors.append(
            torch.linalg.vector_norm(
                estimate_mag - reference_mag,
                dim=(-2, -1),
            )
            / (
                torch.linalg.vector_norm(reference_mag, dim=(-2, -1))
                + 1e-8
            )
        )
    return (
        torch.stack(log_errors, dim=0).mean(dim=0),
        torch.stack(convergence_errors, dim=0).mean(dim=0),
    )


def artifact_gate_target(
    batch: dict[str, Any],
    frames: int,
    device: torch.device,
    dtype: torch.dtype,
    positive_floor: float = 0.15,
) -> Tensor:
    envelope = batch["motion_envelope"].to(device, non_blocking=True).float()
    if envelope.ndim == 2:
        envelope = envelope[:, None, :]
    envelope = F.interpolate(
        envelope,
        size=frames,
        mode="linear",
        align_corners=False,
    ).squeeze(1)
    positive = torch.tensor(
        [
            str(mode) in {"motion_artifact", "combined"}
            for mode in batch["artifact_mode"]
        ],
        device=device,
        dtype=envelope.dtype,
    )
    positive_target = positive_floor + (1.0 - positive_floor) * envelope
    target = positive[:, None] * positive_target
    return target.to(dtype=dtype)


def m14_training_loss(
    output: dict[str, Tensor],
    tensors: dict[str, Tensor],
    batch: dict[str, Any],
    config: dict[str, Any],
    return_components: bool = False,
) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
    weights = get_nested(config, "training.loss", {})
    waveform_error = torch.mean(
        torch.abs(tensors["estimate"] - tensors["clean"]),
        dim=-1,
    )
    magnitude_error = torch.mean(
        torch.abs(
            torch.log1p(tensors["estimate_mag"])
            - torch.log1p(tensors["clean_mag"])
        ),
        dim=(-2, -1),
    )

    s1s2 = batch["s1s2_weak"].to(tensors["clean"].device, non_blocking=True)
    heart = s1s2.amax(dim=1, keepdim=True)
    heart = F.interpolate(
        heart,
        size=tensors["estimate"].shape[-1],
        mode="linear",
        align_corners=False,
    ).squeeze(1)
    heart_alpha = float(weights.get("s1s2_weight_alpha", 3.0))
    heart_weight = 1.0 + heart_alpha * heart
    weighted_waveform_error = torch.sum(
        heart_weight * torch.abs(tensors["estimate"] - tensors["clean"]),
        dim=-1,
    ) / (torch.sum(heart_weight, dim=-1) + 1e-8)

    snr = batch["snr_db"].to(tensors["clean"].device, non_blocking=True)
    high_quality = (snr >= float(weights.get("high_quality_snr_db", 10.0))).float()
    quality_multiplier = 1.0 + high_quality * (
        float(weights.get("high_quality_multiplier", 2.0)) - 1.0
    )
    per_sample_task = (
        float(weights.get("waveform_l1", 1.0)) * waveform_error
        + float(weights.get("log_mag_l1", 0.5)) * magnitude_error
        + float(weights.get("s1s2_weighted_waveform_l1", 0.5))
        * weighted_waveform_error
    )
    mr_log_weight = float(weights.get("mrstft_log_mag", 0.0))
    mr_convergence_weight = float(
        weights.get("mrstft_spectral_convergence", 0.0)
    )
    mr_log_error = torch.zeros_like(per_sample_task)
    mr_convergence_error = torch.zeros_like(per_sample_task)
    if mr_log_weight > 0.0 or mr_convergence_weight > 0.0:
        resolutions = tuple(
            (int(values[0]), int(values[1]))
            for values in weights.get(
                "mrstft_resolutions",
                ((128, 16), (512, 64)),
            )
        )
        mr_log_error, mr_convergence_error = _multi_resolution_spectral_errors(
            tensors["estimate"],
            tensors["clean"],
            resolutions,
        )
        per_sample_task = (
            per_sample_task
            + mr_log_weight * mr_log_error
            + mr_convergence_weight * mr_convergence_error
        )
    if "imu_mask_delta" in output:
        delta_magnitude = torch.mean(
            torch.abs(output["imu_mask_delta"]),
            dim=(-2, -1),
        )
        modes = [str(mode) for mode in batch["artifact_mode"]]
        preservation_weight = torch.tensor(
            [
                (
                    float(weights.get("motion_clean_delta_l1", 1.0))
                    if mode == "motion_clean"
                    else float(weights.get("independent_delta_l1", 0.25))
                    if mode == "independent_artifact"
                    else 0.0
                )
                for mode in modes
            ],
            device=tensors["clean"].device,
            dtype=per_sample_task.dtype,
        )
        high_quality_delta_weight = float(
            weights.get("high_quality_delta_l1", 0.05)
        )
        preservation_weight = preservation_weight + high_quality * high_quality_delta_weight
        per_sample_task = per_sample_task + preservation_weight * delta_magnitude

    task_loss = torch.mean(quality_multiplier * per_sample_task)
    gate_loss = torch.zeros((), device=task_loss.device, dtype=task_loss.dtype)
    gate_weight = float(weights.get("artifact_gate_bce", 0.0))
    if gate_weight > 0.0 and "imu_artifact_gate" in output:
        gate = output["imu_artifact_gate"].float().clamp(1e-6, 1.0 - 1e-6)
        gate_target = artifact_gate_target(
            batch,
            gate.shape[-1],
            gate.device,
            gate.dtype,
            positive_floor=float(weights.get("artifact_gate_positive_floor", 0.15)),
        )
        gate_loss = -torch.mean(
            gate_target * torch.log(gate)
            + (1.0 - gate_target) * torch.log1p(-gate)
        )

    distill_loss = torch.zeros((), device=task_loss.device, dtype=task_loss.dtype)
    si_guard_loss = torch.zeros_like(distill_loss)
    snr_guard_loss = torch.zeros_like(distill_loss)
    if (
        "base_estimate" in tensors
        and "base_estimate_mag" in tensors
        and "base_mask" in output
    ):
        modes = [str(mode) for mode in batch["artifact_mode"]]
        distill_strength = torch.tensor(
            [
                (
                    float(weights.get("motion_clean_distill", 0.0))
                    if mode == "motion_clean"
                    else float(weights.get("independent_distill", 0.0))
                    if mode == "independent_artifact"
                    else float(weights.get("clean_identity_distill", 0.0))
                    if mode == "clean_identity"
                    else 0.0
                )
                for mode in modes
            ],
            device=task_loss.device,
            dtype=torch.float32,
        )
        distill_strength = distill_strength + high_quality.float() * float(
            weights.get("high_quality_distill", 0.0)
        )
        clean_rms = torch.sqrt(
            torch.mean(tensors["clean"].float().square(), dim=-1) + 1e-8
        )
        distill_wave = torch.mean(
            torch.abs(
                tensors["estimate"].float()
                - tensors["base_estimate"].float()
            ),
            dim=-1,
        ) / clean_rms
        distill_mask = torch.mean(
            torch.abs(output["mask"].float() - output["base_mask"].detach().float()),
            dim=(-2, -1),
        )
        distill_log_mag = torch.mean(
            torch.abs(
                torch.log1p(tensors["estimate_mag"].float())
                - torch.log1p(tensors["base_estimate_mag"].float())
            ),
            dim=(-2, -1),
        )
        per_sample_distill = (
            float(weights.get("distill_waveform", 1.0)) * distill_wave
            + float(weights.get("distill_mask", 0.5)) * distill_mask
            + float(weights.get("distill_log_mag", 0.5)) * distill_log_mag
        )
        distill_loss = torch.mean(distill_strength * per_sample_distill)

        si_weight = float(weights.get("si_sdr_guard", 0.0))
        if si_weight > 0.0:
            estimate_si = _si_sdr_db_torch(
                tensors["estimate"],
                tensors["clean"],
            )
            base_si = _si_sdr_db_torch(
                tensors["base_estimate"],
                tensors["clean"],
            ).detach()
            si_guard_loss = torch.mean(
                F.relu(
                    base_si
                    - estimate_si
                    - float(weights.get("si_sdr_guard_tolerance_db", 0.0))
                ).clamp_max(float(weights.get("metric_guard_cap_db", 5.0)))
            )

        snr_weight = float(weights.get("snr_guard", 0.0))
        if snr_weight > 0.0:
            estimate_snr = _snr_db_torch(
                tensors["estimate"],
                tensors["clean"],
            )
            base_snr = _snr_db_torch(
                tensors["base_estimate"],
                tensors["clean"],
            ).detach()
            snr_guard_loss = torch.mean(
                F.relu(
                    base_snr
                    - estimate_snr
                    - float(weights.get("snr_guard_tolerance_db", 0.0))
                ).clamp_max(float(weights.get("metric_guard_cap_db", 5.0)))
            )

    total = (
        task_loss
        + gate_weight * gate_loss
        + distill_loss
        + float(weights.get("si_sdr_guard", 0.0)) * si_guard_loss
        + float(weights.get("snr_guard", 0.0)) * snr_guard_loss
    )
    if not return_components:
        return total
    return total, {
        "task_loss": task_loss.detach(),
        "gate_loss": gate_loss.detach(),
        "distill_loss": distill_loss.detach(),
        "si_guard_loss": si_guard_loss.detach(),
        "snr_guard_loss": snr_guard_loss.detach(),
        "mrstft_log_error": torch.mean(mr_log_error).detach(),
        "mrstft_convergence_error": torch.mean(mr_convergence_error).detach(),
    }


def reconstruct_output(
    output: dict[str, Tensor],
    noisy: Tensor,
    config: dict[str, Any],
) -> Tensor:
    n_fft, hop, win = stft_config(config)
    noisy_complex = stft_waveform(noisy, n_fft, hop, win)
    estimate_mag = output["mask"] * noisy_complex.abs()
    return istft_waveform(
        torch.polar(estimate_mag.clamp_min(0.0), torch.angle(noisy_complex)),
        noisy.shape[-1],
        n_fft,
        hop,
        win,
    )
