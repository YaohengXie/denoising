from __future__ import annotations

import torch
from torch import Tensor


def stft_waveform(x: Tensor, n_fft: int, hop_length: int, win_length: int) -> Tensor:
    window = torch.hann_window(win_length, device=x.device)
    return torch.stft(
        x,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True,
    )


def istft_waveform(z: Tensor, length: int, n_fft: int, hop_length: int, win_length: int) -> Tensor:
    window = torch.hann_window(win_length, device=z.device)
    return torch.istft(
        z,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        length=length,
    )


def reconstruct_from_mag(noisy_complex: Tensor, estimate_mag: Tensor, length: int, n_fft: int, hop: int, win: int) -> Tensor:
    phase = torch.angle(noisy_complex)
    estimate_complex = torch.polar(estimate_mag.clamp_min(0.0), phase)
    return istft_waveform(estimate_complex, length, n_fft, hop, win)


def complex_stft_features(z: Tensor, eps: float = 1e-8) -> Tensor:
    """Return magnitude, cosine-phase, and sine-phase input channels."""
    magnitude = z.abs()
    denominator = magnitude.clamp_min(eps)
    unit_real = z.real / denominator
    unit_imag = z.imag / denominator
    return torch.stack((magnitude, unit_real, unit_imag), dim=1)


def apply_polar_complex_mask(noisy_complex: Tensor, magnitude_mask: Tensor, phase_residual: Tensor) -> Tensor:
    """Apply a polar complex ratio mask to a complex STFT."""
    complex_mask = torch.polar(magnitude_mask.float().clamp_min(0.0), phase_residual.float())
    return complex_mask * noisy_complex
