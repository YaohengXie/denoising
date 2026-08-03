from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class BottleneckTransformer(nn.Module):
    def __init__(
        self,
        channels: int,
        layers: int,
        heads: int,
        dropout: float,
        use_axial_position_encoding: bool = False,
        axial_position_scale_init: float = 0.0,
    ) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=heads,
            dim_feedforward=channels * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.use_axial_position_encoding = use_axial_position_encoding
        if use_axial_position_encoding:
            self.axial_position_scale = nn.Parameter(torch.tensor(float(axial_position_scale_init)))
        else:
            self.register_parameter("axial_position_scale", None)

    def forward(self, x: Tensor) -> Tensor:
        b, c, f, t = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        if self.use_axial_position_encoding:
            frequency_position = sinusoidal_position_encoding(f, c, x.device, x.dtype)
            temporal_position = sinusoidal_position_encoding(t, c, x.device, x.dtype)
            axial_position = (
                frequency_position[:, None, :] + temporal_position[None, :, :]
            ).reshape(f * t, c)
            tokens = tokens + self.axial_position_scale * axial_position.unsqueeze(0)
        tokens = self.encoder(tokens)
        return tokens.transpose(1, 2).reshape(b, c, f, t)


def sinusoidal_position_encoding(length: int, channels: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Create a deterministic axis-specific sinusoidal position encoding."""
    position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    even_channels = torch.arange(0, channels, 2, device=device, dtype=torch.float32)
    divisor = torch.exp(-math.log(10000.0) * even_channels / max(1, channels))
    encoding = torch.zeros(length, channels, device=device, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(position * divisor)
    if channels > 1:
        encoding[:, 1::2] = torch.cos(position * divisor[: encoding[:, 1::2].shape[1]])
    return encoding.to(dtype=dtype)


class ConformerFeedForward(nn.Module):
    def __init__(self, channels: int, multiplier: int, dropout: float) -> None:
        super().__init__()
        hidden = channels * multiplier
        self.net = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, channels),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class ConformerConvModule(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("Conformer convolution kernel size must be a positive odd number.")
        self.norm = nn.LayerNorm(channels)
        self.pointwise_in = nn.Conv1d(channels, channels * 2, kernel_size=1)
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels,
        )
        self.channel_norm = nn.GroupNorm(1, channels)
        self.pointwise_out = nn.Conv1d(channels, channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        y = self.norm(x).transpose(1, 2)
        y = F.glu(self.pointwise_in(y), dim=1)
        y = self.depthwise(y)
        y = F.silu(self.channel_norm(y))
        y = self.dropout(self.pointwise_out(y))
        return y.transpose(1, 2)


class AxisConformerBlock(nn.Module):
    """Conformer block for one explicit time or frequency axis."""

    def __init__(self, channels: int, heads: int, ff_multiplier: int, conv_kernel: int, dropout: float) -> None:
        super().__init__()
        if channels % heads != 0:
            raise ValueError(f"Conformer channels ({channels}) must be divisible by heads ({heads}).")
        self.ffn1 = ConformerFeedForward(channels, ff_multiplier, dropout)
        self.attention_norm = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(channels, heads, dropout=dropout, batch_first=True)
        self.attention_dropout = nn.Dropout(dropout)
        self.conv = ConformerConvModule(channels, conv_kernel, dropout)
        self.ffn2 = ConformerFeedForward(channels, ff_multiplier, dropout)
        self.final_norm = nn.LayerNorm(channels)

    def forward(self, x: Tensor, position: Tensor) -> Tensor:
        x = x + 0.5 * self.ffn1(x)
        normalized = self.attention_norm(x)
        query_key = normalized + position.unsqueeze(0)
        attended, _ = self.attention(query_key, query_key, normalized, need_weights=False)
        x = x + self.attention_dropout(attended)
        x = x + self.conv(x)
        x = x + 0.5 * self.ffn2(x)
        return self.final_norm(x)


class DualPathTFConformerLayer(nn.Module):
    """Model frequency relations per frame, then temporal relations per band."""

    def __init__(self, channels: int, heads: int, ff_multiplier: int, conv_kernel: int, dropout: float) -> None:
        super().__init__()
        self.frequency = AxisConformerBlock(channels, heads, ff_multiplier, conv_kernel, dropout)
        self.temporal = AxisConformerBlock(channels, heads, ff_multiplier, conv_kernel, dropout)

    def forward(self, x: Tensor) -> Tensor:
        batch, channels, frequencies, frames = x.shape
        frequency_position = sinusoidal_position_encoding(frequencies, channels, x.device, x.dtype)
        temporal_position = sinusoidal_position_encoding(frames, channels, x.device, x.dtype)

        y = x.permute(0, 3, 2, 1).reshape(batch * frames, frequencies, channels)
        y = self.frequency(y, frequency_position)
        y = y.reshape(batch, frames, frequencies, channels).permute(0, 3, 2, 1)

        y = y.permute(0, 2, 3, 1).reshape(batch * frequencies, frames, channels)
        y = self.temporal(y, temporal_position)
        return y.reshape(batch, frequencies, frames, channels).permute(0, 3, 1, 2)


class BottleneckTFConformer(nn.Module):
    """Residual dual-path Conformer with an identity-preserving initialization."""

    def __init__(
        self,
        channels: int,
        layers: int,
        heads: int,
        ff_multiplier: int,
        conv_kernel: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                DualPathTFConformerLayer(channels, heads, ff_multiplier, conv_kernel, dropout)
                for _ in range(layers)
            ]
        )
        self.output_projection = nn.Conv2d(channels, channels, kernel_size=1)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, x: Tensor) -> Tensor:
        y = x
        for layer in self.layers:
            y = layer(y)
        return x + self.output_projection(y)


class BottleneckBiLSTM(nn.Module):
    """Temporal BiLSTM over bottleneck features.

    Each frequency bin is treated as one sequence over time, with channels as
    features. The residual projection starts at zero so a newly inserted BiLSTM
    behaves like an identity layer before fine-tuning.
    """

    def __init__(self, channels: int, hidden_size: int | None, layers: int, dropout: float) -> None:
        super().__init__()
        hidden = hidden_size if hidden_size is not None else max(1, channels // 2)
        self.lstm = nn.LSTM(
            input_size=channels,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.proj = nn.Linear(hidden * 2, channels)
        self.dropout = nn.Dropout(dropout)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: Tensor) -> Tensor:
        b, c, f, t = x.shape
        tokens = x.permute(0, 2, 3, 1).reshape(b * f, t, c)
        y, _ = self.lstm(tokens)
        y = self.dropout(self.proj(y))
        y = tokens + y
        return y.reshape(b, f, t, c).permute(0, 3, 1, 2)


class ECGCrossAttention(nn.Module):
    """Cross-attend PCG bottleneck tokens to an ECG-derived beat sequence.

    The output projection is zero-initialized so adding the module to a
    trained checkpoint starts as an exact identity mapping.
    """

    def __init__(self, channels: int, heads: int = 4, conv_kernel: int = 7, dropout: float = 0.1) -> None:
        super().__init__()
        if channels % heads != 0:
            raise ValueError(f"channels={channels} must be divisible by heads={heads}")
        if conv_kernel % 2 == 0:
            raise ValueError("ECG cross-attention conv_kernel must be odd.")
        padding = conv_kernel // 2
        self.ecg_encoder = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=conv_kernel, padding=padding),
            nn.SiLU(),
            nn.Conv1d(channels, channels, kernel_size=conv_kernel, padding=padding, groups=channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, kernel_size=1),
        )
        self.query_norm = nn.LayerNorm(channels)
        self.context_norm = nn.LayerNorm(channels)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(channels, channels)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, x: Tensor, beat: Tensor) -> Tensor:
        b, c, f, t = x.shape
        pcg_tokens = x.permute(0, 2, 3, 1).reshape(b, f * t, c)
        frequency_position = sinusoidal_position_encoding(f, c, x.device, x.dtype)
        temporal_position = sinusoidal_position_encoding(t, c, x.device, x.dtype)
        pcg_position = (frequency_position[:, None] + temporal_position[None, :]).reshape(f * t, c)

        beat_sequence = beat.float().mean(dim=2)
        beat_sequence = F.interpolate(beat_sequence, size=t, mode="linear", align_corners=False)
        ecg_tokens = self.ecg_encoder(beat_sequence).transpose(1, 2).to(dtype=x.dtype)
        query = self.query_norm(pcg_tokens) + pcg_position.unsqueeze(0)
        context = self.context_norm(ecg_tokens)
        attended, _ = self.cross_attention(
            query=query,
            key=context + temporal_position.unsqueeze(0),
            value=context,
            need_weights=False,
        )
        output = pcg_tokens + self.output_projection(self.dropout(attended))
        return output.reshape(b, f, t, c).permute(0, 3, 1, 2)


class ECGSkipGate(nn.Module):
    """ECG-conditioned residual gate for U-Net skip features.

    The projection is zero-initialized, so the module starts as an identity
    mapping when inserted into an already trained checkpoint.
    """

    def __init__(self, channels: int, strength: float = 0.5) -> None:
        super().__init__()
        self.proj = nn.Conv2d(1, channels, kernel_size=1)
        self.strength = float(strength)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: Tensor, beat: Tensor) -> Tensor:
        beat = F.interpolate(beat.float(), size=x.shape[-2:], mode="bilinear", align_corners=False)
        gate = torch.tanh(self.proj(beat))
        return x * (1.0 + self.strength * gate)


class ECGFiLM(nn.Module):
    """ECG-conditioned feature modulation.

    ECG beat maps produce feature-wise scale and bias terms for PCG features.
    The final projection is zero-initialized, so the module starts as an
    identity mapping when added to a trained denoising checkpoint.
    """

    def __init__(self, channels: int, hidden_channels: int = 16, strength: float = 0.25) -> None:
        super().__init__()
        self.strength = float(strength)
        self.net = nn.Sequential(
            nn.Conv2d(1, hidden_channels, kernel_size=(3, 7), padding=(1, 3)),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=(3, 7), padding=(1, 3)),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, channels * 2, kernel_size=1),
        )
        final = self.net[-1]
        if isinstance(final, nn.Conv2d):
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    def forward(self, x: Tensor, beat: Tensor) -> Tensor:
        beat = F.interpolate(beat.float(), size=x.shape[-2:], mode="bilinear", align_corners=False)
        gamma, beta = self.net(beat).chunk(2, dim=1)
        gamma = self.strength * torch.tanh(gamma)
        beta = self.strength * torch.tanh(beta)
        return x * (1.0 + gamma) + beta


class IMULateFusionAdapter(nn.Module):
    """Bounded IMU correction applied only to the final magnitude-mask logits.

    The PCG-dependent frequency projection starts at exactly zero. Therefore a
    newly added adapter is an exact M7 identity at initialization, while the
    IMU encoder learns a temporal motion gate after the projection begins to
    move away from zero.
    """

    def __init__(
        self,
        decoder_channels: int,
        imu_channels: int = 6,
        hidden_channels: int = 32,
        max_logit_delta: float = 2.0,
        use_joint_artifact_gate: bool = False,
        joint_gate_hidden_channels: int = 16,
        artifact_gate_bias: float = -3.0,
        residual_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if hidden_channels < 4:
            raise ValueError("IMU hidden_channels must be at least 4.")
        if joint_gate_hidden_channels < 4:
            raise ValueError("IMU joint_gate_hidden_channels must be at least 4.")
        groups = 4 if hidden_channels % 4 == 0 else 1
        self.imu_channels = int(imu_channels)
        self.max_logit_delta = float(max_logit_delta)
        self.use_joint_artifact_gate = bool(use_joint_artifact_gate)
        self.residual_scale = float(residual_scale)
        self.imu_encoder = nn.Sequential(
            nn.Conv1d(imu_channels, hidden_channels, kernel_size=7, padding=3),
            nn.GroupNorm(groups, hidden_channels),
            nn.SiLU(),
            nn.Conv1d(
                hidden_channels,
                hidden_channels,
                kernel_size=5,
                padding=2,
                groups=hidden_channels,
            ),
            nn.GroupNorm(groups, hidden_channels),
            nn.SiLU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1),
            nn.SiLU(),
        )
        self.temporal_gate = nn.Conv1d(hidden_channels, 1, kernel_size=1)
        if self.use_joint_artifact_gate:
            gate_groups = 4 if joint_gate_hidden_channels % 4 == 0 else 1
            self.artifact_gate_net = nn.Sequential(
                nn.Conv1d(
                    hidden_channels + decoder_channels + 1,
                    joint_gate_hidden_channels,
                    kernel_size=3,
                    padding=1,
                ),
                nn.GroupNorm(gate_groups, joint_gate_hidden_channels),
                nn.SiLU(),
                nn.Conv1d(joint_gate_hidden_channels, 1, kernel_size=1),
            )
            final_gate = self.artifact_gate_net[-1]
            if isinstance(final_gate, nn.Conv1d):
                nn.init.zeros_(final_gate.weight)
                nn.init.constant_(final_gate.bias, float(artifact_gate_bias))
        else:
            self.artifact_gate_net = None
        self.frequency_delta = nn.Conv2d(decoder_channels, 1, kernel_size=1)
        nn.init.zeros_(self.frequency_delta.weight)
        nn.init.zeros_(self.frequency_delta.bias)

    @staticmethod
    def _presence(
        modality_mask: dict[str, int | Tensor] | None,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        value: int | Tensor = 1 if modality_mask is None else modality_mask.get("imu", 1)
        if isinstance(value, Tensor):
            presence = value.to(device=device, dtype=dtype).reshape(batch_size, -1)
            presence = presence[:, :1]
        else:
            presence = torch.full(
                (batch_size, 1),
                float(value),
                device=device,
                dtype=dtype,
            )
        return presence.clamp(0.0, 1.0)

    def forward(
        self,
        decoder_features: Tensor,
        imu_feat: Tensor | None,
        modality_mask: dict[str, int | Tensor] | None,
    ) -> Tensor:
        delta, _ = self.forward_with_gate(
            decoder_features,
            imu_feat,
            modality_mask,
            beat_map=None,
        )
        return delta

    def forward_with_gate(
        self,
        decoder_features: Tensor,
        imu_feat: Tensor | None,
        modality_mask: dict[str, int | Tensor] | None,
        beat_map: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        batch, _, frequencies, frames = decoder_features.shape
        if imu_feat is None:
            delta = torch.zeros(
                batch,
                frequencies,
                frames,
                device=decoder_features.device,
                dtype=decoder_features.dtype,
            )
            gate = torch.zeros(
                batch,
                frames,
                device=decoder_features.device,
                dtype=decoder_features.dtype,
            )
            return delta, gate
        if imu_feat.ndim != 3 or imu_feat.shape[0] != batch:
            raise ValueError(f"imu_feat must be [B,C,T], got {imu_feat.shape}")
        if imu_feat.shape[1] != self.imu_channels:
            raise ValueError(
                f"Expected {self.imu_channels} IMU features, got {imu_feat.shape[1]}"
            )

        imu = imu_feat.to(device=decoder_features.device, dtype=decoder_features.dtype)
        imu = F.interpolate(imu, size=frames, mode="linear", align_corners=False)
        imu_encoded = self.imu_encoder(imu)
        temporal_gate = torch.tanh(self.temporal_gate(imu_encoded))
        frequency_delta = torch.tanh(self.frequency_delta(decoder_features))
        presence = self._presence(
            modality_mask,
            batch,
            decoder_features.device,
            decoder_features.dtype,
        )
        if self.use_joint_artifact_gate:
            if self.artifact_gate_net is None:
                raise RuntimeError("Joint IMU artifact gate is not initialized.")
            pcg_context = decoder_features.mean(dim=2)
            if beat_map is None:
                ecg_context = torch.zeros(
                    batch,
                    1,
                    frames,
                    device=decoder_features.device,
                    dtype=decoder_features.dtype,
                )
            else:
                beat = beat_map.to(
                    device=decoder_features.device,
                    dtype=decoder_features.dtype,
                )
                beat = F.interpolate(
                    beat,
                    size=decoder_features.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                ecg_context = beat.mean(dim=2)
            artifact_gate = torch.sigmoid(
                self.artifact_gate_net(
                    torch.cat((imu_encoded, pcg_context, ecg_context), dim=1)
                )
            )
        else:
            artifact_gate = torch.ones_like(temporal_gate)
        artifact_gate = artifact_gate * presence[:, :, None]
        temporal_multiplier = (
            artifact_gate
            if self.use_joint_artifact_gate
            else temporal_gate * presence[:, :, None]
        )
        delta = (
            self.residual_scale
            * self.max_logit_delta
            * frequency_delta
            * temporal_multiplier[:, :, None, :]
        )
        return delta.squeeze(1), artifact_gate.squeeze(1)


class IMUAuxiliaryAdapter(nn.Module):
    """Late-fuse IMU information into quality and event-confidence outputs.

    This adapter never changes the denoising mask or the S1/S2 location
    probabilities. Its SQI contribution is a bounded, zero-initialized logit
    residual, so adding it to a trained M7 checkpoint starts as an exact
    identity. A missing IMU modality produces a zero SQI residual while the
    confidence heads remain calibratable from PCG/ECG context and an explicit
    zero-presence indicator.
    """

    def __init__(
        self,
        decoder_channels: int,
        imu_channels: int = 6,
        hidden_channels: int = 32,
        joint_hidden_channels: int = 16,
        max_sqi_logit_delta: float = 1.0,
        artifact_bias: float = -3.0,
    ) -> None:
        super().__init__()
        if hidden_channels < 4:
            raise ValueError("IMU auxiliary hidden_channels must be at least 4.")
        if joint_hidden_channels < 4:
            raise ValueError("IMU auxiliary joint_hidden_channels must be at least 4.")
        imu_groups = 4 if hidden_channels % 4 == 0 else 1
        joint_groups = 4 if joint_hidden_channels % 4 == 0 else 1
        self.imu_channels = int(imu_channels)
        self.hidden_channels = int(hidden_channels)
        self.max_sqi_logit_delta = float(max_sqi_logit_delta)
        self.imu_encoder = nn.Sequential(
            nn.Conv1d(imu_channels, hidden_channels, kernel_size=7, padding=3),
            nn.GroupNorm(imu_groups, hidden_channels),
            nn.SiLU(),
            nn.Conv1d(
                hidden_channels,
                hidden_channels,
                kernel_size=5,
                padding=2,
                groups=hidden_channels,
            ),
            nn.GroupNorm(imu_groups, hidden_channels),
            nn.SiLU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1),
            nn.SiLU(),
        )
        self.motion_head = nn.Conv1d(hidden_channels, 1, kernel_size=1)
        self.reliability_head = nn.Conv1d(hidden_channels, 1, kernel_size=1)
        self.joint_encoder = nn.Sequential(
            nn.Conv1d(
                hidden_channels + decoder_channels + 2,
                joint_hidden_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.GroupNorm(joint_groups, joint_hidden_channels),
            nn.SiLU(),
            nn.Conv1d(
                joint_hidden_channels,
                joint_hidden_channels,
                kernel_size=3,
                padding=1,
                groups=joint_hidden_channels,
            ),
            nn.GroupNorm(joint_groups, joint_hidden_channels),
            nn.SiLU(),
        )
        self.artifact_head = nn.Conv1d(joint_hidden_channels, 1, kernel_size=1)
        nn.init.zeros_(self.artifact_head.weight)
        nn.init.constant_(self.artifact_head.bias, float(artifact_bias))
        self.s1s2_confidence_head = nn.Conv1d(joint_hidden_channels, 2, kernel_size=1)
        self.sqi_confidence_head = nn.Linear(joint_hidden_channels, 1)
        nn.init.zeros_(self.s1s2_confidence_head.weight)
        nn.init.zeros_(self.s1s2_confidence_head.bias)
        nn.init.zeros_(self.sqi_confidence_head.weight)
        nn.init.zeros_(self.sqi_confidence_head.bias)
        self.sqi_residual_head = nn.Linear(joint_hidden_channels, 1)
        nn.init.zeros_(self.sqi_residual_head.weight)
        nn.init.zeros_(self.sqi_residual_head.bias)

    @staticmethod
    def _presence(
        modality_mask: dict[str, int | Tensor] | None,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        value: int | Tensor = 1 if modality_mask is None else modality_mask.get("imu", 1)
        if isinstance(value, Tensor):
            presence = value.to(device=device, dtype=dtype).reshape(batch_size, -1)
            presence = presence[:, :1]
        else:
            presence = torch.full(
                (batch_size, 1),
                float(value),
                device=device,
                dtype=dtype,
            )
        return presence.clamp(0.0, 1.0)

    @staticmethod
    def _ecg_context(
        beat_map: Tensor | None,
        decoder_features: Tensor,
    ) -> Tensor:
        batch, _, _, frames = decoder_features.shape
        if beat_map is None:
            return torch.zeros(
                batch,
                1,
                frames,
                device=decoder_features.device,
                dtype=decoder_features.dtype,
            )
        beat = beat_map.to(
            device=decoder_features.device,
            dtype=decoder_features.dtype,
        )
        beat = F.interpolate(
            beat,
            size=decoder_features.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return beat.mean(dim=2)

    def forward(
        self,
        decoder_features: Tensor,
        imu_feat: Tensor | None,
        modality_mask: dict[str, int | Tensor] | None,
        beat_map: Tensor | None,
    ) -> dict[str, Tensor]:
        batch, _, _, frames = decoder_features.shape
        presence = self._presence(
            modality_mask,
            batch,
            decoder_features.device,
            decoder_features.dtype,
        )
        if imu_feat is None:
            presence = torch.zeros_like(presence)
        presence_frame = presence[:, :, None]
        if imu_feat is None:
            imu_encoded = torch.zeros(
                batch,
                self.hidden_channels,
                frames,
                device=decoder_features.device,
                dtype=decoder_features.dtype,
            )
        else:
            if imu_feat.ndim != 3 or imu_feat.shape[0] != batch:
                raise ValueError(f"imu_feat must be [B,C,T], got {imu_feat.shape}")
            if imu_feat.shape[1] != self.imu_channels:
                raise ValueError(
                    f"Expected {self.imu_channels} IMU features, got {imu_feat.shape[1]}"
                )
            imu = imu_feat.to(
                device=decoder_features.device,
                dtype=decoder_features.dtype,
            )
            imu = F.interpolate(imu, size=frames, mode="linear", align_corners=False)
            imu_encoded = self.imu_encoder(imu)
        motion_probability = torch.sigmoid(self.motion_head(imu_encoded)) * presence_frame
        reliability = torch.sigmoid(self.reliability_head(imu_encoded)) * presence_frame

        pcg_context = decoder_features.mean(dim=2)
        ecg_context = self._ecg_context(beat_map, decoder_features)
        presence_context = presence_frame.expand(-1, -1, frames)
        joint = self.joint_encoder(
            torch.cat(
                (
                    imu_encoded * presence_frame,
                    pcg_context,
                    ecg_context,
                    presence_context,
                ),
                dim=1,
            )
        )
        artifact_probability = torch.sigmoid(self.artifact_head(joint)) * presence_frame
        s1s2_confidence = torch.sigmoid(self.s1s2_confidence_head(joint))
        pooled_joint = joint.mean(dim=-1)
        sqi_confidence = torch.sigmoid(
            self.sqi_confidence_head(pooled_joint)
        ).squeeze(1)
        presence_scalar = presence.squeeze(1)
        mean_reliability = reliability.mean(dim=-1).squeeze(1)
        mean_artifact_probability = artifact_probability.mean(dim=-1).squeeze(1)
        sqi_logit_delta = (
            self.max_sqi_logit_delta
            * torch.tanh(self.sqi_residual_head(pooled_joint)).squeeze(1)
            * mean_reliability
            * mean_artifact_probability
            * presence_scalar
        )
        return {
            "imu_motion_probability": motion_probability.squeeze(1),
            "imu_artifact_probability": artifact_probability.squeeze(1),
            "imu_reliability": reliability.squeeze(1),
            "sqi_confidence": sqi_confidence,
            "s1s2_confidence": s1s2_confidence,
            "imu_sqi_logit_delta": sqi_logit_delta,
        }


class ComplexInputAdapter(nn.Module):
    """Project magnitude and unit-phase features to one PCG feature map.

    The adapter starts as an exact magnitude passthrough. This lets M10 load a
    magnitude-only checkpoint while the new phase channels and projection are
    learned during fine-tuning.
    """

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Conv2d(3, 1, kernel_size=1, bias=False)
        nn.init.zeros_(self.proj.weight)
        with torch.no_grad():
            self.proj.weight[0, 0, 0, 0] = 1.0

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f"Complex STFT features must be [B,3,F,T], got {x.shape}")
        return self.proj(x)


def resize_to(x: Tensor, target: Tensor) -> Tensor:
    return F.interpolate(x, size=target.shape[-2:], mode="bilinear", align_corners=False)


@dataclass
class DenoisingOutput:
    mask: Tensor
    denoised_mag: Tensor
    s1s2_prob: Tensor
    sqi_score: Tensor
    phase_residual: Tensor
    complex_mask_real: Tensor
    complex_mask_imag: Tensor

    def as_dict(self) -> dict[str, Tensor]:
        return {
            "mask": self.mask,
            "denoised_mag": self.denoised_mag,
            "s1s2_prob": self.s1s2_prob,
            "sqi_score": self.sqi_score,
            "phase_residual": self.phase_residual,
            "complex_mask_real": self.complex_mask_real,
            "complex_mask_imag": self.complex_mask_imag,
        }


class PureTransformerDenoisingModel(nn.Module):
    """Patch-based Transformer denoiser without a convolutional U-Net path."""

    def __init__(
        self,
        input_channels: int = 1,
        use_ecg: bool = False,
        model_dim: int = 128,
        layers: int = 4,
        heads: int = 4,
        ff_multiplier: int = 2,
        patch_frequency: int = 8,
        patch_time: int = 8,
        output_channels: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if model_dim % heads != 0:
            raise ValueError(f"model_dim={model_dim} must be divisible by heads={heads}")
        if patch_frequency <= 0 or patch_time <= 0:
            raise ValueError("Transformer patch sizes must be positive.")
        self.use_ecg = use_ecg
        self.use_complex_mask = False
        self.patch_frequency = int(patch_frequency)
        self.patch_time = int(patch_time)
        self.patch_embed = nn.Conv2d(
            input_channels,
            model_dim,
            kernel_size=(self.patch_frequency, self.patch_time),
            stride=(self.patch_frequency, self.patch_time),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=model_dim * ff_multiplier,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.output_projection = nn.Sequential(
            nn.Conv2d(model_dim, output_channels, kernel_size=1),
            nn.SiLU(),
        )
        self.mask_head = nn.Sequential(nn.Conv2d(output_channels, 1, kernel_size=1), nn.Sigmoid())
        self.s1s2_head = nn.Sequential(nn.Conv2d(output_channels, 2, kernel_size=1), nn.Sigmoid())
        self.sqi_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(output_channels, output_channels),
            nn.SiLU(),
            nn.Linear(output_channels, 1),
            nn.Sigmoid(),
        )

    @staticmethod
    def _beat_map(ecg_beat: Tensor | None, reference: Tensor) -> Tensor:
        if ecg_beat is None:
            return torch.zeros(
                reference.shape[0],
                1,
                reference.shape[-2],
                reference.shape[-1],
                device=reference.device,
                dtype=reference.dtype,
            )
        if ecg_beat.ndim == 2:
            beat = ecg_beat[:, None, None, :]
        elif ecg_beat.ndim == 3:
            beat = ecg_beat[:, None, :, :]
        elif ecg_beat.ndim == 4:
            beat = ecg_beat
        else:
            raise ValueError(f"Unsupported ecg_beat shape: {ecg_beat.shape}")
        return F.interpolate(
            beat.to(device=reference.device, dtype=reference.dtype),
            size=reference.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    def forward(
        self,
        pcg_stft: Tensor,
        ecg_beat: Tensor | None = None,
        imu_feat: Tensor | None = None,
        modality_mask: dict[str, int] | None = None,
    ) -> dict[str, Tensor]:
        del imu_feat, modality_mask
        if pcg_stft.ndim == 3:
            magnitude = pcg_stft
            x = pcg_stft.unsqueeze(1)
        elif pcg_stft.ndim == 4 and pcg_stft.shape[1] == 1:
            magnitude = pcg_stft[:, 0]
            x = pcg_stft
        else:
            raise ValueError(f"Magnitude input must be [B,F,T] or [B,1,F,T], got {pcg_stft.shape}")
        if self.use_ecg:
            x = torch.cat([x, self._beat_map(ecg_beat, x)], dim=1)

        original_size = x.shape[-2:]
        pad_frequency = (-original_size[0]) % self.patch_frequency
        pad_time = (-original_size[1]) % self.patch_time
        patches = self.patch_embed(F.pad(x, (0, pad_time, 0, pad_frequency)))
        batch, channels, frequencies, frames = patches.shape
        frequency_position = sinusoidal_position_encoding(frequencies, channels, x.device, x.dtype)
        temporal_position = sinusoidal_position_encoding(frames, channels, x.device, x.dtype)
        position = (frequency_position[:, None, :] + temporal_position[None, :, :]).reshape(
            frequencies * frames,
            channels,
        )
        tokens = patches.flatten(2).transpose(1, 2) + position.unsqueeze(0)
        encoded = self.encoder(tokens).transpose(1, 2).reshape(batch, channels, frequencies, frames)
        features = self.output_projection(encoded)
        features = F.interpolate(features, size=original_size, mode="bilinear", align_corners=False)

        mask = self.mask_head(features).squeeze(1)
        phase_residual = torch.zeros_like(mask)
        s1s2_prob = self.s1s2_head(features).mean(dim=2)
        sqi_score = self.sqi_head(features).squeeze(1)
        return DenoisingOutput(
            mask=mask,
            denoised_mag=mask * magnitude,
            s1s2_prob=s1s2_prob,
            sqi_score=sqi_score,
            phase_residual=phase_residual,
            complex_mask_real=mask,
            complex_mask_imag=torch.zeros_like(mask),
        ).as_dict()


class DenoisingModel(nn.Module):
    """TF U-Net with optional ECG beat-map conditioning and bottleneck Transformer.

    The public forward signature reserves IMU arguments for the MotemaSens extension.
    In v1, an absent IMU branch is treated as neutral conditioning.
    """

    def __init__(
        self,
        input_channels: int = 1,
        use_ecg: bool = False,
        use_transformer: bool = False,
        base_channels: int = 16,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        use_axial_position_encoding: bool = False,
        axial_position_scale_init: float = 0.0,
        use_tf_conformer: bool = False,
        tf_conformer_layers: int = 2,
        tf_conformer_heads: int = 4,
        tf_conformer_ff_multiplier: int = 2,
        tf_conformer_conv_kernel: int = 7,
        use_bilstm: bool = False,
        bilstm_layers: int = 1,
        bilstm_hidden_size: int | None = None,
        use_ecg_cross_attention: bool = False,
        ecg_cross_heads: int = 4,
        ecg_cross_conv_kernel: int = 7,
        use_ecg_attention: bool = False,
        ecg_attention_strength: float = 0.5,
        use_ecg_film: bool = False,
        ecg_film_hidden_channels: int = 16,
        ecg_film_strength: float = 0.25,
        use_imu: bool = False,
        imu_input_channels: int = 6,
        imu_hidden_channels: int = 32,
        imu_max_logit_delta: float = 2.0,
        imu_use_joint_artifact_gate: bool = False,
        imu_joint_gate_hidden_channels: int = 16,
        imu_artifact_gate_bias: float = -3.0,
        imu_residual_scale: float = 1.0,
        use_imu_aux: bool = False,
        imu_aux_max_sqi_logit_delta: float = 1.0,
        use_complex_mask: bool = False,
        phase_residual_limit: float = 3.141592653589793,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.use_ecg = use_ecg
        self.use_ecg_cross_attention = use_ecg_cross_attention
        self.use_ecg_attention = use_ecg_attention
        self.use_ecg_film = use_ecg_film
        self.use_imu = use_imu
        self.use_imu_aux = use_imu_aux
        self.use_complex_mask = use_complex_mask
        self.phase_residual_limit = float(phase_residual_limit)
        self.complex_input_adapter = ComplexInputAdapter() if use_complex_mask else None
        channels = input_channels
        self.enc1 = ConvBlock(channels, base_channels, dropout=0.0)
        self.enc2 = ConvBlock(base_channels, base_channels * 2, dropout=dropout)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4, dropout=dropout)
        self.bottleneck = ConvBlock(base_channels * 4, base_channels * 8, dropout=dropout)
        self.pool = nn.MaxPool2d(2, ceil_mode=True)
        self.transformer = (
            BottleneckTransformer(
                base_channels * 8,
                transformer_layers,
                transformer_heads,
                dropout,
                use_axial_position_encoding=use_axial_position_encoding,
                axial_position_scale_init=axial_position_scale_init,
            )
            if use_transformer
            else nn.Identity()
        )
        self.tf_conformer = (
            BottleneckTFConformer(
                channels=base_channels * 8,
                layers=tf_conformer_layers,
                heads=tf_conformer_heads,
                ff_multiplier=tf_conformer_ff_multiplier,
                conv_kernel=tf_conformer_conv_kernel,
                dropout=dropout,
            )
            if use_tf_conformer
            else nn.Identity()
        )
        self.temporal_bilstm = (
            BottleneckBiLSTM(base_channels * 8, bilstm_hidden_size, bilstm_layers, dropout)
            if use_bilstm
            else nn.Identity()
        )
        self.ecg_cross_attention = (
            ECGCrossAttention(
                channels=base_channels * 8,
                heads=ecg_cross_heads,
                conv_kernel=ecg_cross_conv_kernel,
                dropout=dropout,
            )
            if use_ecg_cross_attention
            else nn.Identity()
        )
        if use_ecg_attention:
            self.ecg_gate1 = ECGSkipGate(base_channels, strength=ecg_attention_strength)
            self.ecg_gate2 = ECGSkipGate(base_channels * 2, strength=ecg_attention_strength)
            self.ecg_gate3 = ECGSkipGate(base_channels * 4, strength=ecg_attention_strength)
        else:
            self.ecg_gate1 = None
            self.ecg_gate2 = None
            self.ecg_gate3 = None
        if use_ecg_film:
            self.ecg_film_bottleneck = ECGFiLM(
                base_channels * 8,
                hidden_channels=ecg_film_hidden_channels,
                strength=ecg_film_strength,
            )
            self.ecg_film_dec3 = ECGFiLM(
                base_channels * 4,
                hidden_channels=ecg_film_hidden_channels,
                strength=ecg_film_strength,
            )
        else:
            self.ecg_film_bottleneck = None
            self.ecg_film_dec3 = None
        self.dec3 = ConvBlock(base_channels * 8 + base_channels * 4, base_channels * 4, dropout=dropout)
        self.dec2 = ConvBlock(base_channels * 4 + base_channels * 2, base_channels * 2, dropout=dropout)
        self.dec1 = ConvBlock(base_channels * 2 + base_channels, base_channels, dropout=0.0)
        self.mask_head = nn.Sequential(nn.Conv2d(base_channels, 1, kernel_size=1), nn.Sigmoid())
        self.imu_adapter = (
            IMULateFusionAdapter(
                decoder_channels=base_channels,
                imu_channels=imu_input_channels,
                hidden_channels=imu_hidden_channels,
                max_logit_delta=imu_max_logit_delta,
                use_joint_artifact_gate=imu_use_joint_artifact_gate,
                joint_gate_hidden_channels=imu_joint_gate_hidden_channels,
                artifact_gate_bias=imu_artifact_gate_bias,
                residual_scale=imu_residual_scale,
            )
            if use_imu
            else None
        )
        self.imu_aux_adapter = (
            IMUAuxiliaryAdapter(
                decoder_channels=base_channels,
                imu_channels=imu_input_channels,
                hidden_channels=imu_hidden_channels,
                joint_hidden_channels=imu_joint_gate_hidden_channels,
                max_sqi_logit_delta=imu_aux_max_sqi_logit_delta,
                artifact_bias=imu_artifact_gate_bias,
            )
            if use_imu_aux
            else None
        )
        self.phase_head = nn.Conv2d(base_channels, 1, kernel_size=1) if use_complex_mask else None
        if self.phase_head is not None:
            nn.init.zeros_(self.phase_head.weight)
            nn.init.zeros_(self.phase_head.bias)
        self.s1s2_head = nn.Sequential(nn.Conv2d(base_channels, 2, kernel_size=1), nn.Sigmoid())
        self.sqi_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_channels, base_channels),
            nn.SiLU(),
            nn.Linear(base_channels, 1),
            nn.Sigmoid(),
        )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "DenoisingModel | PureTransformerDenoisingModel":
        model_config = config.get("model", {})
        architecture = str(model_config.get("architecture", "unet"))
        if architecture == "pure_transformer":
            return PureTransformerDenoisingModel(
                input_channels=int(model_config.get("input_channels", 1)),
                use_ecg=bool(model_config.get("use_ecg", False)),
                model_dim=int(model_config.get("pure_transformer_dim", 128)),
                layers=int(model_config.get("pure_transformer_layers", 4)),
                heads=int(model_config.get("pure_transformer_heads", 4)),
                ff_multiplier=int(model_config.get("pure_transformer_ff_multiplier", 2)),
                patch_frequency=int(model_config.get("pure_transformer_patch_frequency", 8)),
                patch_time=int(model_config.get("pure_transformer_patch_time", 8)),
                output_channels=int(model_config.get("base_channels", 16)),
                dropout=float(model_config.get("dropout", 0.1)),
            )
        if architecture != "unet":
            raise ValueError(f"Unsupported model architecture: {architecture}")
        return cls(
            input_channels=int(model_config.get("input_channels", 1)),
            use_ecg=bool(model_config.get("use_ecg", False)),
            use_transformer=bool(model_config.get("use_transformer", False)),
            base_channels=int(model_config.get("base_channels", 16)),
            transformer_layers=int(model_config.get("transformer_layers", 2)),
            transformer_heads=int(model_config.get("transformer_heads", 4)),
            use_axial_position_encoding=bool(model_config.get("use_axial_position_encoding", False)),
            axial_position_scale_init=float(model_config.get("axial_position_scale_init", 0.0)),
            use_tf_conformer=bool(model_config.get("use_tf_conformer", False)),
            tf_conformer_layers=int(model_config.get("tf_conformer_layers", 2)),
            tf_conformer_heads=int(model_config.get("tf_conformer_heads", 4)),
            tf_conformer_ff_multiplier=int(model_config.get("tf_conformer_ff_multiplier", 2)),
            tf_conformer_conv_kernel=int(model_config.get("tf_conformer_conv_kernel", 7)),
            use_bilstm=bool(model_config.get("use_bilstm", False)),
            bilstm_layers=int(model_config.get("bilstm_layers", 1)),
            bilstm_hidden_size=(
                int(model_config["bilstm_hidden_size"]) if model_config.get("bilstm_hidden_size") is not None else None
            ),
            use_ecg_cross_attention=bool(model_config.get("use_ecg_cross_attention", False)),
            ecg_cross_heads=int(model_config.get("ecg_cross_heads", 4)),
            ecg_cross_conv_kernel=int(model_config.get("ecg_cross_conv_kernel", 7)),
            use_ecg_attention=bool(model_config.get("use_ecg_attention", False)),
            ecg_attention_strength=float(model_config.get("ecg_attention_strength", 0.5)),
            use_ecg_film=bool(model_config.get("use_ecg_film", False)),
            ecg_film_hidden_channels=int(model_config.get("ecg_film_hidden_channels", 16)),
            ecg_film_strength=float(model_config.get("ecg_film_strength", 0.25)),
            use_imu=bool(model_config.get("use_imu", False)),
            imu_input_channels=int(model_config.get("imu_input_channels", 6)),
            imu_hidden_channels=int(model_config.get("imu_hidden_channels", 32)),
            imu_max_logit_delta=float(model_config.get("imu_max_logit_delta", 2.0)),
            imu_use_joint_artifact_gate=bool(
                model_config.get("imu_use_joint_artifact_gate", False)
            ),
            imu_joint_gate_hidden_channels=int(
                model_config.get("imu_joint_gate_hidden_channels", 16)
            ),
            imu_artifact_gate_bias=float(
                model_config.get("imu_artifact_gate_bias", -3.0)
            ),
            imu_residual_scale=float(model_config.get("imu_residual_scale", 1.0)),
            use_imu_aux=bool(model_config.get("use_imu_aux", False)),
            imu_aux_max_sqi_logit_delta=float(
                model_config.get("imu_aux_max_sqi_logit_delta", 1.0)
            ),
            use_complex_mask=bool(model_config.get("use_complex_mask", False)),
            phase_residual_limit=float(model_config.get("phase_residual_limit", 3.141592653589793)),
            dropout=float(model_config.get("dropout", 0.1)),
        )

    def _beat_map(
        self,
        ecg_beat: Tensor | None,
        batch_size: int,
        size: tuple[int, int],
        device: torch.device,
    ) -> Tensor:
        if ecg_beat is None:
            return torch.zeros(batch_size, 1, size[0], size[1], device=device)
        if ecg_beat.ndim == 2:
            beat = ecg_beat[:, None, None, :]
        elif ecg_beat.ndim == 3:
            beat = ecg_beat[:, None, :, :]
        elif ecg_beat.ndim == 4:
            beat = ecg_beat
        else:
            raise ValueError(f"Unsupported ecg_beat shape: {ecg_beat.shape}")
        return F.interpolate(beat.float(), size=size, mode="bilinear", align_corners=False)

    def _make_input(self, pcg_stft: Tensor, ecg_beat: Tensor | None) -> tuple[Tensor, Tensor, Tensor]:
        if pcg_stft.ndim == 3:
            x = pcg_stft.unsqueeze(1)
        elif pcg_stft.ndim == 4:
            x = pcg_stft
        else:
            raise ValueError(f"pcg_stft must be [B,F,T] or [B,1,F,T], got {pcg_stft.shape}")

        if self.use_complex_mask:
            if self.complex_input_adapter is None:
                raise RuntimeError("Complex input adapter is not initialized.")
            magnitude = x[:, 0]
            x = self.complex_input_adapter(x)
        else:
            if x.shape[1] != 1:
                raise ValueError(f"Magnitude input must have one channel, got {x.shape}")
            magnitude = x[:, 0]

        beat = self._beat_map(ecg_beat, x.shape[0], x.shape[-2:], x.device)
        if self.use_ecg:
            x = torch.cat([x, beat], dim=1)
        return x, beat, magnitude

    def forward(
        self,
        pcg_stft: Tensor,
        ecg_beat: Tensor | None = None,
        imu_feat: Tensor | None = None,
        modality_mask: dict[str, int | Tensor] | None = None,
        return_aux_context: bool = False,
    ) -> dict[str, Tensor]:
        x, beat, magnitude = self._make_input(pcg_stft, ecg_beat)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        b = self.transformer(b)
        b = self.tf_conformer(b)
        b = self.temporal_bilstm(b)
        if self.use_ecg_cross_attention:
            if not isinstance(self.ecg_cross_attention, ECGCrossAttention):
                raise RuntimeError("ECG cross-attention module is not initialized.")
            b = self.ecg_cross_attention(b, beat)

        if self.use_ecg_film:
            if self.ecg_film_bottleneck is None or self.ecg_film_dec3 is None:
                raise RuntimeError("ECG FiLM modules are not initialized.")
            b = self.ecg_film_bottleneck(b, beat)

        if self.use_ecg_attention:
            if self.ecg_gate1 is None or self.ecg_gate2 is None or self.ecg_gate3 is None:
                raise RuntimeError("ECG attention gates are not initialized.")
            e1 = self.ecg_gate1(e1, beat)
            e2 = self.ecg_gate2(e2, beat)
            e3 = self.ecg_gate3(e3, beat)

        d3 = resize_to(b, e3)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        if self.use_ecg_film:
            if self.ecg_film_dec3 is None:
                raise RuntimeError("ECG FiLM decoder module is not initialized.")
            d3 = self.ecg_film_dec3(d3, beat)
        d2 = resize_to(d3, e2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = resize_to(d2, e1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        base_mask_logits = self.mask_head[0](d1).squeeze(1)
        base_mask = torch.sigmoid(base_mask_logits)
        base_s1s2_prob = self.s1s2_head(d1).mean(dim=2)
        base_sqi_score = self.sqi_head(d1).squeeze(1)
        imu_mask_delta = torch.zeros_like(base_mask_logits)
        imu_artifact_gate = torch.zeros(
            base_mask_logits.shape[0],
            base_mask_logits.shape[-1],
            device=base_mask_logits.device,
            dtype=base_mask_logits.dtype,
        )
        if self.use_imu:
            if self.imu_adapter is None:
                raise RuntimeError("IMU adapter is not initialized.")
            imu_mask_delta, imu_artifact_gate = self.imu_adapter.forward_with_gate(
                d1,
                imu_feat,
                modality_mask,
                beat,
            )
            mask_logits = base_mask_logits + imu_mask_delta
            mask = torch.sigmoid(mask_logits)
        else:
            mask = base_mask
        if self.phase_head is not None:
            phase_residual = self.phase_residual_limit * torch.tanh(self.phase_head(d1).squeeze(1))
        else:
            phase_residual = torch.zeros_like(mask)
        complex_mask_real = mask * torch.cos(phase_residual)
        complex_mask_imag = mask * torch.sin(phase_residual)
        denoised_mag = mask * magnitude
        s1s2_prob = base_s1s2_prob
        sqi_score = base_sqi_score
        imu_aux_output: dict[str, Tensor] | None = None
        if self.use_imu_aux:
            if self.imu_aux_adapter is None:
                raise RuntimeError("IMU auxiliary adapter is not initialized.")
            imu_aux_output = self.imu_aux_adapter(
                d1,
                imu_feat,
                modality_mask,
                beat,
            )
            sqi_delta = imu_aux_output["imu_sqi_logit_delta"]
            base_sqi_float = base_sqi_score.float()
            base_sqi_logit = torch.logit(base_sqi_float.clamp(1e-6, 1.0 - 1e-6))
            base_sqi_roundtrip = torch.sigmoid(base_sqi_logit)
            corrected_sqi = base_sqi_float + (
                torch.sigmoid(base_sqi_logit + sqi_delta.float())
                - base_sqi_roundtrip
            )
            sqi_score = corrected_sqi.clamp(0.0, 1.0).to(base_sqi_score.dtype)
        result = DenoisingOutput(
            mask,
            denoised_mag,
            s1s2_prob,
            sqi_score,
            phase_residual,
            complex_mask_real,
            complex_mask_imag,
        ).as_dict()
        if self.use_imu:
            result["imu_mask_delta"] = imu_mask_delta
            result["imu_artifact_gate"] = imu_artifact_gate
            result["base_mask"] = base_mask
        if imu_aux_output is not None:
            result["base_sqi_score"] = base_sqi_score
            result["base_s1s2_prob"] = base_s1s2_prob
            if not self.use_imu:
                result["base_mask"] = base_mask
                result["imu_mask_delta"] = torch.zeros_like(base_mask_logits)
            result.update(imu_aux_output)
            if return_aux_context:
                # Training/evaluation counterfactuals reuse the frozen M7
                # decoder once and only re-run the small IMU auxiliary branch.
                # These private keys are never part of the public output
                # contract or a checkpoint.
                result["_imu_aux_decoder_features"] = d1
                result["_imu_aux_beat_map"] = beat
        return result
