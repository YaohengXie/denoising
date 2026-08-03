# Model card: M14.3-v2

Release: `m143-v2-repro-v2.0.1` (package version 2.0.1).

## Model summary

M14.3-v2 is the final research system evaluated in the MSc project. Its frozen M7-v2 backbone accepts the noisy PCG time-frequency representation together with an ECG-derived beat-timing map. A U-Net encoder/decoder provides local multiscale processing and a Transformer bottleneck supplies longer-range context. The backbone outputs the enhancement mask and waveform reconstruction quantities, S1/S2 probability proxies and a base SQI estimate.

A fold-specific IMU auxiliary adapter receives six-channel motion features. It estimates motion/artifact probability, modality reliability and task confidence, and applies a bounded correction to SQI. By construction it cannot change the enhancement mask, reconstructed waveform or S1/S2 locations. This separation is the reason the model can use IMU for quality awareness without degrading the frozen M7 denoiser.

## Public artifacts

- `checkpoints/m7_v2/best.pt`: complete M7-v2 model state and minimal model configuration.
- `checkpoints/m143_v2/M001_S0X/best_safe.pt`: one IMU auxiliary adapter for each LOSO fold.
- `checkpoints/checksums.json`: weight and selection-summary SHA-256 values, original archive SHA-256 and canonical tensor-state hashes.

The public artifacts were exported with optimizer state, machine-specific paths and duplicate M7 tensors removed. Tests reconstruct every full fold and verify the canonical composed state plus exact preservation of protected model outputs. All files are loaded with PyTorch's restricted `weights_only=True` path.

## Evaluation evidence

The release was validated by rebuilding all evaluation outputs from the public factorised checkpoints and the frozen controlled data package. The resulting run completed all 22 commands and passed 68/68 declared checks.

The strict ESC-50 M7-v2 evaluation contained 16,230 held-out mixtures and obtained ΔSNR 13.7633 dB, ΔSI-SDR 12.8533 dB, correlation 0.945903 and log-spectral distance 0.013302. In the separate four-fold synthetic-IMU LOSO evaluation, M14.3-v2 achieved a fold-macro SQI MAE of 0.244726 versus 0.245729 for M7, artifact AUROC 0.851594 and artifact AUPRC 0.939972. The maximum difference from M7 for protected denoising/S1-S2 outputs was 0.0.

These results do not show that IMU improves denoising. They show a small SQI improvement and useful artifact discrimination while preserving the M7 signal-processing outputs.

## Intended use

- Academic result reproduction.
- Protocol-level retraining from the separately authorised processed inputs.
- Offline algorithm audit and ablation analysis.
- Research on ECG-assisted PCG enhancement and IMU-assisted quality assessment.

## Out-of-scope use

- Clinical diagnosis, triage or treatment decisions.
- Real-time autonomous patient monitoring.
- Claims of clinical validity or population-wide generalisation.
- Interpreting the real MotemaSens PCG as having paired same-cycle clean targets.

## Limitations

- The main denoising evidence is full-reference evaluation on synthetically corrupted PCG rather than paired real noisy/clean clinical recordings.
- The MotemaSens contribution is limited to four participants and is used primarily as an IMU trajectory source and feasibility study.
- Generalisation across sensors, acquisition sites, pathologies and demographic groups has not been clinically established.
- ECG or IMU absence, timing error and device-domain shift can affect auxiliary outputs; temporal-shift tests characterise only selected perturbations.
- Fixed-checkpoint evaluation can be numerically checked within declared tolerances. From-scratch retraining is not expected to reproduce identical checkpoint bytes, selected epochs or every decimal.
- The Adapter and Full runners reproduce declared training protocols, not a bitwise-deterministic training process; only the Fixed workflow is governed by the 68-value numerical acceptance contract.

## Ethical and data considerations

No raw participant waveform is embedded in the public checkpoints. Nevertheless, use of the separately supplied processed dataset remains subject to the original ethics, participant-information, data-owner and access-control conditions. Users must not attempt re-identification or further distribution.
