# M14.3-v2 result reproduction

This repository is the minimal fixed-checkpoint release for the MSc thesis model M14.3-v2: an ECG-assisted PCG denoising system with IMU-assisted signal-quality and reliability outputs. It contains the model/evaluation source, frozen configurations, one M7-v2 base checkpoint, four fold-specific IMU adapters and machine-checkable thesis reference values. It contains no research dataset.

The immutable release identifier is `m143-v2-repro-v1.0.0`; use that tag rather than a later moving `main` branch when auditing the thesis result.

The supported claim is **fixed-checkpoint numerical reproduction**. This repository does not attempt to retrain the complete M0--M14.3 development history. Retraining can produce different epochs and weights even with the same protocol, whereas the workflow below evaluates the archived model state and checks 68 reported quantities.

## What is included

```text
checkpoints/                 M7-v2 base plus four M14.3-v2 IMU adapters
configs/                     six frozen evaluation/data configurations
src/ecg_pcg_denoise/         model, dataset, evaluation and audit code
repro/                       data contract and 68 golden-result checks
scripts/                     Windows/Linux one-command wrappers
tests/                       no-data architecture and checkpoint tests
data/README.md               controlled-data placement instructions
environment/                 recorded thesis computing environment
```

The four M14.3-v2 files contain only the fold-specific IMU auxiliary adapter. The identical M7 state is stored once. The loader verifies the fold and base-checkpoint binding before composing the model. The five public weight files total 3,718,047 bytes (3.55 MiB) without changing any tensor or model output.

## Prerequisites

- An authorised copy of the **processed frozen evaluation package**, not merely the raw datasets. See [data/README.md](data/README.md).
- Python 3.10 or later. The paper run used Python 3.13.12 and PyTorch 2.11.0+cu128.
- A CUDA-capable GPU is recommended. CPU evaluation is supported but substantially slower.

For the closest match to the paper environment:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-paper-cu128.txt
python -m pip install -e .
```

The exact recorded software and hardware versions are in [paper_environment.json](environment/paper_environment.json). A different supported PyTorch/CUDA build may be used, but the published tolerances are not a promise of identical behaviour on every platform.

## Place the controlled data

After access has been approved, copy the supplied directory to `data/` without renaming its internal folders:

```text
denoising/
  data/
    m143_v2_dataset_manifest.json
    windows/...
    manifests/...
```

The committed public contract binds the supplied manifest to SHA-256:

```text
8c26c8cbe138934edf0a7dd800987fbfccbc677b7ef85451701631a7289450eb
```

The runner then verifies all 23,379 required files and every frozen clean/IMU pairing before inference. A dataset at another location can be passed with `-DataRoot` or `--data-root`, but the supplied root directory itself must remain named `data` because the frozen CSV paths begin with `data/windows/` (for example, `D:\approved_package\data`).

## Reproduce the paper results

Windows PowerShell:

```powershell
.\scripts\reproduce_m143_v2.ps1
```

Linux/macOS shell:

```bash
bash scripts/reproduce_m143_v2.sh
```

The wrapper records the runtime environment, runs 29 no-data tests, validates checkpoints and controlled data, and executes the 22-step protocol:

1. M7-v2 strict ESC-50 evaluation.
2. Four M14.3-v2 LOSO folds, each with validation-only calibration, ordinary testing and three additional IMU time-shift tests.
3. Leakage-audited strict report generation.
4. Canonical result extraction and 68 golden-value checks.

On the thesis computer (RTX 5090 Laptop GPU), the complete checked run took approximately 13 minutes. Outputs are written to a new timestamped directory under `reproduction_runs/`. The decisive files are:

```text
reproduction_status.json       overall pass/fail and terminal stage
checkpoint_integrity_report.json
dataset_integrity_report.json
canonical_actual_results.json  newly calculated canonical metrics
golden_comparison_report.json  all 68 comparisons and tolerances
command_provenance.json        exact executed commands
logs/                          stdout/stderr for every evaluation command
```

A no-data plan check is available before the controlled package is obtained:

```powershell
.\scripts\reproduce_m143_v2.ps1 -DryRun
```

## Expected headline results

| Evaluation population | Quantity | Reference |
|---|---:|---:|
| Strict ESC-50 M7-v2, 16,230 windows | ΔSNR | 13.7633 dB |
|  | ΔSI-SDR | 12.8533 dB |
|  | Correlation | 0.945903 |
|  | Log-spectral distance | 0.013302 |
|  | SQI MAE | 0.234122 |
|  | S1/S2 MAE | 0.134135 |
| Synthetic-IMU LOSO, 3,667 noisy fold-windows | M14.3-v2 SQI MAE | 0.244726 |
|  | SQI MAE improvement over M7 | 0.001003 |
|  | Artifact AUROC | 0.851594 |
|  | Artifact AUPRC | 0.939972 |
|  | Maximum protected-output difference from M7 | 0.0 |

The two populations in this table are different and must not be compared as if they were a paired benchmark. Within the M14.3-v2 LOSO evaluation, the denoising waveform, mask and S1/S2 locations are intentionally identical to M7. The evidence supports IMU-assisted SQI/reliability assessment, **not** an IMU-driven improvement in denoising.

Numerical acceptance uses the explicit per-field rules in [expected_results.json](repro/expected_results.json), generally `1e-4` absolute plus `1e-4` relative tolerance; identifiers and counts are exact, and protected outputs use `1e-7` absolute tolerance.

## Scope and responsible use

This is research software for academic reproduction and offline method audit. It is not a medical device and must not be used for diagnosis, triage, treatment decisions or unsupervised clinical monitoring. Checkpoints contain model tensors and provenance only; they contain no raw participant signals. Access to data remains governed by the original approvals, agreements and deletion/redistribution conditions. See [MODEL_CARD.md](MODEL_CARD.md) for model limitations.
