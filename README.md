# M14.3-v2 reproduction and retraining

This repository contains the model, training, validation, evaluation and audit
source for the MSc thesis system M14.3-v2: ECG-assisted PCG denoising with
IMU-assisted signal-quality and reliability assessment. It contains the
published model checkpoints but **no research dataset**.

The immutable release identifier for this expanded source release is
`m143-v2-repro-v2.0.2`. Use that tag, rather than a later moving `main` branch,
when auditing the thesis workflow.

## Three scientifically distinct modes

`Fixed`, `Adapter` and `Full` are three workflows around the same final model
architecture; they are not three competing model architectures.

| Mode | Starting point | What is trained | Required controlled data | Reproduction claim |
|---|---|---|---|---|
| **Fixed** | Published M7-v2 and four published M14.3-v2 adapters | Nothing | 23,379-file evaluation package | Re-runs inference and checks 68 thesis quantities within declared tolerances |
| **Adapter** | Published M7-v2 | Four randomly initialised fold-specific IMU auxiliary adapters; M7 stays frozen | Same 23,379-file evaluation package | Repeats adapter training, safe selection, validation-only calibration and held-out testing |
| **Full** | M5 and M6 start from new random weights | Complete five-stage M7-v2 chain, then four M14.3-v2 adapters | Evaluation package plus 105,640-file training extension, 129,019 files in total | Repeats the complete declared training, validation, selection, calibration and test protocol |

Only `Fixed` promises numerical agreement with the archived thesis values. The
original GPU training used AMP and did not force bitwise-deterministic CUDA
execution, so `Adapter` and `Full` are protocol-level reproductions: checkpoint
hashes, selected epochs and final decimal values may differ even with the same
seed and data.

## Model boundary

M7-v2 uses a U-Net encoder/decoder and Transformer bottleneck to process noisy
PCG together with an ECG-derived timing map. It produces the enhancement mask,
reconstructed PCG quantities, S1/S2 timing proxies and a base SQI estimate.

M14.3-v2 adds a small `IMUAuxiliaryAdapter`. It receives six IMU features plus
frozen M7 decoder/ECG context and produces motion, coupled-artifact, reliability
and confidence outputs together with a bounded SQI correction. It cannot modify
the denoising mask, reconstructed waveform or S1/S2 locations. During adapter
training every M7 parameter is frozen and only `imu_aux_adapter.*` is updated.

The public weights are factorised: the complete common M7 state is stored once,
and each of the four LOSO folds stores only its 30-tensor IMU adapter. The loader
checks the fold and base-checkpoint binding before composing the complete model.

## Repository contents

```text
checkpoints/                 published M7-v2 base and four M14.3-v2 adapters
configs/                     fixed evaluation, five-stage M7 and four-fold M14.3 configs
src/ecg_pcg_denoise/
  models/                    complete M7/M14.3 model definition
  data/                      online noise/IMU synthesis used by processed-data training
  train/                     M7 and M14.3 training, selection, validation and reporting
  repro/                     Fixed workflow and 68-value golden comparison
  retrain/                   Adapter/Full plan builder and execution runner
repro/                       public evaluation and training-extension data contracts
scripts/                     Windows/Linux wrappers, environment capture and manifest tools
tests/                       data-free architecture, training, integrity and runner tests
docs/RETRAINING.md           detailed data and retraining boundary
data/README.md               controlled-data placement and governance instructions
environment/                 recorded thesis hardware/software environment
```

Historical M0--M14.2 experiments, unrelated baselines and raw-data notebooks are
not required by these three final workflows and are deliberately excluded.

## Installation

Python 3.10 or later is required. The reported run used Python 3.13.12,
PyTorch 2.11.0+cu128, CUDA 12.8 and an RTX 5090 Laptop GPU.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-paper-cu128.txt
python -m pip install -e ".[dev]"
python -m pytest -q
```

The exact recorded environment is in
[`environment/paper_environment.json`](environment/paper_environment.json).
A compatible CPU build can execute the code, but training and complete
evaluation will be substantially slower.

## Controlled processed data

After approval, place the supplied root at `data/` or pass its absolute path.
The root directory itself must remain named `data`, because the frozen pair CSV
files use relative paths beginning with `data/windows/`.

### Fixed and Adapter package

```text
data/
  m143_v2_dataset_manifest.json
  windows/
    bsslab_musan/clean/{train,val,test}/
    bsslab_esc50_v2/mixed/test/
    motema_external_m7/windows/M001_S01...M001_S04/
  manifests/
    bsslab_m14_imu/
    esc50_strict_audit.json
    bsslab_esc50_v2_mixed_audit.json
```

This package contains 23,379 manifested files and is bound by
[`repro/dataset_contract.json`](repro/dataset_contract.json).

### Additional Full-training extension

```text
data/
  m143_v2_training_dataset_manifest.json
  windows/
    bsslab_esc50_v2/mixed/{train,val}/
    bsslab_esc50_enhanced_v2/mixed/{train,val}/
```

The extension contains 105,640 files (13,744,115,616 bytes). Its private
per-file manifest is bound by
[`repro/training_dataset_contract.json`](repro/training_dataset_contract.json)
with SHA-256:

```text
0d2a0ef45c8c022f12c3f8303e233310a2ecab82f4df186592f56f2bafcde4bf
```

Raw BSSLAB, ESC-50 and MotemaSens files are not required for these workflows.
The supplied inputs are already filtered, windowed, split, leakage-audited and,
where applicable, manually aligned. See [`data/README.md`](data/README.md) and
[`docs/RETRAINING.md`](docs/RETRAINING.md) for the exact contract.

## Fixed: reproduce the archived thesis results

Windows:

```powershell
.\scripts\reproduce_m143_v2.ps1 -DataRoot "D:\approved_package\data"
```

Linux/macOS:

```bash
bash scripts/reproduce_m143_v2.sh --data-root /approved/package/data
```

This executes 22 scientific commands: strict M7 evaluation, four M14.3-v2 LOSO
folds with validation-only calibration and temporal-shift tests, strict report
generation, canonical extraction and 68 machine-readable golden checks. A
no-data plan check is available with `-DryRun` or `--dry-run`.

## Adapter: retrain only the four IMU auxiliary adapters

Windows:

```powershell
.\scripts\retrain_m143_v2.ps1 `
  -Scope Adapter `
  -DataRoot "D:\approved_package\data"
```

Linux/macOS:

```bash
bash scripts/retrain_m143_v2.sh \
  --scope adapter \
  --data-root /approved/package/data
```

The runner verifies the published weights and evaluation package, evaluates the
published M7 base, then performs 30 ordered scientific commands. For each fold
it trains a new adapter, applies the hard safe-checkpoint gates, calibrates only
on validation participant 10, tests on held-out BSSLAB participants 11--12 and
runs the required IMU time-shift tests.

## Full: retrain M7-v2 and M14.3-v2 from new weights

Windows:

```powershell
.\scripts\retrain_m143_v2.ps1 `
  -Scope Full `
  -DataRoot "D:\approved_package\data"
```

Linux/macOS:

```bash
bash scripts/retrain_m143_v2.sh \
  --scope full \
  --data-root /approved/package/data
```

The 35-command Full plan is:

1. train M5 from a new random initialisation;
2. train M6 multitask independently from a new random initialisation;
3. initialise M7 distillation from the new M6 and use the new M5 as teacher;
4. robustly fine-tune M7 on Enhanced-v2 mixtures;
5. fine-tune only the SQI head to obtain the new M7-v2 base;
6. evaluate that new base on the strict held-out test set;
7. train, safely select, calibrate and test four M14.3-v2 LOSO adapters;
8. run the required temporal-shift tests and generate the strict report.

Before training, Full verifies both controlled manifests and hashes every one of
the 129,019 sealed inputs. The newly trained M7 is then bound into every fold;
the published M7 checkpoint is not used as a training initialisation.

Both retraining scopes support a no-data plan check:

```powershell
.\scripts\retrain_m143_v2.ps1 -Scope Adapter -DryRun
.\scripts\retrain_m143_v2.ps1 -Scope Full -DryRun
```

Every run writes to a new timestamped directory under `retraining_runs/` and
never overwrites the published checkpoints. It records runtime configurations,
environment details, command logs, training histories, selected checkpoints,
calibration files, test results, the strict report and a descriptive comparison
with the archived thesis values.

The generated examiner-facing report is written in British academic English to
`reports/strict_esc50_m143_v2/report.md`.

## After a run: find, visualise and interpret the results

The wrapper prints the absolute run directory in its final status line. Results
are never written into the public checkpoint directories. The examples below
assume the explicit run names used in this README; substitute the directory
printed by the wrapper if a timestamped default was used.

### Fixed result acceptance

For a run at `reproduction_runs/examiner_fixed`, the decisive files are:

| Artefact | Purpose |
|---|---|
| `reproduction_status.json` | Overall terminal state and the stage reached |
| `golden_comparison_report.json` | All 68 comparisons with the archived thesis values |
| `canonical_actual_results.json` | Metrics recalculated by the current run |
| `reports/strict_esc50_m143_v2/report.md` | English narrative report with embedded figures |
| `reports/strict_esc50_m143_v2/figures/` | Four publication-ready result visualisations |
| `reports/strict_esc50_m143_v2/data/` | Overall, by-SNR, fold, identity and timing-shift CSV/JSON tables |
| `command_provenance.json` and `logs/` | Exact commands and per-command stdout/stderr |

On Windows, inspect the machine-readable acceptance state with:

```powershell
$Run = "reproduction_runs\examiner_fixed"
$Status = Get-Content "$Run\reproduction_status.json" -Raw | ConvertFrom-Json
$Golden = Get-Content "$Run\golden_comparison_report.json" -Raw | ConvertFrom-Json
$Status | Select-Object status, stage, exit_code
$Golden.details | Select-Object checks_total, checks_passed, checks_failed
```

The required Fixed outcome is `status=pass`,
`stage=golden_result_comparison`, `exit_code=0`, and 68/68 checks passed.
Open the English report and its visualisations with:

```powershell
Invoke-Item "$Run\reports\strict_esc50_m143_v2\report.md"
Invoke-Item "$Run\reports\strict_esc50_m143_v2\figures"
```

The four figures show strict M7 denoising overall/by SNR, protected-output
identity, fold-wise SQI comparison, and IMU alignment/counterfactual behaviour.

### Adapter and Full result acceptance

For `retraining_runs/examiner_adapter` or `retraining_runs/examiner_full`, set
`$Run` to the relevant directory and inspect:

| Artefact | Purpose |
|---|---|
| `retraining_status.json` | Protocol-level terminal state |
| `canonical_retrained_results.json` | Metrics calculated from the newly trained checkpoints |
| `reference_difference_report.json` | Descriptive comparison with the archived Fixed values |
| `reports/strict_esc50_m143_v2/report.md` | English retraining report with the same four visualisations |
| `outputs/` | Training histories, selected checkpoints, calibration and test outputs |
| `command_provenance.json` and `logs/` | Complete executable audit trail |

```powershell
$Run = "retraining_runs\examiner_adapter"  # or examiner_full
$Status = Get-Content "$Run\retraining_status.json" -Raw | ConvertFrom-Json
$Status | Select-Object status, stage, exit_code
Invoke-Item "$Run\reports\strict_esc50_m143_v2\report.md"
Invoke-Item "$Run\reports\strict_esc50_m143_v2\figures"
```

The required protocol outcome is `status=pass`, `stage=protocol_complete` and
`exit_code=0`. `reference_difference_report.json` deliberately has top-level
`status=not_enforced`: stochastic Adapter/Full retraining is not required to
reproduce the archived checkpoint hash, selected epoch or every reported
decimal. This is not a failed run. If `report.md` is absent, the protocol did
not reach report generation; read `retraining_status.json` and the latest file
under `logs/` before attempting another run directory.

## Fixed reference values

| Evaluation population | Quantity | Archived value |
|---|---:|---:|
| Strict ESC-50 M7-v2, 16,230 windows | Delta SNR | 13.7633 dB |
|  | Delta SI-SDR | 12.8533 dB |
|  | Correlation | 0.945903 |
|  | Log-spectral distance | 0.013302 |
|  | SQI MAE | 0.234122 |
|  | S1/S2 MAE | 0.134135 |
| Synthetic-IMU LOSO, 3,667 noisy fold windows | M14.3-v2 SQI MAE | 0.244726 |
|  | SQI MAE improvement over M7 | 0.001003 |
|  | Artifact AUROC | 0.851594 |
|  | Artifact AUPRC | 0.939972 |
|  | Maximum protected-output difference from M7 | 0.0 |

The strict M7 population and synthetic-IMU LOSO population are different and
must not be treated as a paired comparison. The evidence supports IMU-assisted
quality/reliability assessment, not an IMU-driven improvement in denoising.

## Responsible use

This is academic research software, not a medical device. It must not be used
for diagnosis, triage, treatment decisions or unsupervised clinical monitoring.
The public checkpoints contain model tensors and provenance only; they contain
no participant waveforms. Data access remains governed by the original ethics,
participant-information, data-owner, access-expiry and deletion conditions.
See [`MODEL_CARD.md`](MODEL_CARD.md) for limitations.
