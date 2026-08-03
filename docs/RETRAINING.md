# Retraining modes and controlled processed data

The repository exposes three separate reproducibility claims. They deliberately
do not share one result-acceptance rule.

| Mode | What is trained | Starting weights | Controlled processed data | Expected outcome |
|---|---|---|---|---|
| `Fixed` | Nothing | Published M7-v2 plus four published adapters | Existing 23,379-file evaluation package | Recalculate the 68 published quantities within their declared tolerances |
| `Adapter` | Four fold-specific IMU auxiliary adapters only; M7-v2 stays frozen | Published M7-v2; adapters are randomly initialized | The same 23,379-file evaluation package | Reproduce LOSO training, validation-only selection/calibration and held-out testing; weights and final decimals may differ |
| `Full` | M5, M6, M7 distillation, robust fine-tuning, SQI fine-tuning, then the four adapters | M5/M6 start randomly; later stages consume the preceding selected checkpoints | Evaluation package plus the 105,640-file training extension (129,019 sealed files in total) | Reproduce the complete declared training/validation/test protocol; checkpoint hashes, selected epochs and final decimals may differ |

`Fixed` is the numerical paper-result reproduction. `Adapter` and `Full` are
protocol reproductions: seeded execution narrows incidental variation, but GPU
kernels, library versions, shuffled batches and checkpoint selection can still
change the learned tensors.

## Data boundary

Only processed, frozen inputs are needed. Raw BSSLAB, ESC-50 and MotemaSens
source files are outside this contract and must not be copied into the release.
The existing evaluation package remains governed by
[`repro/dataset_contract.json`](../repro/dataset_contract.json): it includes the
clean PCG/ECG windows, strict test windows, pseudonymised IMU windows, four-fold
LOSO manifests, fixed validation/test pairs, feature statistics and leakage
audits required by `Fixed` and `Adapter`.

`Full` composes that unchanged package with the additive contract in
[`repro/training_dataset_contract.json`](../repro/training_dataset_contract.json):

```text
data/
  m143_v2_dataset_manifest.json                 existing evaluation seal
  m143_v2_training_dataset_manifest.json        additive Full-training seal
  windows/
    bsslab_esc50_v2/mixed/
      train/                                    63,060 processed NPZ windows
      val/                                      16,170 processed NPZ windows
      test/                                     16,230 files in evaluation seal
    bsslab_esc50_enhanced_v2/mixed/
      train/                                    21,020 processed NPZ windows
      val/                                       5,390 processed NPZ windows
      test/                                      optional; excluded from both seals
    bsslab_musan/clean/...                       files in evaluation seal
    motema_external_m7/windows/...               files in evaluation seal
  manifests/...                                 files in evaluation seal
```

The minimum Full extension is therefore 105,640 files. Together with the
unchanged 23,379-file evaluation package, the minimum controlled Full package is
129,019 files. The 5,410-window Enhanced test split is a historical diagnostic
split, is not read during minimum Full training/validation, and is intentionally
excluded from the sealed training manifest even when it happens to be present.

## Privacy-safe manifest workflow

The public training contract records aggregate group names, glob patterns and
counts only. It contains no individual window or participant filename. Exact
verification still requires a private per-file ledger, so the data custodian
creates `data/m143_v2_training_dataset_manifest.json` inside the separately
controlled package. `data/**` is Git-ignored; the generated ledger must never be
committed or published.

From the repository root, the authorised data custodian runs:

```powershell
python scripts/create_training_dataset_manifest.py --data-root data
```

The output is deterministic and reports its SHA-256. Before distributing the
code release, the custodian places that digest in `manifest_sha256` in the public
training contract. A recipient then checks the additive package with:

```powershell
python scripts/create_training_dataset_manifest.py --data-root data --verify
```

Verification rejects an altered public-manifest binding, unsafe or absolute
paths, duplicate files, overlapping patterns, missing or extra group members,
wrong group assignments, byte-size changes and content-hash changes. The
existing `Fixed` verifier separately validates the evaluation manifest and the
frozen clean/IMU pair bindings. A Full run must pass both verifiers; an Adapter
run requires only the existing evaluation verifier.

Regenerating a manifest is a custodian operation, not a way for a recipient to
accept changed data. If regeneration produces a different digest, the data
package has changed and requires a new reviewed contract/version.

## Commands

Run from the repository root. Windows PowerShell examples are shown; the shell
wrapper exposes the same values as `--scope`, `--data-root`, `--run-root`,
`--python`, `--device` and `--dry-run`.

```powershell
# Published M7, newly initialised four-fold auxiliary adapters.
.\scripts\retrain_m143_v2.ps1 `
  -Scope Adapter `
  -DataRoot "D:\approved_package\data"

# Newly initialised M5/M6, complete M7-v2 chain, then new adapters.
.\scripts\retrain_m143_v2.ps1 `
  -Scope Full `
  -DataRoot "D:\approved_package\data"

# Data-free command-plan checks.
.\scripts\retrain_m143_v2.ps1 -Scope Adapter -DryRun
.\scripts\retrain_m143_v2.ps1 -Scope Full -DryRun
```

The Adapter plan contains 30 scientific commands. It evaluates the archived M7
base, then trains, safely selects, calibrates and tests each fold and generates
the strict report. The Full plan prepends five ordered M7 training stages and
therefore contains 35 commands.

## Full-stage checkpoint chain

The dependency chain is materialised into run-specific configurations; no
stage reads an output from an earlier local project directory:

```text
M5 (new random initialisation) ---------------- teacher ----+
                                                          |
M6 multitask (new random initialisation) --- initialises --+--> M7 distillation
                                                                  |
                                                                  v
                                                            M7 robust
                                                                  |
                                                                  v
                                                       M7 robust-SQI head
                                                                  |
                                                                  v
                                              four frozen-base M14.3 adapters
```

The Full runner binds the M7 checkpoint generated in that same run into all
four adapters. Adapter mode instead binds the archived public M7 checkpoint.
Neither mode initialises a new adapter from a published adapter checkpoint.

## Selection, calibration and test boundary

- M14.3 updates only `imu_aux_adapter.*`; the M7 state is frozen.
- Safe selection rejects any epoch whose mask or S1/S2 location output differs
  from M7, or whose validation SQI MAE is inferior at tolerance zero.
- Participant 9 is used for adapter early stopping/selection.
- Participant 10 alone fits the artifact threshold and confidence mapping.
- Participants 11--12 are final paired test only.
- Each MotemaSens fold trains on three participants and holds out the fourth;
  the held-out participant is excluded from IMU normalisation statistics.
- The ordinary test includes the configured 0.512 s shift evaluation; explicit
  0.128, 0.256 and 1.024 s shift runs are also generated.

## Outputs and acceptance

Every invocation creates a fresh run directory and never writes into `data/` or
`checkpoints/`. It records the environment, materialised runtime configurations,
data-integrity reports, exact commands, stdout/stderr logs, training histories,
stage checkpoints, safe-selection summaries, calibration files, test results
and strict report.

The report's structural gates are hard failures: data leakage, missing folds,
non-finite values, altered protected outputs, selection failure or calibration
against the wrong checkpoint stops the run. A descriptive
`reference_difference_report.json` is also written. Its numerical differences
are not a pass/fail rule for stochastic retraining; the 68-value tolerance
contract applies only to the separate Fixed workflow.
