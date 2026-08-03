# Controlled processed-data packages

Research data are deliberately excluded from Git. An examiner who has received
the required approval should receive processed, frozen inputs rather than a copy
of the original local project tree. Raw BSSLAB, ESC-50 or MotemaSens files alone
are insufficient because the experiment uses fixed windows, source-disjoint
mixtures, manual alignment and frozen cross-modal pair manifests.

The `Fixed` and `Adapter` workflows use the evaluation package described below.
`Full` composes that unchanged package with a separate training extension.

Place the authorised package at `data/` in the repository, or pass its path explicitly. The directory must retain this structure:

```text
data/
  m143_v2_dataset_manifest.json
  windows/
    bsslab_musan/clean/
      train/                         4,204 NPZ windows
      val/                           1,078 NPZ windows
      test/                          1,082 NPZ windows
    bsslab_esc50_v2/mixed/test/     16,230 NPZ windows
    motema_external_m7/windows/
      M001_S01/                        171 NPZ windows
      M001_S02/                        189 NPZ windows
      M001_S03/                        205 NPZ windows
      M001_S04/                        181 NPZ windows
  manifests/
    bsslab_m14_imu/                    37 frozen CSV/JSON files
    esc50_strict_audit.json
    bsslab_esc50_v2_mixed_audit.json
```

The root directory must itself be named `data`, including when it is stored outside the repository (for example, `D:\approved_package\data`). This requirement keeps the relative paths embedded in the frozen pair CSV files unambiguous.

The evaluation package contains 23,379 manifested files (2,660,960,256 bytes
in the frozen release). Its manifest SHA-256 must be:

```text
1ef353fe40b6189ca38196f1cc07eea9bbfd360c3fa7c03203ead7890b6262e5
```

Verify it without running inference:

```powershell
python -m ecg_pcg_denoise.repro verify --data-root data
```

The verifier checks the public manifest binding, exact file sets, counts, sizes and SHA-256 values, then confirms that all 8,640 `clean_path`/`imu_path` pair rows are relative, stay inside the controlled package and name manifested files. Extra matching files are rejected rather than silently entering evaluation.

`scripts/create_dataset_manifest.py` is retained only for the data custodian preparing an authorised copy. A recipient should not regenerate the manifest to make a changed package appear valid: the regenerated file will not match the public manifest hash.

## Full-training extension

The `Full` workflow additionally requires:

```text
data/
  m143_v2_training_dataset_manifest.json
  windows/
    bsslab_esc50_v2/mixed/
      train/                         63,060 NPZ windows
      val/                           16,170 NPZ windows
    bsslab_esc50_enhanced_v2/mixed/
      train/                         21,020 NPZ windows
      val/                            5,390 NPZ windows
```

The additive extension contains 105,640 files and 13,744,115,616 bytes. The
private per-file manifest must match the public binding in
`repro/training_dataset_contract.json`:

```text
0d2a0ef45c8c022f12c3f8303e233310a2ecab82f4df186592f56f2bafcde4bf
```

Verify the extension without training:

```powershell
python scripts/create_training_dataset_manifest.py --data-root data --verify
```

The 5,410-window Enhanced-v2 historical test directory is not read by the
minimum `Full` training/validation protocol and is deliberately excluded from
the training-extension manifest. Together, the evaluation package and minimum
training extension contain 129,019 sealed files.

## Data split and role

- BSSLAB participant IDs 1--8 are the training split, 9--10 the validation split and 11--12 the held-out test split used by the frozen protocol.
- Strict ESC-50 mixtures provide full-reference denoising evaluation with source-group leakage audits.
- The four MotemaSens participants provide real IMU time trajectories for synthetic multimodal pairing. Their real PCG recordings are not treated as same-cycle clean targets.
- During M14.3-v2 adapter training, participant 9 is used for early stopping
  and safe checkpoint selection; participant 10 is reserved for artifact
  threshold and confidence calibration; participants 11--12 are final test only.
- Each MotemaSens LOSO fold excludes its held-out participant from adapter
  training and IMU normalisation statistics.

## Governance

Share either package only after confirming the recipient, approved purpose,
access period and deletion/return conditions. Do not publish it, upload it to
public services, attempt re-identification or redistribute it beyond the
governing approvals and data-owner terms. The public code and checkpoints do
not grant any right to the underlying data. Before delivery, regenerate any
audit summary that embeds a local absolute path so the controlled copy contains
only portable relative paths.
