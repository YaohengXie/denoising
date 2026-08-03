# Controlled processed-data package

Research data are deliberately excluded from Git. An examiner who has received the required approval should be given the **processed frozen evaluation package** together with `m143_v2_dataset_manifest.json`. Raw BSSLAB, ESC-50 or MotemaSens files alone are insufficient for exact paper-result reproduction because the experiment uses frozen windows, strict source-disjoint mixtures and fixed cross-modal pair manifests.

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

The package contains 23,379 manifested files (2,660,960,256 bytes in the frozen release). Its manifest SHA-256 must be:

```text
8c26c8cbe138934edf0a7dd800987fbfccbc677b7ef85451701631a7289450eb
```

Verify it without running inference:

```powershell
python -m ecg_pcg_denoise.repro verify --data-root data
```

The verifier checks the public manifest binding, exact file sets, counts, sizes and SHA-256 values, then confirms that all 8,640 `clean_path`/`imu_path` pair rows are relative, stay inside the controlled package and name manifested files. Extra matching files are rejected rather than silently entering evaluation.

`scripts/create_dataset_manifest.py` is retained only for the data custodian preparing an authorised copy. A recipient should not regenerate the manifest to make a changed package appear valid: the regenerated file will not match the public manifest hash.

## Data split and role

- BSSLAB participant IDs 1--8 are the training split, 9--10 the validation split and 11--12 the held-out test split used by the frozen protocol.
- Strict ESC-50 mixtures provide full-reference denoising evaluation with source-group leakage audits.
- The four MotemaSens participants provide real IMU time trajectories for synthetic multimodal pairing. Their real PCG recordings are not treated as same-cycle clean targets.
- M14.3-v2 uses validation participant 10 to fit each fold's artifact threshold and confidence calibration before the held-out test evaluation.

## Governance

Share this package only after confirming the recipient, approved purpose, access period and deletion/return conditions. Do not publish it, upload it to public services, attempt re-identification or redistribute it beyond the governing approvals and data-owner terms. The public code and checkpoints do not grant any right to the underlying data.
