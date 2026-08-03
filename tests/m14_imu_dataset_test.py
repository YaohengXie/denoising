from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from ecg_pcg_denoise.data.m14_imu_synthesis import (
    build_imu_conditioned_noisy,
    extract_imu_frame_features,
    fit_imu_feature_stats,
    motion_envelope_from_features,
    normalize_imu_features,
)
from ecg_pcg_denoise.train.m14_imu_dataset import M14IMUDataset


def _signals(fs: int = 2000, seconds: float = 4.0) -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(int(fs * seconds)) / fs
    clean = 0.3 * np.sin(2 * np.pi * 45 * t) + 0.1 * np.sin(2 * np.pi * 90 * t)
    imu = np.column_stack(
        (
            0.03 * np.sin(2 * np.pi * 1.4 * t),
            0.02 * np.sin(2 * np.pi * 2.1 * t),
            1.0 + 0.04 * np.sin(2 * np.pi * 1.8 * t),
        )
    )
    signal_burst = np.hanning(400) * 0.5
    imu[3000:3400, 0] += signal_burst
    return clean.astype(np.float32), imu.astype(np.float32)


def test_imu_features_and_synthesis_have_m7_shapes_and_target_snr() -> None:
    clean, imu = _signals()
    features, valid = extract_imu_frame_features(imu, 2000, 32, 256)
    assert features.shape == (6, 251)
    assert valid.shape == (251,)
    stats = fit_imu_feature_stats([features, features * 1.2])
    normalized = normalize_imu_features(features, stats)
    envelope = motion_envelope_from_features(normalized, stats, valid)
    result = build_imu_conditioned_noisy(
        clean,
        normalized,
        envelope,
        fs=2000,
        hop_length=32,
        snr_db=0.0,
        mode="motion_artifact",
        rng=np.random.default_rng(7),
    )
    assert result.noisy_pcg.shape == (8000,)
    assert result.noise_signal.shape == (8000,)
    assert abs(result.achieved_snr_db) < 0.05
    assert np.all(np.isfinite(result.noisy_pcg))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_virtual_dataset_is_deterministic_and_returns_imu(tmp_path: Path) -> None:
    clean, imu = _signals()
    clean_path = tmp_path / "clean.npz"
    imu_path = tmp_path / "imu.npz"
    np.savez_compressed(
        clean_path,
        clean_pcg=clean,
        ecg=np.zeros_like(clean),
        ecg_beat=np.zeros_like(clean),
        s1s2_weak=np.zeros((2, len(clean)), dtype=np.float32),
        fs=2000,
        subject_id=1,
        record_id="record",
        window_start=0,
    )
    np.savez_compressed(
        imu_path,
        imu=imu,
        fs=2000,
        participant_id="M001_S02",
        condition="body_movement",
        record_id="imu_record",
        window_start_sec=0.0,
    )
    features, _ = extract_imu_frame_features(imu, 2000, 32, 256)
    stats = fit_imu_feature_stats([features, features * 1.1])
    fold_root = tmp_path / "manifests" / "fold_M001_S01"
    fold_root.mkdir(parents=True)
    (fold_root / "imu_feature_stats.json").write_text(
        json.dumps(stats.to_dict()),
        encoding="utf-8",
    )
    clean_row = {
        "clean_path": str(clean_path),
        "bsslab_subject_id": "1",
        "bsslab_record_id": "record",
        "bsslab_window_start": 0,
    }
    imu_row = {
        "imu_path": str(imu_path),
        "participant_id": "M001_S02",
        "source_collection_id": "M001",
        "condition": "body_movement",
        "record_id": "imu_record",
        "window_start_sec": 0.0,
        "fs": 2000,
        "samples": 8000,
    }
    _write_csv(fold_root / "clean_train.csv", [clean_row])
    _write_csv(fold_root / "imu_pool_train.csv", [imu_row])
    fixed = {
        **clean_row,
        **imu_row,
        "artifact_mode": "motion_artifact",
        "snr_db": 0.0,
        "motion_score": 0.5,
        "seed": 99,
    }
    _write_csv(fold_root / "fixed_val_pairs.csv", [fixed])
    _write_csv(fold_root / "fixed_test_pairs.csv", [fixed])

    dataset = M14IMUDataset(
        manifest_root=tmp_path / "manifests",
        fold="M001_S01",
        split="val",
        project_root=tmp_path,
    )
    first = dataset[0]
    second = dataset[0]
    np.testing.assert_allclose(first["noisy_pcg"].numpy(), second["noisy_pcg"].numpy())
    assert tuple(first["imu_raw"].shape) == (3, 8000)
    assert tuple(first["imu_feat"].shape) == (6, 251)
    assert tuple(first["motion_envelope"].shape) == (251,)
    assert first["imu_subject_id"] == "M001_S02"
