from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from ecg_pcg_denoise.data.m14_imu_synthesis import (
    IMUFeatureStats,
    SYNTHESIS_MODES,
    build_imu_conditioned_noisy,
    choose_motion_conditioned_snr,
    extract_imu_frame_features,
    motion_envelope_from_features,
    motion_score,
    normalize_imu_features,
    stft_frame_count,
)
from ecg_pcg_denoise.utils.config import get_nested, load_config, require_nested


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _resolve_path(value: str, project_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


class M14IMUDataset(Dataset):
    """Virtual BSSLAB dataset with online Motema-IMU-conditioned contamination."""

    def __init__(
        self,
        manifest_root: str | Path,
        fold: str,
        split: str,
        project_root: str | Path = ".",
        seed: int = 1337,
        fs: int = 2000,
        n_fft: int = 256,
        hop_length: int = 32,
        win_length: int = 256,
        snr_values: tuple[float, ...] = (10.0, 5.0, 0.0, -5.0, -10.0),
        mode_probabilities: dict[str, float] | None = None,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split: {split}")
        self.project_root = Path(project_root).resolve()
        self.fold_root = Path(manifest_root) / f"fold_{fold}"
        self.fold_root = (
            self.fold_root
            if self.fold_root.is_absolute()
            else self.project_root / self.fold_root
        )
        if not self.fold_root.exists():
            raise FileNotFoundError(f"M14 fold manifest does not exist: {self.fold_root}")

        self.fold = fold
        self.split = split
        self.seed = int(seed)
        self.epoch = 0
        self.fs = int(fs)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length)
        self.n_frames = stft_frame_count(int(round(4.0 * fs)), hop_length)
        self.snr_values = tuple(float(value) for value in snr_values)
        self.mode_probabilities = mode_probabilities or {
            "motion_artifact": 0.55,
            "combined": 0.20,
            "independent_artifact": 0.10,
            "motion_clean": 0.10,
            "clean_identity": 0.05,
        }
        if set(self.mode_probabilities) - set(SYNTHESIS_MODES):
            unknown_modes = set(self.mode_probabilities) - set(SYNTHESIS_MODES)
            raise ValueError(f"Unknown synthesis modes: {unknown_modes}")
        probabilities = np.asarray(list(self.mode_probabilities.values()), dtype=np.float64)
        if np.any(probabilities < 0) or float(np.sum(probabilities)) <= 0:
            raise ValueError("mode_probabilities must be non-negative and have positive sum.")
        self.mode_names = tuple(self.mode_probabilities)
        self.mode_probs = probabilities / np.sum(probabilities)
        # Windows are immutable. Keeping decoded arrays and deterministic IMU
        # features in-process avoids reopening thousands of compressed NPZ
        # files on every online-synthesis epoch. Noise/SNR/mode sampling still
        # changes with ``set_epoch`` and is never cached.
        self._clean_cache: dict[Path, dict[str, Any]] = {}
        self._imu_cache: dict[Path, dict[str, Any]] = {}

        stats_path = self.fold_root / "imu_feature_stats.json"
        with stats_path.open("r", encoding="utf-8") as handle:
            self.stats = IMUFeatureStats.from_dict(json.load(handle))

        if split == "train":
            self.clean_rows = _read_csv(self.fold_root / "clean_train.csv")
            self.fixed_rows: list[dict[str, str]] | None = None
            self.imu_rows = _read_csv(self.fold_root / "imu_pool_train.csv")
            if not self.imu_rows:
                raise FileNotFoundError(f"No training IMU windows in {self.fold_root}")
            grouped: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(
                lambda: defaultdict(list)
            )
            for row in self.imu_rows:
                grouped[row["participant_id"]][row["condition"]].append(row)
            self.imu_groups = {
                participant: dict(conditions) for participant, conditions in grouped.items()
            }
        else:
            self.fixed_rows = _read_csv(self.fold_root / f"fixed_{split}_pairs.csv")
            self.clean_rows = []
            self.imu_rows = []
            self.imu_groups = {}
            if not self.fixed_rows:
                raise FileNotFoundError(f"No fixed {split} pairs in {self.fold_root}")

    @classmethod
    def from_config(
        cls,
        config_path: str | Path,
        split: str,
        fold: str | None = None,
    ) -> "M14IMUDataset":
        config = load_config(config_path)
        selected_fold = str(fold or require_nested(config, "m14_imu.default_fold"))
        probabilities = get_nested(config, "m14_imu.mode_probabilities", None)
        return cls(
            manifest_root=require_nested(config, "paths.m14_manifest_dir"),
            fold=selected_fold,
            split=split,
            project_root=get_nested(config, "paths.project_root", "."),
            seed=int(get_nested(config, "project.seed", 1337)),
            fs=int(require_nested(config, "data.fs_model")),
            n_fft=int(require_nested(config, "stft.n_fft")),
            hop_length=int(require_nested(config, "stft.hop_length")),
            win_length=int(require_nested(config, "stft.win_length")),
            snr_values=tuple(float(value) for value in require_nested(config, "m14_imu.snr_db")),
            mode_probabilities=(
                {str(name): float(value) for name, value in probabilities.items()}
                if isinstance(probabilities, dict)
                else None
            ),
        )

    def set_epoch(self, epoch: int) -> None:
        """Change deterministic online pairings; call once before each training epoch."""
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.clean_rows) if self.split == "train" else len(self.fixed_rows or [])

    def _rng(self, index: int, row_seed: int | None = None) -> np.random.Generator:
        if row_seed is not None:
            return np.random.default_rng(int(row_seed))
        seed_sequence = np.random.SeedSequence([self.seed, self.epoch, int(index)])
        return np.random.default_rng(seed_sequence)

    def _sample_balanced_imu(self, rng: np.random.Generator) -> dict[str, str]:
        participants = sorted(self.imu_groups)
        participant = participants[int(rng.integers(0, len(participants)))]
        conditions = sorted(self.imu_groups[participant])
        condition = conditions[int(rng.integers(0, len(conditions)))]
        rows = self.imu_groups[participant][condition]
        return rows[int(rng.integers(0, len(rows)))]

    def _load_pair(
        self,
        clean_path: Path,
        imu_path: Path,
        rng: np.random.Generator,
        mode: str,
        fixed_snr: float | None,
    ) -> dict[str, Any]:
        clean_item = self._clean_cache.get(clean_path)
        if clean_item is None:
            with np.load(clean_path, allow_pickle=False) as item:
                clean_item = {
                    "clean": np.asarray(item["clean_pcg"], dtype=np.float32),
                    "ecg": np.asarray(item["ecg"], dtype=np.float32),
                    "ecg_beat": np.asarray(item["ecg_beat"], dtype=np.float32),
                    "s1s2": np.asarray(item["s1s2_weak"], dtype=np.float32),
                    "fs": int(item["fs"]),
                    "subject": str(item["subject_id"]),
                    "record": str(item["record_id"]),
                    "start": int(item["window_start"]),
                }
            self._clean_cache[clean_path] = clean_item
        clean = clean_item["clean"]
        ecg = clean_item["ecg"]
        ecg_beat = clean_item["ecg_beat"]
        s1s2 = clean_item["s1s2"]
        clean_fs = int(clean_item["fs"])
        clean_subject = str(clean_item["subject"])
        clean_record = str(clean_item["record"])
        clean_start = int(clean_item["start"])

        imu_item = self._imu_cache.get(imu_path)
        if imu_item is None:
            with np.load(imu_path, allow_pickle=False) as item:
                imu = np.asarray(item["imu"], dtype=np.float32)
                imu_fs = int(item["fs"])
                participant = str(item["participant_id"])
                condition = str(item["condition"])
                imu_record = str(item["record_id"])
                imu_start = float(item["window_start_sec"])
            raw_features, frame_valid = extract_imu_frame_features(
                imu,
                fs=self.fs,
                hop_length=self.hop_length,
                win_length=self.win_length,
                n_frames=self.n_frames,
            )
            normalized_features = normalize_imu_features(raw_features, self.stats)
            envelope = motion_envelope_from_features(
                normalized_features,
                self.stats,
                frame_valid,
            )
            imu_item = {
                "imu": imu,
                "fs": imu_fs,
                "participant": participant,
                "condition": condition,
                "record": imu_record,
                "start": imu_start,
                "features": normalized_features,
                "valid": frame_valid,
                "envelope": envelope,
                "score": motion_score(envelope),
            }
            self._imu_cache[imu_path] = imu_item
        imu = imu_item["imu"]
        imu_fs = int(imu_item["fs"])
        participant = str(imu_item["participant"])
        condition = str(imu_item["condition"])
        imu_record = str(imu_item["record"])
        imu_start = float(imu_item["start"])

        if clean_fs != self.fs or imu_fs != self.fs:
            raise ValueError(
                f"Expected {self.fs} Hz, got clean={clean_fs}, IMU={imu_fs}: "
                f"{clean_path}, {imu_path}"
            )
        if clean.shape != (len(imu),):
            raise ValueError(
                f"BSSLAB/IMU window length mismatch: {clean.shape} versus {imu.shape}"
            )

        normalized_features = imu_item["features"]
        frame_valid = imu_item["valid"]
        envelope = imu_item["envelope"]
        score = float(imu_item["score"])
        snr = (
            float(fixed_snr)
            if fixed_snr is not None
            else choose_motion_conditioned_snr(self.snr_values, score, rng)
        )
        result = build_imu_conditioned_noisy(
            clean=clean,
            normalized_imu_features=normalized_features,
            motion_envelope=envelope,
            fs=self.fs,
            hop_length=self.hop_length,
            snr_db=snr,
            mode=mode,
            rng=rng,
        )

        imu_present = mode != "clean_identity"
        if not imu_present:
            normalized_features = np.zeros_like(normalized_features)
            envelope = np.zeros_like(envelope)
            frame_valid = np.zeros_like(frame_valid)
            imu_for_model = np.zeros_like(imu.T)
        else:
            imu_for_model = imu.T

        return {
            "noisy_pcg": torch.from_numpy(result.noisy_pcg),
            "clean_pcg": torch.from_numpy(clean),
            "noise_signal": torch.from_numpy(result.noise_signal),
            "ecg": torch.from_numpy(ecg),
            "ecg_beat": torch.from_numpy(ecg_beat),
            "s1s2_weak": torch.from_numpy(s1s2),
            "imu_raw": torch.from_numpy(imu_for_model.astype(np.float32)),
            "imu_feat": torch.from_numpy(normalized_features.astype(np.float32)),
            "imu_valid_mask": torch.from_numpy(frame_valid.astype(np.float32)),
            "motion_envelope": torch.from_numpy(envelope.astype(np.float32)),
            "imu_present": torch.tensor(float(imu_present), dtype=torch.float32),
            "motion_score": torch.tensor(score, dtype=torch.float32),
            "sqi_target": torch.tensor(result.sqi_target, dtype=torch.float32),
            "snr_db": torch.tensor(result.snr_db, dtype=torch.float32),
            "achieved_snr_db": torch.tensor(result.achieved_snr_db, dtype=torch.float32),
            "fs": self.fs,
            "artifact_mode": mode,
            "contamination_components": result.components,
            "bsslab_subject_id": clean_subject,
            "bsslab_record_id": clean_record,
            "bsslab_window_start": clean_start,
            "imu_subject_id": participant,
            "imu_condition": condition,
            "imu_record_id": imu_record,
            "imu_window_start_sec": imu_start,
            "clean_path": str(clean_path),
            "imu_path": str(imu_path),
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self.split == "train":
            clean_row = self.clean_rows[index]
            rng = self._rng(index)
            imu_row = self._sample_balanced_imu(rng)
            mode = str(rng.choice(self.mode_names, p=self.mode_probs))
            fixed_snr = None
        else:
            if self.fixed_rows is None:
                raise RuntimeError("Fixed pair rows are unavailable.")
            pair = self.fixed_rows[index]
            rng = self._rng(index, int(pair["seed"]))
            clean_row = pair
            imu_row = pair
            mode = pair["artifact_mode"]
            fixed_snr = float(pair["snr_db"])

        clean_path = _resolve_path(clean_row["clean_path"], self.project_root)
        imu_path = _resolve_path(imu_row["imu_path"], self.project_root)
        return self._load_pair(clean_path, imu_path, rng, mode, fixed_snr)
