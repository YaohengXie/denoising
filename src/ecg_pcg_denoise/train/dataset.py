from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from ecg_pcg_denoise.utils.files import list_files


class MixedWindowDataset(Dataset):
    def __init__(self, windows_dir: str | Path, split: str, cache_in_memory: bool = False) -> None:
        self.root = Path(windows_dir) / "mixed" / split
        self.files = list_files(self.root, (".npz",))
        if not self.files:
            raise FileNotFoundError(f"No mixed .npz windows found under {self.root}")
        self.cache: list[dict[str, Any]] | None = None
        if cache_in_memory:
            print(f"Caching {len(self.files)} {split} windows from {self.root}")
            self.cache = []
            for index, path in enumerate(self.files, start=1):
                self.cache.append(self._load_path(path))
                if index % 5000 == 0 or index == len(self.files):
                    print(f"Cached {index}/{len(self.files)} {split} windows")

    def __len__(self) -> int:
        return len(self.files)

    @staticmethod
    def _load_path(path: Path) -> dict[str, Any]:
        with np.load(path, allow_pickle=False) as item:
            return {
                "noisy_pcg": torch.from_numpy(np.asarray(item["noisy_pcg"], dtype=np.float32)),
                "clean_pcg": torch.from_numpy(np.asarray(item["clean_pcg"], dtype=np.float32)),
                "ecg_beat": torch.from_numpy(np.asarray(item["ecg_beat"], dtype=np.float32)),
                "s1s2_weak": torch.from_numpy(np.asarray(item["s1s2_weak"], dtype=np.float32)),
                "sqi_target": torch.tensor(float(item["sqi_target"]), dtype=torch.float32),
                "snr_db": torch.tensor(float(item["snr_db"]), dtype=torch.float32),
                "fs": int(item["fs"]),
                "path": str(path),
            }

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self.cache is not None:
            return self.cache[index]
        return self._load_path(self.files[index])


_IN_MEMORY_DATASETS: dict[tuple[str, str], MixedWindowDataset] = {}


def get_mixed_window_dataset(
    windows_dir: str | Path,
    split: str,
    cache_in_memory: bool = False,
) -> MixedWindowDataset:
    if not cache_in_memory:
        return MixedWindowDataset(windows_dir, split)
    key = (str(Path(windows_dir).resolve()), split)
    if key not in _IN_MEMORY_DATASETS:
        _IN_MEMORY_DATASETS[key] = MixedWindowDataset(windows_dir, split, cache_in_memory=True)
    return _IN_MEMORY_DATASETS[key]
