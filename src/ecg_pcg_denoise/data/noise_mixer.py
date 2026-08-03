from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from ecg_pcg_denoise.utils.config import get_nested, load_config, require_nested
from ecg_pcg_denoise.utils.files import ensure_dir, list_files
from ecg_pcg_denoise.utils.signal import mix_at_snr, resample_to


@lru_cache(maxsize=2048)
def load_audio(path: Path, target_fs: int) -> np.ndarray:
    try:
        import soundfile as sf

        audio, fs = sf.read(path, always_2d=False)
    except Exception:
        fs, audio = wavfile.read(path)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if np.max(np.abs(audio)) > 2.0:
        audio = audio / (np.max(np.abs(audio)) + 1e-8)
    return resample_to(audio, int(fs), target_fs)


def crop_or_tile_noise(
    noise: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    if len(noise) >= n:
        start = int(rng.integers(0, len(noise) - n + 1))
        return noise[start : start + n].astype(np.float32), start
    repeats = int(np.ceil(n / max(1, len(noise))))
    tiled = np.tile(noise, repeats)
    return tiled[:n].astype(np.float32), 0


def sqi_target_from_snr(snr_db: float) -> float:
    if snr_db >= 10:
        return 1.0
    if snr_db >= 5:
        return 0.66
    if snr_db >= 0:
        return 0.33
    return 0.0


def collect_external_noise(
    config: dict,
    noise_type: str,
    split: str | None = None,
) -> list[Path]:
    split_directories = get_nested(config, "noise.split_directories", {})
    if split is not None and isinstance(split_directories, dict):
        split_config = split_directories.get(split, {})
        if isinstance(split_config, dict) and noise_type in split_config:
            directory_config = split_config[noise_type]
            if isinstance(directory_config, list):
                files: list[Path] = []
                for directory in directory_config:
                    files.extend(
                        list_files(
                            Path(directory),
                            (".wav", ".flac", ".ogg", ".mp3"),
                        )
                    )
                return sorted(files)
            return list_files(
                Path(directory_config),
                (".wav", ".flac", ".ogg", ".mp3"),
            )

    directories = get_nested(config, "noise.directories", {})
    if isinstance(directories, dict) and noise_type in directories:
        directory_config = directories[noise_type]
        if isinstance(directory_config, list):
            files: list[Path] = []
            for directory in directory_config:
                files.extend(list_files(Path(directory), (".wav", ".flac", ".ogg", ".mp3")))
            return sorted(files)
        return list_files(Path(directory_config), (".wav", ".flac", ".ogg", ".mp3"))

    if noise_type == "speech":
        directory = Path(require_nested(config, "noise.speech_dir"))
    elif noise_type == "music":
        directory = Path(require_nested(config, "noise.music_dir"))
    else:
        return []
    return list_files(directory, (".wav", ".flac", ".ogg", ".mp3"))


def make_noise(
    clean: np.ndarray,
    fs: int,
    noise_type: str,
    external_files: list[Path],
    rng: np.random.Generator,
) -> tuple[np.ndarray, str, int]:
    if noise_type == "gaussian":
        return (
            rng.normal(0.0, 1.0, size=len(clean)).astype(np.float32),
            "generated_gaussian",
            -1,
        )
    if not external_files:
        raise FileNotFoundError(f"No audio files found for noise_type={noise_type}")
    path = external_files[int(rng.integers(0, len(external_files)))]
    audio = load_audio(path, fs)
    cropped, crop_start = crop_or_tile_noise(audio, len(clean), rng)
    return cropped, str(path), crop_start


def mix_windows_from_config(config_path: str | Path, limit: int | None = None) -> None:
    config = load_config(config_path)
    seed = int(get_nested(config, "project.seed", 1337))
    rng = np.random.default_rng(seed)
    windows_dir = Path(require_nested(config, "paths.windows_dir"))
    clean_root = Path(get_nested(config, "paths.clean_windows_dir", windows_dir / "clean"))
    noise_types = list(require_nested(config, "noise.types"))
    snr_values = [float(x) for x in require_nested(config, "noise.snr_db")]
    examples_per_clean_window = int(get_nested(config, "noise.examples_per_clean_window", 1))

    clean_files = list_files(clean_root, (".npz",))
    if limit is not None:
        clean_files = clean_files[:limit]
    external_by_split: dict[str, dict[str, list[Path]]] = {}
    written = 0
    for clean_path in clean_files:
        clean_item = np.load(clean_path, allow_pickle=False)
        clean = clean_item["clean_pcg"].astype(np.float32)
        fs = int(clean_item["fs"])
        split = str(clean_item["split"])
        if split not in external_by_split:
            external_by_split[split] = {
                noise_type: collect_external_noise(config, noise_type, split)
                for noise_type in noise_types
            }
        external_by_type = external_by_split[split]
        stem = clean_path.stem
        for noise_type in noise_types:
            for snr in snr_values:
                for example_idx in range(examples_per_clean_window):
                    noise, noise_source, noise_crop_start = make_noise(
                        clean, fs, noise_type, external_by_type[noise_type], rng
                    )
                    noisy, scaled_noise = mix_at_snr(clean, noise, snr)
                    out_dir = ensure_dir(windows_dir / "mixed" / split)
                    out_path = out_dir / f"{stem}_{noise_type}_{snr:g}db_e{example_idx:02d}.npz"
                    np.savez_compressed(
                        out_path,
                        clean_pcg=clean,
                        noisy_pcg=noisy,
                        noise_signal=scaled_noise,
                        ecg=clean_item["ecg"].astype(np.float32),
                        ecg_beat=clean_item["ecg_beat"].astype(np.float32),
                        s1s2_weak=clean_item["s1s2_weak"].astype(np.float32),
                        r_peaks=clean_item["r_peaks"].astype(np.int64),
                        fs=fs,
                        subject_id=int(clean_item["subject_id"]),
                        record_id=str(clean_item["record_id"]),
                        site=str(clean_item["site"]),
                        split=split,
                        window_start=int(clean_item["window_start"]),
                        noise_type=noise_type,
                        noise_source=noise_source,
                        noise_split=split,
                        noise_crop_start=int(noise_crop_start),
                        noise_crop_end=(
                            int(noise_crop_start + len(clean))
                            if noise_crop_start >= 0
                            else -1
                        ),
                        snr_db=float(snr),
                        sqi_target=sqi_target_from_snr(snr),
                    )
                    written += 1
    print(f"Wrote {written} mixed noisy-clean windows to {windows_dir / 'mixed'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Mix Gaussian/speech/music noise into PCG only.")
    parser.add_argument("config", type=Path)
    parser.add_argument("--limit", type=int, default=None, help="Optional clean-window limit for smoke tests.")
    args = parser.parse_args()
    mix_windows_from_config(args.config, args.limit)


if __name__ == "__main__":
    main()
