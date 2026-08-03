"""Capture a non-sensitive runtime manifest for reproduction records.

The script only uses the Python standard library unless PyTorch is already
installed.  Missing optional packages or GPU utilities are recorded instead of
causing the capture to fail.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "environment" / "runtime_environment.json"
PACKAGES = (
    "torch",
    "numpy",
    "scipy",
    "PyYAML",
    "matplotlib",
    "pytest",
)


def project_runtime() -> dict[str, Any]:
    try:
        import ecg_pcg_denoise
    except Exception as exc:  # pragma: no cover - depends on installation state
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    module_path = Path(ecg_pcg_denoise.__file__).resolve()
    try:
        display_path = module_path.relative_to(ROOT).as_posix()
    except ValueError:
        display_path = str(module_path)
    try:
        distribution_version = importlib.metadata.version("ecg-pcg-denoise")
    except importlib.metadata.PackageNotFoundError:
        distribution_version = None
    return {
        "available": True,
        "source_version": ecg_pcg_denoise.__version__,
        "source_file": display_path,
        "installed_distribution_version": distribution_version,
    }


def command_output(command: list[str]) -> str | None:
    """Return stripped command output, or ``None`` when unavailable."""
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return completed.stdout.strip()


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def torch_runtime() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on the local runtime
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}

    cuda_available = bool(torch.cuda.is_available())
    devices: list[dict[str, Any]] = []
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_bytes": int(properties.total_memory),
                    "compute_capability": [int(properties.major), int(properties.minor)],
                }
            )

    return {
        "available": True,
        "version": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "devices": devices,
    }


def git_runtime() -> dict[str, Any]:
    commit = command_output(["git", "rev-parse", "HEAD"])
    status = command_output(["git", "status", "--short"])
    return {
        "commit": commit,
        "working_tree_clean": status == "" if status is not None else None,
        "status": status,
    }


def build_manifest() -> dict[str, Any]:
    nvidia_smi = shutil.which("nvidia-smi")
    gpu_query = None
    if nvidia_smi:
        gpu_query = command_output(
            [
                nvidia_smi,
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ]
        )

    return {
        "schema_version": 1,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": git_runtime(),
        "project": project_runtime(),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable_name": Path(sys.executable).name,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
        },
        "packages": package_versions(),
        "torch": torch_runtime(),
        "nvidia_smi": gpu_query,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"JSON output path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Also print the complete JSON manifest to standard output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_manifest()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    try:
        display_path = str(output.relative_to(ROOT))
    except ValueError:
        # ``ascii`` keeps this message printable on legacy Windows code pages
        # when a parent directory contains non-ASCII characters.
        display_path = ascii(str(output))
    print(f"Environment manifest written to {display_path}")
    if args.stdout:
        print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
