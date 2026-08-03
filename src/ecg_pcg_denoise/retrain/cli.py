"""Examiner-facing command line for M14.3-v2 protocol retraining."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from ecg_pcg_denoise.repro.exit_codes import ExitCode
from ecg_pcg_denoise.retrain.runner import (
    RetrainingSpecError,
    run_retraining,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ecg_pcg_denoise.retrain",
        description=(
            "Retrain either the four M14.3-v2 IMU adapters from the archived "
            "M7-v2 base, or the complete M5-to-M14.3-v2 protocol."
        ),
    )
    parser.add_argument("--scope", choices=("adapter", "full"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--python", default=None)
    parser.add_argument("--device", default=None, help="Optional PyTorch device override.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write and validate the command plan without reading data or training.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="backslashreplace")
    args = _parser().parse_args(argv)
    try:
        exit_code, plan = run_retraining(
            scope=args.scope,
            data_root=args.data_root,
            run_root=args.run_root,
            repo_root=args.repo_root,
            python_executable=args.python,
            device=args.device,
            dry_run=args.dry_run,
        )
        label = "DRY-RUN PASS" if args.dry_run and exit_code == 0 else (
            "PASS" if exit_code == 0 else "FAIL"
        )
        print(
            f"m143_v2_{plan.scope}_retraining: {label} "
            f"(exit={exit_code}, commands={len(plan.commands)}, run={plan.run_root})"
        )
        return int(exit_code)
    except (RetrainingSpecError, FileExistsError, ValueError) as error:
        print(f"Invalid retraining specification (exit=2): {error}", file=sys.stderr)
        return int(ExitCode.INVALID_SPEC)
    except FileNotFoundError as error:
        print(f"Required artifact missing (exit=3): {error}", file=sys.stderr)
        return int(ExitCode.MISSING_ARTIFACT)
    except Exception as error:  # pragma: no cover - final CLI boundary
        print(
            f"Retraining runner internal error: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return int(ExitCode.INTERNAL_ERROR)
