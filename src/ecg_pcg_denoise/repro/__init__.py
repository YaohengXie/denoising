"""Examiner-facing M14.3-v2 integrity and result-reproduction checks."""

from ecg_pcg_denoise.repro.exit_codes import ExitCode
from ecg_pcg_denoise.repro.golden import compare_results
from ecg_pcg_denoise.repro.integrity import verify_checkpoints, verify_dataset
from ecg_pcg_denoise.repro.runner import build_plan, run_reproduction

__all__ = [
    "ExitCode",
    "build_plan",
    "compare_results",
    "run_reproduction",
    "verify_checkpoints",
    "verify_dataset",
]
