"""Stable process exit codes for reproduction checks."""

from __future__ import annotations

from enum import IntEnum


class ExitCode(IntEnum):
    """Public exit-code contract used by the command-line interface."""

    OK = 0
    INVALID_SPEC = 2
    MISSING_ARTIFACT = 3
    INTEGRITY_MISMATCH = 4
    BINDING_MISMATCH = 5
    RESULT_MISMATCH = 6
    PIPELINE_FAILURE = 7
    INTERNAL_ERROR = 70
