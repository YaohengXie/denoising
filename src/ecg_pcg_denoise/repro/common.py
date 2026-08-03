"""Shared report and JSON helpers for reproducibility checks."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from ecg_pcg_denoise.repro.exit_codes import ExitCode


@dataclass(frozen=True)
class Issue:
    """A machine-readable audit finding."""

    category: str
    code: str
    message: str
    path: str | None = None
    severity: str = "error"


def make_report(
    kind: str,
    issues: Iterable[Issue],
    exit_code: ExitCode,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    issue_list = list(issues)
    return {
        "schema_version": 1,
        "kind": kind,
        "status": "pass" if exit_code == ExitCode.OK else "fail",
        "exit_code": int(exit_code),
        "summary": {
            "errors": sum(item.severity == "error" for item in issue_list),
            "warnings": sum(item.severity == "warning" for item in issue_list),
        },
        "issues": [asdict(item) for item in issue_list],
        "details": details or {},
    }


def load_json_object(path: Path) -> dict[str, Any]:
    """Load a UTF-8 JSON object, rejecting arrays and scalar documents."""

    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def dotted_get(value: Any, path: str) -> Any:
    """Resolve a non-empty dot-separated path through JSON objects."""

    if not path or path.startswith(".") or path.endswith(".") or ".." in path:
        raise KeyError(path)
    current = value
    for component in path.split("."):
        if not isinstance(current, dict) or component not in current:
            raise KeyError(path)
        current = current[component]
    return current
