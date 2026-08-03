"""Compare generated JSON results with an explicit golden-result contract."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from ecg_pcg_denoise.repro.common import Issue, dotted_get, load_json_object, make_report
from ecg_pcg_denoise.repro.exit_codes import ExitCode


_REFERENCE_SECTIONS = (
    "experiment",
    "population_counts",
    "leakage_audit",
    "m7_strict_esc50",
    "m143_synthetic_imu_loso",
    "folds",
)


def _validate_checks(spec: dict[str, Any]) -> list[dict[str, Any]]:
    if spec.get("schema_version") != 1:
        raise ValueError("golden result schema_version must be 1")
    checks = spec.get("checks")
    if not isinstance(checks, list) or not checks:
        raise ValueError("checks must be a non-empty list")
    seen: set[str] = set()
    output: list[dict[str, Any]] = []
    for index, item in enumerate(checks):
        if not isinstance(item, dict):
            raise ValueError(f"checks[{index}] must be an object")
        path = item.get("path")
        if not isinstance(path, str) or not path or path in seen:
            raise ValueError(f"checks[{index}].path must be a unique non-empty string")
        seen.add(path)
        if "expected" not in item:
            raise ValueError(f"checks[{index}] must contain expected")
        comparison = item.get("comparison")
        if comparison is None:
            comparison = "close" if "atol" in item or "rtol" in item else "exact"
        if comparison not in {"exact", "close"}:
            raise ValueError(f"checks[{index}].comparison must be exact or close")
        expected = item["expected"]
        atol = item.get("atol", 0.0)
        rtol = item.get("rtol", 0.0)
        if comparison == "close":
            if isinstance(expected, bool) or not isinstance(expected, (int, float)):
                raise ValueError(f"checks[{index}] close comparison requires a numeric expected value")
            if "atol" not in item and "rtol" not in item:
                raise ValueError(f"checks[{index}] close comparison must declare atol or rtol")
            for name, value in (("atol", atol), ("rtol", rtol)):
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(f"checks[{index}].{name} must be numeric")
                if not math.isfinite(float(value)) or float(value) < 0.0:
                    raise ValueError(f"checks[{index}].{name} must be finite and non-negative")
            if not math.isfinite(float(expected)):
                raise ValueError(f"checks[{index}].expected must be finite")
        output.append(
            {
                "path": path,
                "expected": expected,
                "comparison": comparison,
                "atol": float(atol),
                "rtol": float(rtol),
            }
        )
    return output


def _flatten(value: Any, prefix: str) -> list[tuple[str, Any]]:
    if isinstance(value, dict):
        output: list[tuple[str, Any]] = []
        for key, nested in value.items():
            output.extend(_flatten(nested, f"{prefix}.{key}" if prefix else key))
        return output
    return [(prefix, value)]


def _checks_from_structured_reference(spec: dict[str, Any]) -> list[dict[str, Any]]:
    acceptance = spec.get("acceptance")
    if not isinstance(acceptance, dict):
        raise ValueError("structured reference must contain an acceptance object")
    absolute = float(acceptance["continuous_absolute_tolerance"])
    relative = float(acceptance["continuous_relative_tolerance"])
    protected = float(acceptance["protected_output_max_abs_tolerance"])
    threshold = float(acceptance["refitted_threshold_absolute_tolerance"])
    if any(not math.isfinite(value) or value < 0.0 for value in (absolute, relative, protected, threshold)):
        raise ValueError("structured-reference tolerances must be finite and non-negative")

    checks: list[dict[str, Any]] = []
    for section in _REFERENCE_SECTIONS:
        if section not in spec:
            raise ValueError(f"structured reference lacks required section {section!r}")
        for path, expected in _flatten(spec[section], section):
            if isinstance(expected, bool) or isinstance(expected, (str, int)):
                checks.append(
                    {
                        "path": path,
                        "expected": expected,
                        "comparison": "exact",
                        "atol": 0.0,
                        "rtol": 0.0,
                    }
                )
                continue
            if not isinstance(expected, float) or not math.isfinite(expected):
                raise ValueError(f"Unsupported structured reference value at {path}")
            if "protected_output" in path:
                atol, rtol = protected, 0.0
            elif path.endswith(".artifact_threshold"):
                atol, rtol = threshold, 0.0
            else:
                atol, rtol = absolute, relative
            checks.append(
                {
                    "path": path,
                    "expected": expected,
                    "comparison": "close",
                    "atol": atol,
                    "rtol": rtol,
                }
            )
    return checks


def compare_results(expected_path: str | Path, actual_path: str | Path) -> dict[str, Any]:
    """Compare one actual JSON result document to a golden-result specification."""

    expected_file = Path(expected_path)
    actual_file = Path(actual_path)
    details: dict[str, Any] = {
        "expected": str(expected_file.resolve()),
        "actual": str(actual_file.resolve()),
        "checks_total": 0,
        "checks_passed": 0,
        "checks_failed": 0,
    }
    missing = [path for path in (expected_file, actual_file) if not path.is_file()]
    if missing:
        issues = [
            Issue("missing_artifact", "result_file_missing", "Required result file does not exist.", str(path))
            for path in missing
        ]
        return make_report("golden_result_comparison", issues, ExitCode.MISSING_ARTIFACT, details)
    try:
        spec = load_json_object(expected_file)
        checks = (
            _validate_checks(spec)
            if "checks" in spec
            else _checks_from_structured_reference(spec)
        )
        actual = load_json_object(actual_file)
    except (OSError, ValueError, TypeError) as error:
        issue = Issue("invalid_spec", "invalid_result_json", str(error))
        return make_report("golden_result_comparison", [issue], ExitCode.INVALID_SPEC, details)

    details["suite_id"] = spec.get("suite_id")
    details["reference_format"] = (
        "explicit_checks_v1" if "checks" in spec else "structured_thesis_reference_v1"
    )
    if "checks" not in spec:
        details["separately_audited_reference_sections"] = [
            "checkpoint_sha256",
            "scientific_boundaries",
        ]
    details["checks_total"] = len(checks)
    issues: list[Issue] = []
    check_results: list[dict[str, Any]] = []
    for check in checks:
        path = check["path"]
        expected = check["expected"]
        try:
            observed = dotted_get(actual, path)
        except KeyError:
            issues.append(
                Issue("result_mismatch", "result_path_missing", "Actual results lack the required path.", path)
            )
            check_results.append({"path": path, "status": "fail", "reason": "missing"})
            continue

        passed = False
        error_value: float | None = None
        tolerance: float | None = None
        if check["comparison"] == "exact":
            passed = type(observed) is type(expected) and observed == expected
        elif (
            not isinstance(observed, bool)
            and isinstance(observed, (int, float))
            and math.isfinite(float(observed))
        ):
            error_value = abs(float(observed) - float(expected))
            tolerance = check["atol"] + check["rtol"] * abs(float(expected))
            passed = error_value <= tolerance

        result = {
            "path": path,
            "status": "pass" if passed else "fail",
            "comparison": check["comparison"],
            "expected": expected,
            "actual": observed,
        }
        if error_value is not None:
            result["absolute_error"] = error_value
            result["allowed_error"] = tolerance
        check_results.append(result)
        if not passed:
            issues.append(
                Issue(
                    "result_mismatch",
                    "golden_value_mismatch",
                    f"Expected {expected!r} using {check['comparison']}, found {observed!r}.",
                    path,
                )
            )

    details["checks"] = check_results
    details["checks_failed"] = len(issues)
    details["checks_passed"] = len(checks) - len(issues)
    exit_code = ExitCode.RESULT_MISMATCH if issues else ExitCode.OK
    return make_report("golden_result_comparison", issues, exit_code, details)
