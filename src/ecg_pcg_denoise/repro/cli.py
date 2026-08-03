"""Examiner-facing command line for M14.3-v2 result reproduction."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from ecg_pcg_denoise.repro.common import write_report
from ecg_pcg_denoise.repro.exit_codes import ExitCode
from ecg_pcg_denoise.repro.golden import compare_results
from ecg_pcg_denoise.repro.integrity import verify_checkpoints, verify_dataset
from ecg_pcg_denoise.repro.runner import ReproductionSpecError, run_reproduction


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ecg_pcg_denoise.repro",
        description="Verify the controlled processed data and reproduce fixed-checkpoint M14.3-v2 results.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    verify = commands.add_parser("verify", help="Verify public checkpoints and controlled data.")
    verify.add_argument("--data-root", type=Path, required=True)
    verify.add_argument("--repo-root", type=Path, default=None)
    verify.add_argument("--report-root", type=Path, default=Path("reproduction_runs/verification"))
    verify.add_argument("--json", action="store_true")

    run = commands.add_parser("run", help="Run the complete fixed-checkpoint evaluation protocol.")
    run.add_argument("--data-root", type=Path, required=True)
    run.add_argument("--run-root", type=Path, required=True)
    run.add_argument("--repo-root", type=Path, default=None)
    run.add_argument("--python", default=None)
    run.add_argument("--device", default=None, help="Optional PyTorch device override, e.g. cpu or cuda.")
    run.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify checkpoints and write the 22-command plan without reading the controlled data.",
    )

    compare = commands.add_parser("compare-results", help="Compare canonical JSON with thesis values.")
    compare.add_argument("--expected", type=Path, required=True)
    compare.add_argument("--actual", type=Path, required=True)
    compare.add_argument("--report", type=Path, default=None)
    compare.add_argument("--json", action="store_true")
    return parser


def _repo_root(value: Path | None) -> Path:
    return value.resolve() if value else Path(__file__).resolve().parents[3]


def _require_output_outside_data(data_root: Path, output_root: Path) -> None:
    data = data_root.resolve()
    output = output_root.resolve()
    if output == data or data in output.parents:
        raise ReproductionSpecError("Report/run output must be outside the read-only data directory.")


def _print_report(report: dict[str, Any], full: bool) -> None:
    if full:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return
    summary = report["summary"]
    print(
        f"{report['kind']}: {report['status'].upper()} "
        f"(exit={report['exit_code']}, errors={summary['errors']}, warnings={summary['warnings']})"
    )
    for issue in report["issues"][:20]:
        location = f" [{issue['path']}]" if issue.get("path") else ""
        print(f"- {issue['code']}{location}: {issue['message']}")


def main(argv: Sequence[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="backslashreplace")
    args = _parser().parse_args(argv)
    try:
        if args.command == "compare-results":
            report = compare_results(args.expected, args.actual)
            if args.report:
                write_report(args.report, report)
            _print_report(report, args.json)
            return int(report["exit_code"])

        repository = _repo_root(args.repo_root)
        if args.command == "verify":
            _require_output_outside_data(args.data_root, args.report_root)
            output = args.report_root.resolve()
            checkpoint_report = verify_checkpoints(repository)
            write_report(output / "checkpoint_integrity_report.json", checkpoint_report)
            _print_report(checkpoint_report, args.json)
            if checkpoint_report["exit_code"] != 0:
                return int(checkpoint_report["exit_code"])
            dataset_report = verify_dataset(
                args.data_root,
                repository / "repro" / "dataset_contract.json",
            )
            write_report(output / "dataset_integrity_report.json", dataset_report)
            _print_report(dataset_report, args.json)
            return int(dataset_report["exit_code"])

        exit_code, plan = run_reproduction(
            args.data_root,
            args.run_root,
            repo_root=repository,
            python_executable=args.python,
            device=args.device,
            dry_run=args.dry_run,
        )
        label = "DRY-RUN PASS" if args.dry_run and exit_code == 0 else ("PASS" if exit_code == 0 else "FAIL")
        print(
            f"m143_v2_reproduction: {label} "
            f"(exit={exit_code}, commands={len(plan.commands)}, run={plan.run_root})"
        )
        return exit_code
    except (ReproductionSpecError, FileExistsError, ValueError) as error:
        print(f"Invalid reproduction specification (exit=2): {error}")
        return int(ExitCode.INVALID_SPEC)
    except FileNotFoundError as error:
        print(f"Required artifact missing (exit=3): {error}")
        return int(ExitCode.MISSING_ARTIFACT)
    except Exception as error:  # pragma: no cover
        print(f"Reproduction checker internal error: {type(error).__name__}: {error}")
        return int(ExitCode.INTERNAL_ERROR)
