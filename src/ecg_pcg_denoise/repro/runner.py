"""One-command fixed-checkpoint reproduction of the thesis M14.3-v2 results."""

from __future__ import annotations

import copy
import json
import shutil
import subprocess
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ecg_pcg_denoise.repro.canonical import write_canonical_actual
from ecg_pcg_denoise.repro.common import write_report
from ecg_pcg_denoise.repro.exit_codes import ExitCode
from ecg_pcg_denoise.repro.golden import compare_results
from ecg_pcg_denoise.repro.integrity import verify_checkpoints, verify_dataset
from ecg_pcg_denoise.utils.config import load_config


FOLDS = ("M001_S01", "M001_S02", "M001_S03", "M001_S04")
FOLD_CONFIGS = {
    fold: f"configs/bsslab_m143_imu_fold{index}_aligned_calibrated_v2.yaml"
    for index, fold in enumerate(FOLDS, start=1)
}


class ReproductionSpecError(ValueError):
    """A requested run cannot be represented safely."""


@dataclass(frozen=True)
class ReproductionPlan:
    repo_root: Path
    data_root: Path
    run_root: Path
    runtime_config_root: Path
    outputs_root: Path
    report_root: Path
    expected_results: Path
    actual_results: Path
    analysis_summary: Path
    fold_summary: Path
    commands: tuple[tuple[str, ...], ...]
    selection_copies: tuple[tuple[Path, Path], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "mode": "fixed_checkpoint_evaluation",
            "repo_root": str(self.repo_root),
            "data_root": str(self.data_root),
            "run_root": str(self.run_root),
            "runtime_config_root": str(self.runtime_config_root),
            "outputs_root": str(self.outputs_root),
            "report_root": str(self.report_root),
            "expected_results": str(self.expected_results),
            "actual_results": str(self.actual_results),
            "commands": [list(command) for command in self.commands],
            "selection_copies": [
                {"source": str(source), "destination": str(destination)}
                for source, destination in self.selection_copies
            ],
        }


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _python_command(value: str | Path | None) -> str:
    if value is None:
        return sys.executable
    raw = str(value)
    candidate = Path(raw).expanduser()
    if candidate.is_absolute() or candidate.parent != Path("."):
        resolved = candidate.resolve()
        if not resolved.is_file():
            raise ReproductionSpecError(f"Python executable does not exist: {resolved}")
        return str(resolved)
    located = shutil.which(raw)
    if located is None:
        raise ReproductionSpecError(f"Python command is unavailable on PATH: {raw}")
    return located


def _write_runtime_config(path: Path, config: dict[str, Any]) -> None:
    """Write JSON syntax to a .yaml file; JSON is a strict YAML subset."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _prepare_run_root(run_root: Path) -> None:
    run_root.mkdir(parents=True, exist_ok=True)
    allowed = {"environment.json"}
    occupied = [path for path in run_root.iterdir() if path.name not in allowed]
    if occupied:
        raise FileExistsError(
            "Run root is not clean; choose a new --run-root: "
            + ", ".join(path.name for path in occupied[:5])
        )


def _require_output_outside_data(data_root: Path, output_root: Path) -> None:
    if output_root == data_root or data_root in output_root.parents:
        raise ReproductionSpecError("Output root must be outside the read-only data directory.")


def build_plan(
    data_root: str | Path,
    run_root: str | Path,
    *,
    repo_root: str | Path | None = None,
    python_executable: str | Path | None = None,
    expected_results_path: str | Path | None = None,
    device: str | None = None,
) -> ReproductionPlan:
    repository = Path(repo_root).resolve() if repo_root else _default_repo_root()
    data = Path(data_root).resolve()
    run = Path(run_root).resolve()
    if data.name != "data":
        raise ReproductionSpecError(
            "The controlled root directory must be named 'data' because frozen pair paths "
            "begin with data/windows/."
        )
    _require_output_outside_data(data, run)
    python = _python_command(python_executable)
    expected = (
        Path(expected_results_path).resolve()
        if expected_results_path
        else repository / "repro" / "expected_results.json"
    )
    if not expected.is_file():
        raise FileNotFoundError(expected)

    configs = repository / "configs"
    required_configs = [
        configs / "bsslab_esc50_v2_m7_robust_sqi_eval.yaml",
        configs / "bsslab_m141_imu_data_balanced.yaml",
        *[repository / FOLD_CONFIGS[fold] for fold in FOLDS],
    ]
    missing = [path for path in required_configs if not path.is_file()]
    if missing:
        raise FileNotFoundError(", ".join(str(path) for path in missing))

    runtime_configs = run / "runtime_configs"
    outputs = run / "outputs"
    reports = run / "reports" / "strict_esc50_m143_v2"
    m7_checkpoint = repository / "checkpoints" / "m7_v2" / "best.pt"
    m7_config = copy.deepcopy(load_config(required_configs[0]))
    m7_output = outputs / "bsslab_esc50_v2_m7_robust_sqi_eval"
    m7_config.setdefault("paths", {})["windows_dir"] = str(
        data / "windows" / "bsslab_esc50_v2"
    )
    m7_config["paths"]["output_dir"] = str(m7_output)
    if device:
        m7_config.setdefault("training", {})["device"] = device
    m7_runtime = runtime_configs / "m7_eval.yaml"
    _write_runtime_config(m7_runtime, m7_config)

    data_config = copy.deepcopy(load_config(required_configs[1]))
    data_config.setdefault("paths", {})["project_root"] = str(data.parent)
    data_config["paths"]["clean_windows_dir"] = str(
        data / "windows" / "bsslab_musan" / "clean"
    )
    data_config["paths"]["motema_windows_dir"] = str(
        data / "windows" / "motema_external_m7" / "windows"
    )
    data_config["paths"]["m14_manifest_dir"] = str(
        data / "manifests" / "bsslab_m14_imu"
    )
    data_runtime = runtime_configs / "m14_data.yaml"
    _write_runtime_config(data_runtime, data_config)

    commands: list[tuple[str, ...]] = [
        (
            python,
            "-m",
            "ecg_pcg_denoise.train.eval_denoise",
            str(m7_runtime),
            "--checkpoint",
            str(m7_checkpoint),
        )
    ]
    selection_copies: list[tuple[Path, Path]] = []
    for fold in FOLDS:
        fold_output = outputs / f"bsslab_m143_imu_fold_{fold}_aligned_calibrated_v2"
        checkpoint_root = repository / "checkpoints" / "m143_v2" / fold
        checkpoint = checkpoint_root / "best_safe.pt"
        selection_copies.append(
            (
                checkpoint_root / "checkpoint_selection_summary.json",
                fold_output / "checkpoint_selection_summary.json",
            )
        )
        config = copy.deepcopy(load_config(repository / FOLD_CONFIGS[fold]))
        config.setdefault("paths", {})["data_config"] = str(data_runtime)
        config["paths"]["base_m7_checkpoint"] = str(m7_checkpoint)
        config["paths"]["output_dir"] = str(fold_output)
        if device:
            config.setdefault("training", {})["device"] = device
        config_path = runtime_configs / f"m143_{fold}.yaml"
        _write_runtime_config(config_path, config)
        prefix = (
            python,
            "-m",
            "ecg_pcg_denoise.train.eval_m143_v2",
            str(config_path),
            "--checkpoint",
            str(checkpoint),
        )
        commands.append(
            prefix
            + (
                "--split",
                "val",
                "--bsslab-subjects",
                "10",
                "--output-dir",
                str(fold_output / "eval_calibration"),
            )
        )
        commands.append(prefix + ("--split", "test"))
        for lag, directory in (
            ("0.128", "eval_lag_0128"),
            ("0.256", "eval_lag_0256"),
            ("1.024", "eval_lag_1024"),
        ):
            commands.append(
                prefix
                + (
                    "--split",
                    "test",
                    "--imu-shift-seconds",
                    lag,
                    "--output-dir",
                    str(fold_output / directory),
                )
            )

    commands.append(
        (
            python,
            "-m",
            "ecg_pcg_denoise.train.report_strict_v2",
            "--m7-root",
            str(m7_output),
            "--outputs-root",
            str(outputs),
            "--report-root",
            str(reports),
            "--manifests-root",
            str(data / "manifests"),
        )
    )
    return ReproductionPlan(
        repo_root=repository,
        data_root=data,
        run_root=run,
        runtime_config_root=runtime_configs,
        outputs_root=outputs,
        report_root=reports,
        expected_results=expected,
        actual_results=run / "canonical_actual_results.json",
        analysis_summary=reports / "data" / "analysis_summary.json",
        fold_summary=reports / "data" / "m143_fold_summary.csv",
        commands=tuple(commands),
        selection_copies=tuple(selection_copies),
    )


def run_reproduction(
    data_root: str | Path,
    run_root: str | Path,
    *,
    repo_root: str | Path | None = None,
    python_executable: str | Path | None = None,
    expected_results_path: str | Path | None = None,
    device: str | None = None,
    dry_run: bool = False,
) -> tuple[int, ReproductionPlan]:
    data = Path(data_root).resolve()
    run = Path(run_root).resolve()
    _require_output_outside_data(data, run)
    _prepare_run_root(run)
    plan = build_plan(
        data_root,
        run,
        repo_root=repo_root,
        python_executable=python_executable,
        expected_results_path=expected_results_path,
        device=device,
    )
    (run / "checkpoint_plan.json").write_text(
        json.dumps(plan.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    started_at = datetime.now(timezone.utc).isoformat()

    def finish(exit_code: int, stage: str) -> tuple[int, ReproductionPlan]:
        write_report(
            run / "reproduction_status.json",
            {
                "schema_version": 1,
                "mode": "fixed_checkpoint_evaluation",
                "status": "pass" if exit_code == 0 else "fail",
                "exit_code": int(exit_code),
                "stage": stage,
                "started_at_utc": started_at,
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                "run_root": str(run),
                "golden_comparison_report": str(run / "golden_comparison_report.json"),
            },
        )
        return int(exit_code), plan

    try:
        checkpoint_report = verify_checkpoints(plan.repo_root)
        write_report(run / "checkpoint_integrity_report.json", checkpoint_report)
        if checkpoint_report["exit_code"] != 0:
            return finish(int(checkpoint_report["exit_code"]), "checkpoint_integrity")
        if dry_run:
            return finish(int(ExitCode.OK), "dry_run_plan")

        contract = plan.repo_root / "repro" / "dataset_contract.json"
        dataset_report = verify_dataset(plan.data_root, contract)
        write_report(run / "dataset_integrity_report.json", dataset_report)
        if dataset_report["exit_code"] != 0:
            return finish(int(dataset_report["exit_code"]), "dataset_integrity")

        for source, destination in plan.selection_copies:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

        logs = run / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        records: list[dict[str, Any]] = []
        for index, command in enumerate(plan.commands, start=1):
            print(f"[{index:02d}/{len(plan.commands):02d}] {' '.join(command)}", flush=True)
            completed = subprocess.run(
                command,
                cwd=plan.repo_root,
                check=False,
                capture_output=True,
                text=True,
            )
            log_path = logs / f"command_{index:02d}.log"
            log_path.write_text(
                completed.stdout
                + ("\n[stderr]\n" if completed.stderr else "")
                + completed.stderr,
                encoding="utf-8",
            )
            records.append(
                {
                    "index": index,
                    "command": list(command),
                    "return_code": completed.returncode,
                    "log": str(log_path),
                }
            )
            if completed.returncode != 0:
                write_report(run / "command_provenance.json", {"schema_version": 1, "commands": records})
                return finish(int(ExitCode.PIPELINE_FAILURE), f"command_{index:02d}")
        write_report(run / "command_provenance.json", {"schema_version": 1, "commands": records})

        write_canonical_actual(
            plan.analysis_summary,
            plan.fold_summary,
            plan.outputs_root,
            plan.actual_results,
        )
        comparison = compare_results(plan.expected_results, plan.actual_results)
        write_report(run / "golden_comparison_report.json", comparison)
        return finish(int(comparison["exit_code"]), "golden_result_comparison")
    except Exception as error:  # retain a status file even at the final boundary
        (run / "internal_error.log").write_text(traceback.format_exc(), encoding="utf-8")
        print(f"Reproduction failed: {type(error).__name__}: {error}", file=sys.stderr)
        return finish(int(ExitCode.INTERNAL_ERROR), "internal_error")
