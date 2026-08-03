"""One-command Adapter and Full protocol retraining for M14.3-v2."""

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
    fold: f"bsslab_m143_imu_fold{index}_aligned_calibrated_v2.yaml"
    for index, fold in enumerate(FOLDS, start=1)
}
M7_STAGE_CONFIGS = (
    "bsslab_esc50_v2_m5.yaml",
    "bsslab_esc50_v2_m6_multitask.yaml",
    "bsslab_esc50_v2_m7_distill.yaml",
    "bsslab_esc50_enhanced_v2_m7_robust.yaml",
    "bsslab_esc50_enhanced_v2_m7_robust_sqi.yaml",
)


class RetrainingSpecError(ValueError):
    """The requested retraining run cannot be represented safely."""


@dataclass(frozen=True)
class PlannedCommand:
    label: str
    argv: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"label": self.label, "argv": list(self.argv)}


@dataclass(frozen=True)
class RetrainingPlan:
    scope: str
    repo_root: Path
    data_root: Path
    run_root: Path
    runtime_config_root: Path
    outputs_root: Path
    report_root: Path
    m7_checkpoint: Path
    commands: tuple[PlannedCommand, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "mode": "protocol_retraining",
            "scope": self.scope,
            "claim": "protocol_level_reproduction_not_bitwise_checkpoint_reproduction",
            "repo_root": str(self.repo_root),
            "data_root": str(self.data_root),
            "run_root": str(self.run_root),
            "runtime_config_root": str(self.runtime_config_root),
            "outputs_root": str(self.outputs_root),
            "report_root": str(self.report_root),
            "m7_checkpoint": str(self.m7_checkpoint),
            "commands": [command.to_dict() for command in self.commands],
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
            raise RetrainingSpecError(f"Python executable does not exist: {resolved}")
        return str(resolved)
    located = shutil.which(raw)
    if located is None:
        raise RetrainingSpecError(f"Python command is unavailable on PATH: {raw}")
    return located


def _write_runtime_config(path: Path, config: dict[str, Any]) -> None:
    """Write JSON syntax to a .yaml path; JSON is a strict YAML subset."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _prepare_run_root(run_root: Path) -> None:
    run_root.mkdir(parents=True, exist_ok=True)
    occupied = [path for path in run_root.iterdir() if path.name != "environment.json"]
    if occupied:
        raise FileExistsError(
            "Run root is not clean; choose a new --run-root: "
            + ", ".join(path.name for path in occupied[:5])
        )


def _require_output_outside_data(data_root: Path, output_root: Path) -> None:
    if output_root == data_root or data_root in output_root.parents:
        raise RetrainingSpecError("Run output must be outside the read-only data directory.")


def _rewrite_common_data_paths(config: dict[str, Any], data: Path) -> None:
    paths = config.setdefault("paths", {})
    paths["raw_bsslab_dir"] = str(data / "raw" / "bsslab")
    paths["manifest_csv"] = str(data / "manifests" / "bsslab_manifest.csv")
    paths["processed_dir"] = str(data / "processed" / "bsslab_npz")
    paths["clean_windows_dir"] = str(data / "windows" / "bsslab_musan" / "clean")
    paths["external_noise_dir"] = str(data / "external_noise")


def _apply_device(config: dict[str, Any], device: str | None) -> None:
    if device:
        config.setdefault("training", {})["device"] = device


def build_plan(
    scope: str,
    data_root: str | Path,
    run_root: str | Path,
    *,
    repo_root: str | Path | None = None,
    python_executable: str | Path | None = None,
    device: str | None = None,
) -> RetrainingPlan:
    normalized_scope = str(scope).lower()
    if normalized_scope not in {"adapter", "full"}:
        raise RetrainingSpecError("scope must be 'adapter' or 'full'")
    repository = Path(repo_root).resolve() if repo_root else _default_repo_root()
    data = Path(data_root).resolve()
    run = Path(run_root).resolve()
    if data.name != "data":
        raise RetrainingSpecError(
            "The controlled root must be named 'data' because frozen pair paths "
            "begin with data/windows/."
        )
    _require_output_outside_data(data, run)
    python = _python_command(python_executable)
    configs = repository / "configs"
    required = [
        configs / "bsslab_esc50_v2_m7_robust_sqi_eval.yaml",
        configs / "bsslab_m141_imu_data_balanced.yaml",
        *[configs / name for name in FOLD_CONFIGS.values()],
    ]
    if normalized_scope == "full":
        required.extend(configs / name for name in M7_STAGE_CONFIGS)
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(", ".join(str(path) for path in missing))

    runtime = run / "runtime_configs"
    outputs = run / "outputs"
    reports = run / "reports" / "strict_esc50_m143_v2"
    commands: list[PlannedCommand] = []

    if normalized_scope == "adapter":
        m7_checkpoint = repository / "checkpoints" / "m7_v2" / "best.pt"
        if not m7_checkpoint.is_file():
            raise FileNotFoundError(m7_checkpoint)
    else:
        stage_outputs = {
            "m5": outputs / "bsslab_esc50_v2_m5",
            "m6": outputs / "bsslab_esc50_v2_m6_multitask",
            "distill": outputs / "bsslab_esc50_v2_m7_distill",
            "robust": outputs / "bsslab_esc50_enhanced_v2_m7_robust",
            "sqi": outputs / "bsslab_esc50_enhanced_v2_m7_robust_sqi",
        }
        stage_runtime: dict[str, Path] = {}
        stage_names = ("m5", "m6", "distill", "robust", "sqi")
        for key, source_name in zip(stage_names, M7_STAGE_CONFIGS, strict=True):
            config = copy.deepcopy(load_config(configs / source_name))
            _rewrite_common_data_paths(config, data)
            config["paths"]["windows_dir"] = str(
                data
                / "windows"
                / (
                    "bsslab_esc50_v2"
                    if key in {"m5", "m6", "distill"}
                    else "bsslab_esc50_enhanced_v2"
                )
            )
            config["paths"]["output_dir"] = str(stage_outputs[key])
            _apply_device(config, device)
            stage_runtime[key] = runtime / f"{key}.yaml"
            _write_runtime_config(stage_runtime[key], config)

        distill = load_config(stage_runtime["distill"])
        distill.setdefault("training", {})["init_checkpoint"] = str(
            stage_outputs["m6"] / "checkpoints" / "best.pt"
        )
        distill["training"]["teacher_checkpoint"] = str(
            stage_outputs["m5"] / "checkpoints" / "best.pt"
        )
        _write_runtime_config(stage_runtime["distill"], distill)
        robust = load_config(stage_runtime["robust"])
        robust.setdefault("training", {})["init_checkpoint"] = str(
            stage_outputs["distill"] / "checkpoints" / "best.pt"
        )
        _write_runtime_config(stage_runtime["robust"], robust)
        sqi = load_config(stage_runtime["sqi"])
        sqi.setdefault("training", {})["init_checkpoint"] = str(
            stage_outputs["robust"] / "checkpoints" / "best.pt"
        )
        _write_runtime_config(stage_runtime["sqi"], sqi)
        for key in stage_names:
            commands.append(
                PlannedCommand(
                    f"train_{key}",
                    (
                        python,
                        "-m",
                        "ecg_pcg_denoise.train.train_denoise",
                        str(stage_runtime[key]),
                    ),
                )
            )
        m7_checkpoint = stage_outputs["sqi"] / "checkpoints" / "best.pt"

    m7_eval = copy.deepcopy(
        load_config(configs / "bsslab_esc50_v2_m7_robust_sqi_eval.yaml")
    )
    _rewrite_common_data_paths(m7_eval, data)
    m7_eval["paths"]["windows_dir"] = str(data / "windows" / "bsslab_esc50_v2")
    m7_output = outputs / "bsslab_esc50_v2_m7_robust_sqi_eval"
    m7_eval["paths"]["output_dir"] = str(m7_output)
    _apply_device(m7_eval, device)
    m7_eval_runtime = runtime / "m7_eval.yaml"
    _write_runtime_config(m7_eval_runtime, m7_eval)
    commands.append(
        PlannedCommand(
            "evaluate_m7_strict",
            (
                python,
                "-m",
                "ecg_pcg_denoise.train.eval_denoise",
                str(m7_eval_runtime),
                "--checkpoint",
                str(m7_checkpoint),
            ),
        )
    )

    data_config = copy.deepcopy(load_config(configs / "bsslab_m141_imu_data_balanced.yaml"))
    data_paths = data_config.setdefault("paths", {})
    data_paths["project_root"] = str(data.parent)
    data_paths["clean_windows_dir"] = str(data / "windows" / "bsslab_musan" / "clean")
    data_paths["motema_windows_dir"] = str(
        data / "windows" / "motema_external_m7" / "windows"
    )
    data_paths["m14_manifest_dir"] = str(data / "manifests" / "bsslab_m14_imu")
    data_runtime = runtime / "m14_data.yaml"
    _write_runtime_config(data_runtime, data_config)

    for fold in FOLDS:
        fold_output = outputs / f"bsslab_m143_imu_fold_{fold}_aligned_calibrated_v2"
        config = copy.deepcopy(load_config(configs / FOLD_CONFIGS[fold]))
        fold_paths = config.setdefault("paths", {})
        fold_paths["data_config"] = str(data_runtime)
        fold_paths["base_m7_checkpoint"] = str(m7_checkpoint)
        fold_paths["output_dir"] = str(fold_output)
        _apply_device(config, device)
        config_path = runtime / f"m143_{fold}.yaml"
        _write_runtime_config(config_path, config)
        checkpoint = fold_output / "checkpoints" / "best_safe.pt"
        commands.extend(
            [
                PlannedCommand(
                    f"train_m143_{fold}",
                    (
                        python,
                        "-m",
                        "ecg_pcg_denoise.train.train_m143_v2",
                        str(config_path),
                    ),
                ),
                PlannedCommand(
                    f"select_m143_{fold}",
                    (
                        python,
                        "-m",
                        "ecg_pcg_denoise.train.select_m142_checkpoint",
                        str(fold_output),
                        "--sqi-tolerance",
                        "0.0",
                    ),
                ),
            ]
        )
        prefix = (
            python,
            "-m",
            "ecg_pcg_denoise.train.eval_m143_v2",
            str(config_path),
            "--checkpoint",
            str(checkpoint),
        )
        commands.append(
            PlannedCommand(
                f"calibrate_m143_{fold}",
                prefix
                + (
                    "--split",
                    "val",
                    "--bsslab-subjects",
                    "10",
                    "--output-dir",
                    str(fold_output / "eval_calibration"),
                ),
            )
        )
        commands.append(PlannedCommand(f"test_m143_{fold}", prefix + ("--split", "test")))
        for lag, directory in (
            ("0.128", "eval_lag_0128"),
            ("0.256", "eval_lag_0256"),
            ("1.024", "eval_lag_1024"),
        ):
            commands.append(
                PlannedCommand(
                    f"test_m143_{fold}_lag_{lag.replace('.', '')}",
                    prefix
                    + (
                        "--split",
                        "test",
                        "--imu-shift-seconds",
                        lag,
                        "--output-dir",
                        str(fold_output / directory),
                    ),
                )
            )

    commands.append(
        PlannedCommand(
            "report_strict_v2",
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
            ),
        )
    )
    return RetrainingPlan(
        scope=normalized_scope,
        repo_root=repository,
        data_root=data,
        run_root=run,
        runtime_config_root=runtime,
        outputs_root=outputs,
        report_root=reports,
        m7_checkpoint=m7_checkpoint,
        commands=tuple(commands),
    )


def _run_command(
    command: PlannedCommand,
    *,
    cwd: Path,
    log_path: Path,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command.argv,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    log_path.write_text(
        completed.stdout
        + ("\n[stderr]\n" if completed.stderr else "")
        + completed.stderr,
        encoding="utf-8",
    )
    return completed


def run_retraining(
    scope: str,
    data_root: str | Path,
    run_root: str | Path,
    *,
    repo_root: str | Path | None = None,
    python_executable: str | Path | None = None,
    device: str | None = None,
    dry_run: bool = False,
) -> tuple[int, RetrainingPlan]:
    data = Path(data_root).resolve()
    run = Path(run_root).resolve()
    _require_output_outside_data(data, run)
    _prepare_run_root(run)
    plan = build_plan(
        scope,
        data,
        run,
        repo_root=repo_root,
        python_executable=python_executable,
        device=device,
    )
    write_report(run / "retraining_plan.json", plan.to_dict())
    started_at = datetime.now(timezone.utc).isoformat()

    def finish(exit_code: int, stage: str) -> tuple[int, RetrainingPlan]:
        write_report(
            run / "retraining_status.json",
            {
                "schema_version": 1,
                "mode": "protocol_retraining",
                "scope": plan.scope,
                "status": "pass" if exit_code == 0 else "fail",
                "exit_code": int(exit_code),
                "stage": stage,
                "started_at_utc": started_at,
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                "run_root": str(run),
            },
        )
        return int(exit_code), plan

    try:
        if plan.scope == "adapter":
            checkpoint_report = verify_checkpoints(plan.repo_root)
            write_report(run / "checkpoint_integrity_report.json", checkpoint_report)
            if checkpoint_report["exit_code"] != 0:
                return finish(int(checkpoint_report["exit_code"]), "checkpoint_integrity")
        if dry_run:
            return finish(int(ExitCode.OK), "dry_run_plan")

        evaluation_report = verify_dataset(
            plan.data_root,
            plan.repo_root / "repro" / "dataset_contract.json",
        )
        write_report(run / "evaluation_dataset_integrity_report.json", evaluation_report)
        if evaluation_report["exit_code"] != 0:
            return finish(int(evaluation_report["exit_code"]), "evaluation_dataset_integrity")

        logs = run / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        if plan.scope == "full":
            integrity_command = PlannedCommand(
                "verify_training_extension",
                (
                    plan.commands[0].argv[0],
                    str(plan.repo_root / "scripts" / "create_training_dataset_manifest.py"),
                    "--data-root",
                    str(plan.data_root),
                    "--contract",
                    str(plan.repo_root / "repro" / "training_dataset_contract.json"),
                    "--verify",
                ),
            )
            completed = _run_command(
                integrity_command,
                cwd=plan.repo_root,
                log_path=logs / "training_dataset_integrity.log",
            )
            if completed.returncode != 0:
                return finish(int(ExitCode.INTEGRITY_MISMATCH), "training_dataset_integrity")
            try:
                integrity_payload = json.loads(completed.stdout)
            except json.JSONDecodeError:
                integrity_payload = {"status": "pass", "stdout": completed.stdout}
            write_report(run / "training_dataset_integrity_report.json", integrity_payload)

        records: list[dict[str, Any]] = []
        for index, command in enumerate(plan.commands, start=1):
            print(
                f"[{index:02d}/{len(plan.commands):02d}] {command.label}: "
                f"{' '.join(command.argv)}",
                flush=True,
            )
            log_path = logs / f"command_{index:02d}_{command.label}.log"
            completed = _run_command(command, cwd=plan.repo_root, log_path=log_path)
            records.append(
                {
                    "index": index,
                    "label": command.label,
                    "command": list(command.argv),
                    "return_code": completed.returncode,
                    "log": str(log_path),
                }
            )
            if completed.returncode != 0:
                write_report(
                    run / "command_provenance.json",
                    {"schema_version": 1, "commands": records},
                )
                return finish(int(ExitCode.PIPELINE_FAILURE), command.label)
        write_report(
            run / "command_provenance.json",
            {"schema_version": 1, "commands": records},
        )

        actual = run / "canonical_retrained_results.json"
        write_canonical_actual(
            plan.report_root / "data" / "analysis_summary.json",
            plan.report_root / "data" / "m143_fold_summary.csv",
            plan.outputs_root,
            actual,
        )
        archived_comparison = compare_results(
            plan.repo_root / "repro" / "expected_results.json",
            actual,
        )
        write_report(
            run / "reference_difference_report.json",
            {
                "schema_version": 1,
                "kind": "descriptive_retraining_reference_comparison",
                "status": "not_enforced",
                "acceptance_note": (
                    "Stochastic retraining is not required to match archived checkpoint "
                    "metrics, selected epochs or hashes. The nested fixed-checkpoint "
                    "comparison is descriptive and does not affect retraining_status.json."
                ),
                "archived_fixed_checkpoint_comparison": archived_comparison,
            },
        )
        return finish(int(ExitCode.OK), "protocol_complete")
    except Exception as error:  # preserve status at the final boundary
        (run / "internal_error.log").write_text(traceback.format_exc(), encoding="utf-8")
        print(f"Retraining failed: {type(error).__name__}: {error}", file=sys.stderr)
        return finish(int(ExitCode.INTERNAL_ERROR), "internal_error")
