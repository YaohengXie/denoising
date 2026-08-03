from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ecg_pcg_denoise.retrain.runner import FOLDS, build_plan, run_retraining


def test_adapter_plan_has_expected_training_validation_sequence(tmp_path: Path) -> None:
    data = tmp_path / "data"
    run = tmp_path / "adapter-run"
    plan = build_plan("adapter", data, run, repo_root=ROOT, python_executable=sys.executable)

    assert plan.scope == "adapter"
    assert len(plan.commands) == 30
    labels = [command.label for command in plan.commands]
    assert labels[0] == "evaluate_m7_strict"
    assert labels[-1] == "report_strict_v2"
    for fold in FOLDS:
        assert f"train_m143_{fold}" in labels
        assert f"select_m143_{fold}" in labels
        assert f"calibrate_m143_{fold}" in labels
        assert f"test_m143_{fold}" in labels

    fold_config = json.loads(
        (run / "runtime_configs" / "m143_M001_S01.yaml").read_text(encoding="utf-8")
    )
    assert fold_config["paths"]["base_m7_checkpoint"] == str(
        ROOT / "checkpoints" / "m7_v2" / "best.pt"
    )
    assert fold_config["paths"]["data_config"].startswith(str(run))


def test_full_plan_rebinds_every_m7_dependency_to_current_run(tmp_path: Path) -> None:
    data = tmp_path / "data"
    run = tmp_path / "full-run"
    plan = build_plan("full", data, run, repo_root=ROOT, python_executable=sys.executable)

    assert plan.scope == "full"
    assert len(plan.commands) == 35
    assert [command.label for command in plan.commands[:5]] == [
        "train_m5",
        "train_m6",
        "train_distill",
        "train_robust",
        "train_sqi",
    ]
    runtime = run / "runtime_configs"
    distill = json.loads((runtime / "distill.yaml").read_text(encoding="utf-8"))
    robust = json.loads((runtime / "robust.yaml").read_text(encoding="utf-8"))
    sqi = json.loads((runtime / "sqi.yaml").read_text(encoding="utf-8"))
    assert str(run / "outputs" / "bsslab_esc50_v2_m6_multitask") in distill["training"]["init_checkpoint"]
    assert str(run / "outputs" / "bsslab_esc50_v2_m5") in distill["training"]["teacher_checkpoint"]
    assert str(run / "outputs" / "bsslab_esc50_v2_m7_distill") in robust["training"]["init_checkpoint"]
    assert str(run / "outputs" / "bsslab_esc50_enhanced_v2_m7_robust") in sqi["training"]["init_checkpoint"]
    assert plan.m7_checkpoint == (
        run
        / "outputs"
        / "bsslab_esc50_enhanced_v2_m7_robust_sqi"
        / "checkpoints"
        / "best.pt"
    )


def test_adapter_dry_run_writes_plan_without_data(tmp_path: Path) -> None:
    data = tmp_path / "data"
    run = tmp_path / "dry-run"
    exit_code, plan = run_retraining(
        "adapter",
        data,
        run,
        repo_root=ROOT,
        python_executable=sys.executable,
        dry_run=True,
    )

    assert exit_code == 0
    assert plan.scope == "adapter"
    status = json.loads((run / "retraining_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "pass"
    assert status["stage"] == "dry_run_plan"
    assert (run / "retraining_plan.json").is_file()
