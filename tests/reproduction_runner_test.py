from __future__ import annotations

import json
from pathlib import Path

from ecg_pcg_denoise.repro.golden import compare_results
from ecg_pcg_denoise.repro.runner import ReproductionSpecError, run_reproduction


ROOT = Path(__file__).resolve().parents[1]


def test_dry_run_builds_complete_fixed_checkpoint_plan(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    exit_code, plan = run_reproduction(
        tmp_path / "data",
        run_root,
        repo_root=ROOT,
        dry_run=True,
    )
    assert exit_code == 0
    assert len(plan.commands) == 22
    rendered = [" ".join(command) for command in plan.commands]
    assert sum("eval_denoise" in command for command in rendered) == 1
    assert sum("eval_m143_v2" in command for command in rendered) == 20
    assert sum("report_strict_v2" in command for command in rendered) == 1
    status = json.loads((run_root / "reproduction_status.json").read_text())
    assert status["status"] == "pass"
    assert status["stage"] == "dry_run_plan"


def test_public_reference_contains_68_self_consistent_checks(tmp_path: Path) -> None:
    expected = ROOT / "repro" / "expected_results.json"
    report = compare_results(expected, expected)
    assert report["status"] == "pass", report
    assert report["details"]["checks_total"] == 68
    assert report["details"]["checks_passed"] == 68


def test_run_root_inside_data_is_rejected_before_write(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    unsafe_output = data / "must-not-be-created"
    try:
        run_reproduction(data, unsafe_output, repo_root=ROOT, dry_run=True)
    except ReproductionSpecError as error:
        assert "outside" in str(error)
    else:  # pragma: no cover
        raise AssertionError("An output directory inside controlled data was accepted.")
    assert not unsafe_output.exists()
