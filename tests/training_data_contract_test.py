from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.create_training_dataset_manifest import (
    TrainingDataIntegrityError,
    create_training_manifest,
    sha256_file,
    verify_training_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_CONTRACT = ROOT / "repro" / "training_dataset_contract.json"


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _small_contract(tmp_path: Path) -> tuple[Path, Path]:
    data = tmp_path / "data"
    strict = data / "windows" / "strict" / "train"
    enhanced = data / "windows" / "enhanced" / "val"
    optional = data / "windows" / "enhanced" / "test"
    strict.mkdir(parents=True)
    enhanced.mkdir(parents=True)
    optional.mkdir(parents=True)
    (strict / "window-a.npz").write_bytes(b"strict-a")
    (strict / "window-b.npz").write_bytes(b"strict-b")
    (enhanced / "window-c.npz").write_bytes(b"enhanced-c")
    (optional / "historical-only.npz").write_bytes(b"not-a-training-input")
    contract = tmp_path / "training-contract.json"
    _write_json(
        contract,
        {
            "schema_version": 1,
            "dataset_id": "test-training-extension-v1",
            "scope": "full_training_extension",
            "required_root_name": "data",
            "manifest_path": "training-manifest.json",
            "manifest_sha256": "TO_BE_SEALED_BY_DATA_CUSTODIAN",
            "required_total_files": 3,
            "required_total_bytes": 26,
            "groups": [
                {
                    "name": "strict_train",
                    "pattern": "windows/strict/train/*.npz",
                    "count": 2,
                    "required_for": ["full"],
                },
                {
                    "name": "enhanced_val",
                    "pattern": "windows/enhanced/val/*.npz",
                    "count": 1,
                    "required_for": ["full"],
                },
            ],
            "optional_excluded_groups": [
                {
                    "name": "enhanced_test",
                    "pattern": "windows/enhanced/test/*.npz",
                    "expected_count_if_present": 1,
                    "included_in_manifest": False,
                }
            ],
        },
    )
    return data, contract


def _bind_generated_manifest(data: Path, contract_path: Path) -> str:
    created = create_training_manifest(data, contract_path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["manifest_sha256"] = created["manifest_sha256"]
    _write_json(contract_path, contract)
    return str(created["manifest_sha256"])


def test_public_contract_is_additive_and_omits_individual_names() -> None:
    contract = json.loads(PUBLIC_CONTRACT.read_text(encoding="utf-8"))
    assert len(contract["manifest_sha256"]) == 64
    int(contract["manifest_sha256"], 16)
    assert contract["required_total_files"] == 105_640
    assert contract["required_total_bytes"] == 13_744_115_616
    assert sum(group["count"] for group in contract["groups"]) == 105_640
    assert contract["composes_with"]["required_files"] == 23_379
    assert contract["composes_with"]["combined_full_files"] == 129_019
    assert {group["name"] for group in contract["groups"]} == {
        "strict_esc50_train",
        "strict_esc50_val",
        "enhanced_esc50_train",
        "enhanced_esc50_val",
    }
    optional = contract["optional_excluded_groups"]
    assert optional == [
        {
            "name": "enhanced_esc50_test",
            "pattern": "windows/bsslab_esc50_enhanced_v2/mixed/test/*.npz",
            "expected_count_if_present": 5410,
            "included_in_manifest": False,
            "reason": "Historical diagnostic split; not read by the minimum Full training and validation protocol.",
        }
    ]
    public_text = PUBLIC_CONTRACT.read_text(encoding="utf-8")
    assert "M001_" not in public_text
    assert '"files"' not in public_text


def test_create_and_verify_private_per_file_manifest(tmp_path: Path) -> None:
    data, contract = _small_contract(tmp_path)
    bound_hash = _bind_generated_manifest(data, contract)

    report = verify_training_manifest(data, contract)
    assert report["status"] == "pass"
    assert report["manifest_sha256"] == bound_hash
    assert report["files_verified"] == 3
    manifest = json.loads((data / "training-manifest.json").read_text(encoding="utf-8"))
    assert [entry["path"] for entry in manifest["files"]] == [
        "windows/strict/train/window-a.npz",
        "windows/strict/train/window-b.npz",
        "windows/enhanced/val/window-c.npz",
    ]
    assert all(set(entry) == {"group", "path", "bytes", "sha256"} for entry in manifest["files"])
    assert "historical-only.npz" not in json.dumps(manifest)


def test_verifier_rejects_extra_required_group_member(tmp_path: Path) -> None:
    data, contract = _small_contract(tmp_path)
    _bind_generated_manifest(data, contract)
    (data / "windows" / "strict" / "train" / "extra.npz").write_bytes(b"extra")

    with pytest.raises(TrainingDataIntegrityError, match="expected exactly 2 files, found 3"):
        verify_training_manifest(data, contract)


def test_verifier_rejects_changed_training_content(tmp_path: Path) -> None:
    data, contract = _small_contract(tmp_path)
    _bind_generated_manifest(data, contract)
    (data / "windows" / "strict" / "train" / "window-a.npz").write_bytes(b"changed!")

    with pytest.raises(TrainingDataIntegrityError, match="SHA-256 mismatch"):
        verify_training_manifest(data, contract)


def test_verifier_rejects_unsafe_manifest_path_even_when_rebound(tmp_path: Path) -> None:
    data, contract = _small_contract(tmp_path)
    _bind_generated_manifest(data, contract)
    manifest_path = data / "training-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][0]["path"] = "../outside.npz"
    _write_json(manifest_path, manifest)
    contract_value = json.loads(contract.read_text(encoding="utf-8"))
    contract_value["manifest_sha256"] = sha256_file(manifest_path)
    _write_json(contract, contract_value)

    with pytest.raises(TrainingDataIntegrityError, match="not a safe relative path"):
        verify_training_manifest(data, contract)


def test_manifest_generation_is_deterministic_and_ignores_optional_test(tmp_path: Path) -> None:
    data, contract = _small_contract(tmp_path)
    first = create_training_manifest(data, contract)["manifest_sha256"]
    (data / "windows" / "enhanced" / "test" / "another.npz").write_bytes(b"also excluded")
    second = create_training_manifest(data, contract)["manifest_sha256"]

    assert first == second
    assert first == hashlib.sha256((data / "training-manifest.json").read_bytes()).hexdigest()
