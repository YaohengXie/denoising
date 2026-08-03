from __future__ import annotations

import json
from pathlib import Path

from ecg_pcg_denoise.repro.integrity import sha256_file, verify_dataset


def _write_contract_and_manifest(tmp_path: Path) -> tuple[Path, Path]:
    data = tmp_path / "data"
    payload_file = data / "windows" / "sample.txt"
    payload_file.parent.mkdir(parents=True)
    payload_file.write_text("frozen\n", encoding="utf-8")
    manifest = data / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_id": "test-data-v1",
                "files_total": 1,
                "bytes_total": payload_file.stat().st_size,
                "groups": {"windows": 1},
                "files": [
                    {
                        "path": "windows/sample.txt",
                        "bytes": payload_file.stat().st_size,
                        "sha256": sha256_file(payload_file),
                    }
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    contract = tmp_path / "contract.json"
    contract.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_id": "test-data-v1",
                "required_root_name": "data",
                "manifest_path": "manifest.json",
                "manifest_sha256": sha256_file(manifest),
                "required_total_files": 1,
                "groups": [
                    {"name": "windows", "pattern": "windows/*.txt", "count": 1}
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return data, contract


def test_dataset_verifier_rejects_unmanifested_matching_file(tmp_path: Path) -> None:
    data, contract = _write_contract_and_manifest(tmp_path)
    assert verify_dataset(data, contract)["status"] == "pass"
    (data / "windows" / "extra.txt").write_text("extra\n", encoding="utf-8")
    report = verify_dataset(data, contract)
    assert report["status"] == "fail"
    codes = {issue["code"] for issue in report["issues"]}
    assert "dataset_group_count_mismatch" in codes
    assert "unmanifested_dataset_files" in codes


def test_dataset_verifier_requires_data_root_name(tmp_path: Path) -> None:
    data, contract = _write_contract_and_manifest(tmp_path)
    renamed = tmp_path / "wrong-name"
    data.rename(renamed)
    report = verify_dataset(renamed, contract)
    assert report["exit_code"] == 2
    assert report["issues"][0]["code"] == "invalid_dataset_manifest"
