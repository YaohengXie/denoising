"""Integrity checks for public weights and the separately supplied data package."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path, PurePosixPath
from typing import Any

from ecg_pcg_denoise.repro.common import Issue, load_json_object, make_report
from ecg_pcg_denoise.repro.exit_codes import ExitCode


CHECKPOINT_IDS = {"m7_v2_base", "M001_S01", "M001_S02", "M001_S03", "M001_S04"}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_file(root: Path, relative: Any, field: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{field} must be a non-empty relative path")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or pure.parts[0] in {"", "."}:
        raise ValueError(f"{field} is not a safe relative path: {relative!r}")
    candidate = (root / Path(*pure.parts)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{field} escapes its declared root: {relative!r}") from error
    return candidate


def verify_checkpoints(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    ledger_path = root / "checkpoints" / "checksums.json"
    issues: list[Issue] = []
    details: dict[str, Any] = {
        "ledger": str(ledger_path),
        "files_verified": 0,
        "metadata_files_verified": 0,
    }
    if not ledger_path.is_file():
        return make_report(
            "checkpoint_integrity",
            [Issue("missing_artifact", "checkpoint_ledger_missing", "Checkpoint ledger is missing.", str(ledger_path))],
            ExitCode.MISSING_ARTIFACT,
            details,
        )
    try:
        ledger = load_json_object(ledger_path)
        files = ledger.get("files")
        if ledger.get("schema_version") != 1 or not isinstance(files, dict):
            raise ValueError("checkpoint ledger must use schema_version 1 and contain a files object")
        if set(files) != CHECKPOINT_IDS:
            raise ValueError(f"checkpoint ledger IDs differ: {sorted(set(files) ^ CHECKPOINT_IDS)}")
        for checkpoint_id, item in files.items():
            if not isinstance(item, dict):
                raise ValueError(f"files.{checkpoint_id} must be an object")
            path = _safe_file(root, item.get("path"), f"files.{checkpoint_id}.path")
            try:
                path.relative_to(root / "checkpoints")
            except ValueError as error:
                raise ValueError(f"{checkpoint_id} is outside checkpoints/") from error
            if not path.is_file():
                issues.append(Issue("missing_artifact", "checkpoint_missing", "Checkpoint file is missing.", str(path)))
                continue
            observed_size = path.stat().st_size
            observed_hash = sha256_file(path)
            if observed_size != int(item.get("bytes", -1)):
                issues.append(Issue("integrity_mismatch", "checkpoint_size_mismatch", f"Expected {item.get('bytes')} bytes, found {observed_size}.", str(path)))
            if observed_hash != str(item.get("sha256", "")).lower():
                issues.append(Issue("integrity_mismatch", "checkpoint_hash_mismatch", "Checkpoint SHA-256 does not match the public ledger.", str(path)))
            details["files_verified"] += 1
            if checkpoint_id != "m7_v2_base":
                selection = item.get("selection_summary")
                if not isinstance(selection, dict):
                    raise ValueError(f"files.{checkpoint_id}.selection_summary must be an object")
                selection_path = _safe_file(
                    root,
                    selection.get("path"),
                    f"files.{checkpoint_id}.selection_summary.path",
                )
                if selection_path.parent != path.parent:
                    raise ValueError(f"{checkpoint_id} selection summary is outside its fold directory")
                if not selection_path.is_file():
                    issues.append(Issue("missing_artifact", "selection_summary_missing", "Checkpoint selection summary is missing.", str(selection_path)))
                    continue
                if selection_path.stat().st_size != int(selection.get("bytes", -1)):
                    issues.append(Issue("integrity_mismatch", "selection_summary_size_mismatch", "Selection-summary size does not match the ledger.", str(selection_path)))
                if sha256_file(selection_path) != str(selection.get("sha256", "")).lower():
                    issues.append(Issue("integrity_mismatch", "selection_summary_hash_mismatch", "Selection-summary SHA-256 does not match the ledger.", str(selection_path)))
                summary = load_json_object(selection_path)
                epochs = {
                    int(item.get("epoch", -1)),
                    int(selection.get("selected_epoch", -2)),
                    int(summary.get("selected_epoch", -3)),
                }
                if len(epochs) != 1:
                    issues.append(Issue("binding_mismatch", "selection_epoch_mismatch", "Adapter and selection-summary epochs are not identical.", str(selection_path)))
                details["metadata_files_verified"] += 1
    except (OSError, TypeError, ValueError) as error:
        issues.append(Issue("invalid_spec", "invalid_checkpoint_ledger", str(error), str(ledger_path)))

    if any(issue.category == "invalid_spec" for issue in issues):
        code = ExitCode.INVALID_SPEC
    elif any(issue.category == "missing_artifact" for issue in issues):
        code = ExitCode.MISSING_ARTIFACT
    elif any(issue.category == "binding_mismatch" for issue in issues):
        code = ExitCode.BINDING_MISMATCH
    elif issues:
        code = ExitCode.INTEGRITY_MISMATCH
    else:
        code = ExitCode.OK
    return make_report("checkpoint_integrity", issues, code, details)


def _verify_frozen_csv_paths(
    data_root: Path,
    declared_files: set[str],
) -> tuple[list[Issue], int]:
    issues: list[Issue] = []
    rows_checked = 0
    paths = sorted((data_root / "manifests" / "bsslab_m14_imu").glob("fold_*/fixed_*_pairs.csv"))
    for csv_path in paths:
        with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if not {"clean_path", "imu_path"}.issubset(reader.fieldnames or []):
                issues.append(Issue("invalid_spec", "pair_csv_columns_missing", "Frozen-pair CSV lacks clean_path or imu_path.", str(csv_path)))
                continue
            for row_number, row in enumerate(reader, start=2):
                rows_checked += 1
                for field in ("clean_path", "imu_path"):
                    value = str(row[field])
                    pure = PurePosixPath(value)
                    valid_prefix = len(pure.parts) >= 3 and pure.parts[:2] == ("data", "windows")
                    if pure.is_absolute() or ".." in pure.parts or not valid_prefix:
                        issues.append(Issue("binding_mismatch", "unsafe_pair_path", f"{field} must be a relative data/windows/... path (row {row_number}).", str(csv_path)))
                        continue
                    relative_to_data = PurePosixPath(*pure.parts[1:]).as_posix()
                    if relative_to_data not in declared_files:
                        issues.append(Issue("binding_mismatch", "pair_path_not_manifested", f"{field} is absent from the sealed dataset manifest (row {row_number}): {value}", str(csv_path)))
                if len(issues) >= 100:
                    return issues, rows_checked
    return issues, rows_checked


def verify_dataset(
    data_root: str | Path,
    contract_path: str | Path,
) -> dict[str, Any]:
    root = Path(data_root).resolve()
    contract_file = Path(contract_path).resolve()
    issues: list[Issue] = []
    details: dict[str, Any] = {"data_root": str(root), "contract": str(contract_file), "files_verified": 0}
    if not root.is_dir():
        return make_report(
            "dataset_integrity",
            [Issue("missing_artifact", "data_root_missing", "Controlled data root does not exist.", str(root))],
            ExitCode.MISSING_ARTIFACT,
            details,
        )
    try:
        contract = load_json_object(contract_file)
        required_root_name = str(contract.get("required_root_name", ""))
        if root.name != required_root_name:
            raise ValueError(
                f"controlled data root must be named {required_root_name!r}, found {root.name!r}"
            )
        manifest = _safe_file(root, contract.get("manifest_path"), "manifest_path")
        if not manifest.is_file():
            raise FileNotFoundError(manifest)
        manifest_hash = sha256_file(manifest)
        details["manifest"] = str(manifest)
        details["manifest_sha256"] = manifest_hash
        if manifest_hash != str(contract.get("manifest_sha256", "")).lower():
            issues.append(Issue("integrity_mismatch", "dataset_manifest_hash_mismatch", "The supplied dataset manifest is not the frozen public manifest.", str(manifest)))
            return make_report("dataset_integrity", issues, ExitCode.INTEGRITY_MISMATCH, details)

        payload = load_json_object(manifest)
        if payload.get("schema_version") != 1 or payload.get("dataset_id") != contract.get("dataset_id"):
            raise ValueError("dataset manifest schema or dataset_id does not match the contract")
        entries = payload.get("files")
        if not isinstance(entries, list):
            raise ValueError("dataset manifest files must be a list")
        expected_total = int(contract["required_total_files"])
        if len(entries) != expected_total or int(payload.get("files_total", -1)) != expected_total:
            raise ValueError(f"dataset manifest must contain exactly {expected_total} files")
        expected_groups = {str(group["name"]): int(group["count"]) for group in contract["groups"]}
        if payload.get("groups") != expected_groups:
            raise ValueError("dataset manifest group counts do not match the public contract")

        declared: set[str] = set()
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError("each dataset manifest entry must be an object")
            relative = str(entry.get("path", ""))
            if relative in declared:
                raise ValueError(f"duplicate dataset manifest path: {relative}")
            declared.add(relative)
            path = _safe_file(root, relative, "files[].path")
            if not path.is_file():
                issues.append(Issue("missing_artifact", "dataset_file_missing", "A sealed dataset file is missing.", relative))
                if len(issues) >= 100:
                    break
                continue
            if path.stat().st_size != int(entry.get("bytes", -1)):
                issues.append(Issue("integrity_mismatch", "dataset_size_mismatch", "Dataset file size differs from the manifest.", relative))
            elif sha256_file(path) != str(entry.get("sha256", "")).lower():
                issues.append(Issue("integrity_mismatch", "dataset_hash_mismatch", "Dataset file SHA-256 differs from the manifest.", relative))
            details["files_verified"] += 1

        matched_by_contract: set[str] = set()
        for group in contract["groups"]:
            name = str(group["name"])
            matches = {
                path.relative_to(root).as_posix()
                for path in root.glob(str(group["pattern"]))
                if path.is_file()
            }
            expected_count = int(group["count"])
            if len(matches) != expected_count:
                issues.append(
                    Issue(
                        "integrity_mismatch",
                        "dataset_group_count_mismatch",
                        f"{name} expected {expected_count} files, found {len(matches)}.",
                    )
                )
            overlap = matched_by_contract & matches
            if overlap:
                issues.append(
                    Issue(
                        "invalid_spec",
                        "dataset_contract_patterns_overlap",
                        f"Contract group {name} overlaps another group at {sorted(overlap)[:3]}.",
                    )
                )
            matched_by_contract.update(matches)

        undeclared = matched_by_contract - declared
        outside_contract = declared - matched_by_contract
        if undeclared:
            issues.append(
                Issue(
                    "integrity_mismatch",
                    "unmanifested_dataset_files",
                    f"Contract patterns found unmanifested files: {sorted(undeclared)[:5]}",
                )
            )
        if outside_contract:
            issues.append(
                Issue(
                    "invalid_spec",
                    "manifest_files_outside_contract",
                    f"Manifest contains files outside contract groups: {sorted(outside_contract)[:5]}",
                )
            )

        csv_issues, rows_checked = _verify_frozen_csv_paths(root, declared)
        issues.extend(csv_issues)
        details["frozen_pair_rows_checked"] = rows_checked
    except FileNotFoundError as error:
        issues.append(Issue("missing_artifact", "dataset_manifest_missing", "The controlled package lacks its dataset manifest.", str(error)))
    except (OSError, TypeError, ValueError) as error:
        issues.append(Issue("invalid_spec", "invalid_dataset_manifest", str(error), str(contract_file)))

    if any(issue.category == "invalid_spec" for issue in issues):
        code = ExitCode.INVALID_SPEC
    elif any(issue.category == "missing_artifact" for issue in issues):
        code = ExitCode.MISSING_ARTIFACT
    elif any(issue.category == "binding_mismatch" for issue in issues):
        code = ExitCode.BINDING_MISMATCH
    elif issues:
        code = ExitCode.INTEGRITY_MISMATCH
    else:
        code = ExitCode.OK
    return make_report("dataset_integrity", issues, code, details)
