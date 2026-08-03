"""Create or verify the private manifest for the Full-training data extension.

The public contract contains only aggregate groups.  The generated manifest is
stored below the ignored ``data/`` tree because it lists the safe relative path,
size and SHA-256 digest of every processed training input.  Raw source data are
neither required nor indexed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
MANIFEST_KEYS = {
    "schema_version",
    "dataset_id",
    "scope",
    "files_total",
    "bytes_total",
    "groups",
    "files",
}
ENTRY_KEYS = {"group", "path", "bytes", "sha256"}
GROUP_SUMMARY_KEYS = {"files", "bytes", "sha256"}


class TrainingDataIntegrityError(ValueError):
    """Raised when the public contract or private data package is invalid."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        raise TrainingDataIntegrityError(f"Cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise TrainingDataIntegrityError(f"{label} must be a JSON object: {path}")
    return value


def _safe_relative(value: Any, field: str, *, allow_glob: bool = False) -> str:
    if not isinstance(value, str) or not value:
        raise TrainingDataIntegrityError(f"{field} must be a non-empty relative path")
    if "\\" in value or "\x00" in value:
        raise TrainingDataIntegrityError(f"{field} must use a safe POSIX relative path")
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or pure.parts[0] in {"", "."}:
        raise TrainingDataIntegrityError(f"{field} is not a safe relative path: {value!r}")
    if pure.as_posix() != value or any(":" in part for part in pure.parts):
        raise TrainingDataIntegrityError(f"{field} is not canonical: {value!r}")
    if not allow_glob and any(character in value for character in "*?[]"):
        raise TrainingDataIntegrityError(f"{field} may not contain glob metacharacters")
    return value


def _validate_contract(contract: dict[str, Any]) -> list[dict[str, Any]]:
    if contract.get("schema_version") != 1:
        raise TrainingDataIntegrityError("training contract schema_version must be 1")
    if contract.get("scope") != "full_training_extension":
        raise TrainingDataIntegrityError("training contract scope must be 'full_training_extension'")
    if not isinstance(contract.get("dataset_id"), str) or not contract["dataset_id"]:
        raise TrainingDataIntegrityError("training contract dataset_id must be non-empty")
    if contract.get("required_root_name") != "data":
        raise TrainingDataIntegrityError("the controlled root name in the contract must be 'data'")
    _safe_relative(contract.get("manifest_path"), "manifest_path")

    groups = contract.get("groups")
    if not isinstance(groups, list) or not groups:
        raise TrainingDataIntegrityError("training contract groups must be a non-empty list")
    names: set[str] = set()
    patterns: set[str] = set()
    count_total = 0
    validated: list[dict[str, Any]] = []
    for index, group in enumerate(groups):
        if not isinstance(group, dict):
            raise TrainingDataIntegrityError(f"groups[{index}] must be an object")
        name = group.get("name")
        if not isinstance(name, str) or not name or name in names:
            raise TrainingDataIntegrityError(f"groups[{index}].name must be unique and non-empty")
        pattern = _safe_relative(group.get("pattern"), f"groups[{index}].pattern", allow_glob=True)
        if pattern in patterns:
            raise TrainingDataIntegrityError(f"duplicate group pattern: {pattern}")
        count = group.get("count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise TrainingDataIntegrityError(f"groups[{index}].count must be a positive integer")
        required_for = group.get("required_for")
        if required_for != ["full"]:
            raise TrainingDataIntegrityError(f"groups[{index}].required_for must be ['full']")
        names.add(name)
        patterns.add(pattern)
        count_total += count
        validated.append(group)

    declared_total = contract.get("required_total_files")
    if isinstance(declared_total, bool) or not isinstance(declared_total, int):
        raise TrainingDataIntegrityError("required_total_files must be an integer")
    if count_total != declared_total:
        raise TrainingDataIntegrityError(
            f"group counts sum to {count_total}, not required_total_files={declared_total}"
        )
    declared_bytes = contract.get("required_total_bytes")
    if (
        isinstance(declared_bytes, bool)
        or not isinstance(declared_bytes, int)
        or declared_bytes < 1
    ):
        raise TrainingDataIntegrityError("required_total_bytes must be a positive integer")

    optional = contract.get("optional_excluded_groups", [])
    if not isinstance(optional, list):
        raise TrainingDataIntegrityError("optional_excluded_groups must be a list")
    for index, group in enumerate(optional):
        if not isinstance(group, dict):
            raise TrainingDataIntegrityError(f"optional_excluded_groups[{index}] must be an object")
        name = group.get("name")
        if not isinstance(name, str) or not name or name in names:
            raise TrainingDataIntegrityError(
                f"optional_excluded_groups[{index}].name must be unique and non-empty"
            )
        pattern = _safe_relative(
            group.get("pattern"),
            f"optional_excluded_groups[{index}].pattern",
            allow_glob=True,
        )
        if pattern in patterns:
            raise TrainingDataIntegrityError(f"optional pattern duplicates a required pattern: {pattern}")
        if group.get("included_in_manifest") is not False:
            raise TrainingDataIntegrityError(
                f"optional_excluded_groups[{index}] must set included_in_manifest to false"
            )
        names.add(name)
        patterns.add(pattern)
    return validated


def _resolve_root(data_root: Path, contract: dict[str, Any]) -> Path:
    root = data_root.resolve()
    if not root.is_dir():
        raise TrainingDataIntegrityError(f"controlled data root does not exist: {root}")
    if root.name != contract["required_root_name"]:
        raise TrainingDataIntegrityError(
            f"controlled data root must be named {contract['required_root_name']!r}, "
            f"found {root.name!r}"
        )
    return root


def _relative_file(root: Path, path: Path, field: str) -> str:
    if path.is_symlink():
        raise TrainingDataIntegrityError(f"symbolic links are not accepted for {field}: {path}")
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise TrainingDataIntegrityError(f"{field} escapes the controlled data root: {path}") from error
    value = relative.as_posix()
    _safe_relative(value, field)
    return value


def _collect_groups(
    root: Path,
    groups: Iterable[dict[str, Any]],
) -> tuple[dict[str, list[tuple[str, Path]]], dict[str, str]]:
    collected: dict[str, list[tuple[str, Path]]] = {}
    owners: dict[str, str] = {}
    for group in groups:
        name = str(group["name"])
        matches: list[tuple[str, Path]] = []
        for path in root.glob(str(group["pattern"])):
            if not path.is_file():
                continue
            relative = _relative_file(root, path, f"group {name}")
            previous = owners.get(relative)
            if previous is not None:
                raise TrainingDataIntegrityError(
                    f"contract patterns overlap at {relative!r}: {previous} and {name}"
                )
            owners[relative] = name
            matches.append((relative, path))
        matches.sort(key=lambda item: item[0])
        expected = int(group["count"])
        if len(matches) != expected:
            raise TrainingDataIntegrityError(
                f"{name}: expected exactly {expected} files, found {len(matches)}"
            )
        collected[name] = matches
    return collected, owners


def _group_digest(entries: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for entry in entries:
        digest.update(str(entry["path"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(entry["bytes"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(entry["sha256"]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def create_training_manifest(data_root: Path, contract_path: Path) -> dict[str, Any]:
    """Hash every required processed input and write a deterministic manifest."""

    contract_file = contract_path.resolve()
    contract = _load_object(contract_file, "training contract")
    groups = _validate_contract(contract)
    root = _resolve_root(data_root, contract)
    collected, _ = _collect_groups(root, groups)

    entries: list[dict[str, Any]] = []
    summaries: dict[str, dict[str, Any]] = {}
    hashed = 0
    for group in groups:
        name = str(group["name"])
        group_entries: list[dict[str, Any]] = []
        group_bytes = 0
        for relative, path in collected[name]:
            size = path.stat().st_size
            entry = {
                "group": name,
                "path": relative,
                "bytes": size,
                "sha256": sha256_file(path),
            }
            group_entries.append(entry)
            group_bytes += size
            hashed += 1
            if hashed % 1000 == 0:
                print(f"Hashed {hashed}/{contract['required_total_files']} files", file=sys.stderr)
        summaries[name] = {
            "files": len(group_entries),
            "bytes": group_bytes,
            "sha256": _group_digest(group_entries),
        }
        entries.extend(group_entries)

    payload = {
        "schema_version": 1,
        "dataset_id": contract["dataset_id"],
        "scope": contract["scope"],
        "files_total": len(entries),
        "bytes_total": sum(int(entry["bytes"]) for entry in entries),
        "groups": summaries,
        "files": entries,
    }
    if payload["bytes_total"] != int(contract["required_total_bytes"]):
        raise TrainingDataIntegrityError(
            "training file byte total differs from the public contract: "
            f"expected {contract['required_total_bytes']}, found {payload['bytes_total']}"
        )
    manifest_relative = _safe_relative(contract["manifest_path"], "manifest_path")
    manifest_path = root / Path(*PurePosixPath(manifest_relative).parts)
    if manifest_path.parent != root:
        raise TrainingDataIntegrityError("training manifest must be written directly below data root")
    _write_json_atomic(manifest_path, payload)
    return {
        "status": "created",
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "files_total": payload["files_total"],
        "bytes_total": payload["bytes_total"],
    }


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise TrainingDataIntegrityError(f"{field} must be a lowercase SHA-256 digest")
    return value


def verify_training_manifest(data_root: Path, contract_path: Path) -> dict[str, Any]:
    """Verify public binding, exact group membership, sizes and file hashes."""

    contract_file = contract_path.resolve()
    contract = _load_object(contract_file, "training contract")
    groups = _validate_contract(contract)
    root = _resolve_root(data_root, contract)
    manifest_relative = _safe_relative(contract["manifest_path"], "manifest_path")
    manifest_path = root / Path(*PurePosixPath(manifest_relative).parts)
    if not manifest_path.is_file():
        raise TrainingDataIntegrityError(f"training manifest is missing: {manifest_path}")
    expected_manifest_hash = _require_sha256(contract.get("manifest_sha256"), "manifest_sha256")
    observed_manifest_hash = sha256_file(manifest_path)
    if observed_manifest_hash != expected_manifest_hash:
        raise TrainingDataIntegrityError(
            "training manifest SHA-256 does not match the public contract: "
            f"expected {expected_manifest_hash}, found {observed_manifest_hash}"
        )

    manifest = _load_object(manifest_path, "training manifest")
    if set(manifest) != MANIFEST_KEYS:
        raise TrainingDataIntegrityError(
            f"training manifest fields differ: {sorted(set(manifest) ^ MANIFEST_KEYS)}"
        )
    if manifest.get("schema_version") != 1:
        raise TrainingDataIntegrityError("training manifest schema_version must be 1")
    if manifest.get("dataset_id") != contract["dataset_id"]:
        raise TrainingDataIntegrityError("training manifest dataset_id differs from the contract")
    if manifest.get("scope") != contract["scope"]:
        raise TrainingDataIntegrityError("training manifest scope differs from the contract")

    expected_names = [str(group["name"]) for group in groups]
    summaries = manifest.get("groups")
    if not isinstance(summaries, dict) or list(summaries) != expected_names:
        observed = list(summaries) if isinstance(summaries, dict) else []
        raise TrainingDataIntegrityError(
            f"training manifest group set/order differs: expected {expected_names}, found {observed}"
        )
    entries = manifest.get("files")
    if not isinstance(entries, list):
        raise TrainingDataIntegrityError("training manifest files must be a list")
    expected_total = int(contract["required_total_files"])
    if len(entries) != expected_total or manifest.get("files_total") != expected_total:
        raise TrainingDataIntegrityError(
            f"training manifest must contain exactly {expected_total} file entries"
        )

    collected, owners = _collect_groups(root, groups)
    observed_paths = set(owners)
    declared_paths: set[str] = set()
    entries_by_group: dict[str, list[dict[str, Any]]] = {name: [] for name in expected_names}
    verified_bytes = 0
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or set(entry) != ENTRY_KEYS:
            raise TrainingDataIntegrityError(f"files[{index}] must contain exactly {sorted(ENTRY_KEYS)}")
        group = entry.get("group")
        if group not in entries_by_group:
            raise TrainingDataIntegrityError(f"files[{index}].group is not declared by the contract")
        relative = _safe_relative(entry.get("path"), f"files[{index}].path")
        if relative in declared_paths:
            raise TrainingDataIntegrityError(f"duplicate training manifest path: {relative}")
        declared_paths.add(relative)
        if owners.get(relative) != group:
            raise TrainingDataIntegrityError(
                f"files[{index}] is assigned to {group!r}, not its exact contract group"
            )
        path = root / Path(*PurePosixPath(relative).parts)
        resolved_relative = _relative_file(root, path, f"files[{index}].path")
        if resolved_relative != relative:
            raise TrainingDataIntegrityError(f"files[{index}].path is not canonical")
        size = entry.get("bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise TrainingDataIntegrityError(f"files[{index}].bytes must be a non-negative integer")
        if path.stat().st_size != size:
            raise TrainingDataIntegrityError(f"size mismatch for manifested training file: {relative}")
        expected_hash = _require_sha256(entry.get("sha256"), f"files[{index}].sha256")
        if sha256_file(path) != expected_hash:
            raise TrainingDataIntegrityError(f"SHA-256 mismatch for manifested training file: {relative}")
        entries_by_group[str(group)].append(entry)
        verified_bytes += size

    if declared_paths != observed_paths:
        missing = len(observed_paths - declared_paths)
        outside = len(declared_paths - observed_paths)
        raise TrainingDataIntegrityError(
            f"manifest/contract file set differs: {missing} unmanifested, {outside} outside contract"
        )

    for group in groups:
        name = str(group["name"])
        summary = summaries[name]
        if not isinstance(summary, dict) or set(summary) != GROUP_SUMMARY_KEYS:
            raise TrainingDataIntegrityError(
                f"groups.{name} must contain exactly {sorted(GROUP_SUMMARY_KEYS)}"
            )
        group_entries = entries_by_group[name]
        expected_count = int(group["count"])
        expected_bytes = sum(int(entry["bytes"]) for entry in group_entries)
        expected_digest = _group_digest(group_entries)
        if summary.get("files") != expected_count or len(group_entries) != expected_count:
            raise TrainingDataIntegrityError(f"group count mismatch for {name}")
        if summary.get("bytes") != expected_bytes:
            raise TrainingDataIntegrityError(f"group byte total mismatch for {name}")
        if summary.get("sha256") != expected_digest:
            raise TrainingDataIntegrityError(f"group digest mismatch for {name}")

    if manifest.get("bytes_total") != verified_bytes:
        raise TrainingDataIntegrityError("training manifest bytes_total is incorrect")
    if verified_bytes != int(contract["required_total_bytes"]):
        raise TrainingDataIntegrityError(
            "verified training byte total differs from the public contract"
        )
    return {
        "status": "pass",
        "manifest": str(manifest_path),
        "manifest_sha256": observed_manifest_hash,
        "files_verified": len(entries),
        "bytes_verified": verified_bytes,
        "groups_verified": expected_names,
    }


def _default_contract() -> Path:
    return Path(__file__).resolve().parents[1] / "repro" / "training_dataset_contract.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=_default_contract())
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify the sealed manifest instead of generating it.",
    )
    args = parser.parse_args(argv)
    try:
        if args.verify:
            result = verify_training_manifest(args.data_root, args.contract)
        else:
            result = create_training_manifest(args.data_root, args.contract)
    except TrainingDataIntegrityError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    # ASCII escaping keeps CLI output portable when the repository path contains
    # characters unsupported by the active Windows console code page.
    print(json.dumps(result, indent=2, ensure_ascii=True))
    if not args.verify:
        print(
            "Custodian action: copy manifest_sha256 into "
            "repro/training_dataset_contract.json before release.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
