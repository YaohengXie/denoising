"""Create the deterministic manifest shipped with the authorised data package."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def create_manifest(data_root: Path, contract_path: Path) -> Path:
    root = data_root.resolve()
    contract = _load_object(contract_path.resolve())
    if contract.get("schema_version") != 1:
        raise ValueError("dataset contract schema_version must be 1")
    if root.name != str(contract.get("required_root_name", "")):
        raise ValueError("The controlled data root directory must be named 'data'.")

    indexed: dict[str, Path] = {}
    observed_groups: dict[str, int] = {}
    for group in contract["groups"]:
        name = str(group["name"])
        paths = sorted(path for path in root.glob(str(group["pattern"])) if path.is_file())
        observed_groups[name] = len(paths)
        expected_count = int(group["count"])
        if len(paths) != expected_count:
            raise ValueError(f"{name}: expected {expected_count} files, found {len(paths)}")
        for path in paths:
            relative = path.relative_to(root).as_posix()
            if relative in indexed:
                raise ValueError(f"Dataset contract patterns overlap at {relative}")
            indexed[relative] = path

    expected_total = int(contract["required_total_files"])
    if len(indexed) != expected_total:
        raise ValueError(f"Expected {expected_total} unique files, found {len(indexed)}")

    files: list[dict[str, Any]] = []
    bytes_total = 0
    for index, (relative, path) in enumerate(sorted(indexed.items()), start=1):
        size = path.stat().st_size
        bytes_total += size
        files.append({"path": relative, "bytes": size, "sha256": _sha256(path)})
        if index % 1000 == 0:
            print(f"Hashed {index}/{len(indexed)} files")

    payload = {
        "schema_version": 1,
        "dataset_id": contract["dataset_id"],
        "files_total": len(files),
        "bytes_total": bytes_total,
        "groups": observed_groups,
        "files": files,
    }
    output = root / str(contract["manifest_path"])
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(f"Manifest written: {output.name}")
    print(f"SHA-256: {_sha256(output)}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "repro" / "dataset_contract.json",
    )
    args = parser.parse_args()
    create_manifest(args.data_root, args.contract)


if __name__ == "__main__":
    main()
