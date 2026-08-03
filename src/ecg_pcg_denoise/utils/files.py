from __future__ import annotations

from pathlib import Path


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def list_files(root: str | Path, suffixes: tuple[str, ...]) -> list[Path]:
    root_path = Path(root)
    if not root_path.exists():
        return []
    suffixes_lower = tuple(s.lower() for s in suffixes)
    return sorted(p for p in root_path.rglob("*") if p.suffix.lower() in suffixes_lower)


def clean_id(value: object) -> str:
    text = str(value).strip()
    keep = [c if c.isalnum() or c in ("-", "_") else "_" for c in text]
    return "".join(keep).strip("_")
