from __future__ import annotations

from pathlib import Path
import os


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ML_DES_ROOT = PROJECT_ROOT / "ml_des_mp"


def resolve_path(path: str | os.PathLike[str], *, base_dir: str | os.PathLike[str] | None = None) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate

    roots = []
    if base_dir is not None:
        roots.append(Path(base_dir))
    roots.extend([Path.cwd(), PROJECT_ROOT, ML_DES_ROOT])

    seen = set()
    for root in roots:
        root = root.resolve()
        if root in seen:
            continue
        seen.add(root)
        resolved = (root / candidate).resolve()
        if resolved.exists():
            return resolved

    return (Path.cwd() / candidate).resolve()


def resolve_existing_path(path: str | os.PathLike[str], *, base_dir: str | os.PathLike[str] | None = None) -> Path:
    resolved = resolve_path(path, base_dir=base_dir)
    if not resolved.exists():
        raise FileNotFoundError(f"Path does not exist: {resolved}")
    return resolved
