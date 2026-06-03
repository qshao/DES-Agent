from __future__ import annotations

from pathlib import Path
import sys


def ensure_ml_des_path() -> Path:
    ml_des_root = Path(__file__).resolve().parents[1] / "ml_des_mp"
    ml_des_str = str(ml_des_root)
    if ml_des_str not in sys.path:
        sys.path.insert(0, ml_des_str)
    return ml_des_root


def resolve_existing_path(path: str | Path, *, base_dir: str | Path | None = None) -> Path:
    ensure_ml_des_path()
    from src.shared_paths import resolve_existing_path as _resolve_existing_path

    return _resolve_existing_path(path, base_dir=base_dir)
