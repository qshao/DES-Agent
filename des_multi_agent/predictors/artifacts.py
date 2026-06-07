from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ..config import PROJECT_ROOT

MANIFEST_PATH = PROJECT_ROOT / 'des_multi_agent' / 'artifacts' / 'manifest.yaml'
DEFAULT_ARTIFACT_ROOT = PROJECT_ROOT / 'artifacts'


def default_artifact_root() -> Path:
    return DEFAULT_ARTIFACT_ROOT


def load_manifest(path: str | Path = MANIFEST_PATH) -> dict[str, Any]:
    with Path(path).open('r', encoding='utf-8') as fh:
        return yaml.safe_load(fh) or {}


def require_artifact(base_dir: str | Path, relative_path: str) -> Path:
    candidate = Path(base_dir) / relative_path
    if not candidate.exists():
        raise FileNotFoundError(f'Missing local artifact: {candidate}')
    return candidate


def resolve_artifact(path: str | Path | None, manifest_key: str, *, manifest_path: str | Path = MANIFEST_PATH) -> Path:
    if path is not None:
        candidate = Path(path)
        if not candidate.exists():
            raise FileNotFoundError(f'Missing explicit local artifact: {candidate}')
        return candidate
    manifest = load_manifest(manifest_path)
    relative_path = manifest['workflows'][manifest_key]['artifacts']['model']
    return require_artifact(default_artifact_root(), relative_path)
