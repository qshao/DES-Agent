from __future__ import annotations

import os
from pathlib import Path

import yaml


_DEFAULT_CONFIG_PATH = Path.home() / ".des-agent" / "config.yaml"
_ENV_VAR = "DES_AGENT_CONFIG"

# Keys recognized in the user config file
KNOWN_KEYS = frozenset({"checkpoint_path", "config_path", "llm_config"})


def get_user_config_path() -> Path:
    env = os.environ.get(_ENV_VAR)
    if env:
        return Path(env)
    return _DEFAULT_CONFIG_PATH


def load_user_config() -> dict:
    path = get_user_config_path()
    if not path.exists():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return {k: v for k, v in data.items() if k in KNOWN_KEYS} if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_user_config(settings: dict) -> None:
    path = get_user_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_user_config()
    existing.update({k: v for k, v in settings.items() if k in KNOWN_KEYS})
    path.write_text(yaml.dump(existing, default_flow_style=False), encoding="utf-8")


def _resolve_config_value_path(value: object, repo_root: Path) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    return candidate


def validate_user_config(path: str | Path | None = None, repo_root: str | Path | None = None) -> list[str]:
    config_path = Path(path) if path is not None else get_user_config_path()
    root = Path(repo_root) if repo_root is not None else Path.cwd()
    if not config_path.exists():
        return []
    warnings: list[str] = []
    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        return [f"[config] invalid YAML in {config_path}: {exc}"]
    except OSError as exc:
        return [f"[config] cannot read {config_path}: {exc}"]
    if not isinstance(data, dict):
        return [f"[config] {config_path} must contain a mapping"]
    unknown_keys = sorted(str(key) for key in data if key not in KNOWN_KEYS)
    if unknown_keys:
        warnings.append(f"[config] unknown key(s) ignored in {config_path}: {', '.join(unknown_keys)}")
    for key in sorted(KNOWN_KEYS & set(data)):
        value_path = _resolve_config_value_path(data.get(key), root)
        if value_path is None:
            warnings.append(f"[config] {key} must be a non-empty path string")
        elif not value_path.exists():
            warnings.append(f"[config] {key} path does not exist: {value_path}")
    return warnings
