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
