from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch

from .shared_paths import ML_DES_ROOT, PROJECT_ROOT, resolve_path as _resolve_path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device_str: str) -> torch.device:
    if device_str.lower().startswith("cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_dir(path: str | os.PathLike[str]) -> None:
    os.makedirs(path, exist_ok=True)


def resolve_path(path: str | os.PathLike[str], *, base_dir: str | os.PathLike[str] | None = None) -> Path:
    return _resolve_path(path, base_dir=base_dir)
