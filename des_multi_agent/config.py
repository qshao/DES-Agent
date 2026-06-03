from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent

DEFAULT_RATIO_GRID: Tuple[float, ...] = tuple(round(0.1 + 0.1 * i, 3) for i in range(9))
DEFAULT_ABSOLUTE_TM_MAX_K = 260.0
DEFAULT_RELATIVE_DROP_MIN = 0.10
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "ml_des_mp" / "config.yaml"


@dataclass(frozen=True)
class RuntimeConfig:
    checkpoint_path: str | Path
    config_path: str | Path = DEFAULT_CONFIG_PATH
    absolute_tm_max_k: float = DEFAULT_ABSOLUTE_TM_MAX_K
    relative_drop_min: float = DEFAULT_RELATIVE_DROP_MIN
