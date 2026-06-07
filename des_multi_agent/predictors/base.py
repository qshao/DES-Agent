from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
import json
import pickle

try:
    import torch
except Exception:  # pragma: no cover - torch is available in the project env
    torch = None


def default_warning_tuple(warnings: list[str] | tuple[str, ...] | None = None) -> tuple[str, ...]:
    if warnings is None:
        return ()
    return tuple(str(item) for item in warnings)


def load_artifact_payload(path: str | Path) -> Any:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {'.json'}:
        return json.loads(path.read_text(encoding='utf-8'))
    if suffix in {'.yaml', '.yml'}:
        import yaml
        return yaml.safe_load(path.read_text(encoding='utf-8'))
    if suffix in {'.pkl', '.pickle'}:
        with path.open('rb') as fh:
            return pickle.load(fh)
    if suffix in {'.pt', '.ckpt'} and torch is not None:
        return torch.load(path, map_location='cpu')
    return json.loads(path.read_text(encoding='utf-8'))


def parse_local_model(payload: Any) -> dict[str, Any]:
    if isinstance(payload, Mapping):
        return dict(payload)
    if hasattr(payload, 'predict'):
        return {'kind': 'callable', 'model': payload}
    raise TypeError(f'Unsupported local model payload: {type(payload)!r}')


def linear_predict(spec: Mapping[str, Any], features: Mapping[str, float]) -> float:
    bias = float(spec.get('bias', 0.0))
    coefficients = spec.get('coefficients', {})
    if not isinstance(coefficients, Mapping):
        raise TypeError('model coefficients must be a mapping')
    total = bias
    for name, weight in coefficients.items():
        total += float(weight) * float(features.get(name, 0.0))
    return float(total)


def clamp(value: float, min_value: float | None = None, max_value: float | None = None) -> float:
    if min_value is not None:
        value = max(min_value, value)
    if max_value is not None:
        value = min(max_value, value)
    return float(value)
