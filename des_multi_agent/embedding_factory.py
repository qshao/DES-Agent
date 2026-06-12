"""Process-global cache for ChemBERTa embedders.

The DES eutectic model and the QSPR melting-point model both embed SMILES with
``DeepChem/ChemBERTa-77M-MTR``. Loading the transformer twice wastes time and
~0.3 GB; this factory returns one shared instance for any identical
(model, pooling, batch_size, max_length, device, cache_dir) request.
"""
from __future__ import annotations

from functools import lru_cache

from .paths import ensure_ml_des_path


def _build_chemberta(model_name, pooling, batch_size, max_length, device_str, cache_dir):
    ensure_ml_des_path()
    import torch
    from src.embeddings.chemberta import ChemBERTaEmbedder

    return ChemBERTaEmbedder(
        model_name=model_name,
        pooling=pooling,
        batch_size=batch_size,
        max_length=max_length,
        device=torch.device(device_str),
        cache_dir=cache_dir,
    )


@lru_cache(maxsize=8)
def _cached_chemberta(model_name, pooling, batch_size, max_length, device_str, cache_dir):
    return _build_chemberta(model_name, pooling, batch_size, max_length, device_str, cache_dir)


def get_chemberta_embedder(
    model_name: str,
    pooling: str = "mean",
    batch_size: int = 64,
    max_length: int = 256,
    device="cpu",
    cache_dir: str | None = None,
):
    """Return a shared ChemBERTa embedder for the given configuration."""
    return _cached_chemberta(
        model_name, pooling, int(batch_size), int(max_length), str(device), cache_dir
    )


def clear_embedder_cache() -> None:
    _cached_chemberta.cache_clear()
