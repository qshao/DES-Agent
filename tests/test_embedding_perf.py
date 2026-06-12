"""Shared ChemBERTa embedder factory (P2) and bounded embedding cache (P3)."""
from __future__ import annotations

from types import SimpleNamespace

import torch

import des_multi_agent.embedding_factory as ef
from des_multi_agent import prediction as P


def test_get_chemberta_embedder_shares_instance_by_params(monkeypatch):
    ef._cached_chemberta.cache_clear()
    calls = []

    def fake_build(*args):
        calls.append(args)
        return object()

    monkeypatch.setattr(ef, "_build_chemberta", fake_build)
    e1 = ef.get_chemberta_embedder("m", device="cpu", max_length=128)
    e2 = ef.get_chemberta_embedder("m", device="cpu", max_length=128)
    e3 = ef.get_chemberta_embedder("m", device="cuda", max_length=128)
    e4 = ef.get_chemberta_embedder("m", device="cpu", max_length=256)
    assert e1 is e2            # identical params -> shared instance
    assert e3 is not e1        # different device -> distinct
    assert e4 is not e1        # different max_length -> distinct
    assert len(calls) == 3


def test_embed_cache_is_bounded(monkeypatch):
    P.clear_prediction_caches()
    monkeypatch.setattr(P, "_EMBED_CACHE_MAXSIZE", 3)
    bundle = SimpleNamespace(embedder=SimpleNamespace(embed=lambda s: [[0.0, 1.0]]))
    for i in range(10):
        P._embed_cached(bundle, f"C{i}", torch.device("cpu"), "scope")
    assert len(P._EMBED_CACHE) <= 3
