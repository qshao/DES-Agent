"""Controls and cache management for the layered melting-point resolver:
QSPR device selection and resolver cache reset."""
from __future__ import annotations

import des_multi_agent.property_resolution as pr
import des_multi_agent.predictors.melting_point as mp


def test_resolve_mp_device_defaults_to_cpu(monkeypatch):
    monkeypatch.delenv("DES_MP_DEVICE", raising=False)
    assert pr._resolve_mp_device() == "cpu"


def test_resolve_mp_device_honors_env(monkeypatch):
    monkeypatch.setenv("DES_MP_DEVICE", "cuda")
    assert pr._resolve_mp_device() == "cuda"


def test_qspr_model_passes_resolved_device(monkeypatch):
    seen = {}

    def fake_load(path=mp.QSPR_MODEL_PATH, device="cpu"):
        seen["device"] = device
        return object()

    monkeypatch.delenv("DES_DISABLE_QSPR", raising=False)
    monkeypatch.setenv("DES_MP_DEVICE", "cuda")
    monkeypatch.setattr(mp, "load_qspr_model", fake_load)
    pr._qspr_model.cache_clear()
    try:
        pr._qspr_model()
    finally:
        pr._qspr_model.cache_clear()
    assert seen["device"] == "cuda"


def test_qspr_disabled_by_env(monkeypatch):
    monkeypatch.setenv("DES_DISABLE_QSPR", "1")
    pr._qspr_model.cache_clear()
    try:
        assert pr._qspr_model() is None
    finally:
        pr._qspr_model.cache_clear()


def test_clear_resolver_caches_resets_experimental_table():
    pr._experimental_table()  # populate
    assert pr._experimental_table.cache_info().currsize == 1
    pr.clear_resolver_caches()
    assert pr._experimental_table.cache_info().currsize == 0


def test_warn_once_dedups(capsys):
    pr._WARNED_ONCE.clear()
    pr._warn_once("qspr boom")
    pr._warn_once("qspr boom")
    pr._warn_once("other")
    err = capsys.readouterr().err
    assert err.count("qspr boom") == 1
    assert "other" in err


def test_qspr_load_failure_warns_once(monkeypatch, capsys):
    pr._WARNED_ONCE.clear()

    def boom(*a, **k):
        raise RuntimeError("corrupt artifact")

    import des_multi_agent.predictors.melting_point as mp
    monkeypatch.setattr(mp, "load_qspr_model", boom)
    monkeypatch.delenv("DES_DISABLE_QSPR", raising=False)
    pr._qspr_model.cache_clear()
    try:
        assert pr._qspr_model() is None  # degrades gracefully
    finally:
        pr._qspr_model.cache_clear()
    assert "QSPR" in capsys.readouterr().err  # but not silently
