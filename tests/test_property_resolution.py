import pytest

import des_multi_agent.property_resolution as pr
from des_multi_agent.property_resolution import resolve_melting_point
from des_multi_agent.predictors.melting_point import QSPRPrediction


@pytest.fixture(autouse=True)
def _disable_qspr_by_default(monkeypatch):
    """Tests are deterministic and independent of the (heavy, optional) QSPR
    artifact. Tests that exercise the QSPR layer opt in with their own fake."""
    monkeypatch.setattr(pr, "_qspr_model", lambda: None)


class _FakeQSPR:
    def __init__(self, tm_k, std_k):
        self._p = QSPRPrediction(tm_k=tm_k, std_k=std_k)

    def predict(self, smiles):
        return self._p


def test_resolve_melting_point_accepts_override():
    estimate = resolve_melting_point("CCO", override_k=310.0)
    assert estimate.tm_k == 310.0
    assert estimate.source == "override"
    assert estimate.confidence == 1.0


def test_resolve_melting_point_uses_heuristic_for_unknown_smiles():
    # ethanol is not in the experimental table and QSPR is disabled here
    estimate = resolve_melting_point("CCO")
    assert estimate.source == "heuristic"
    assert estimate.tm_k > 150.0


def test_resolve_melting_point_uses_experimental_table_when_available():
    # ethylene glycol is in the training table with experimental Tm 260.6 K
    estimate = resolve_melting_point("OCCO")
    assert estimate.source == "experimental"
    assert estimate.tm_k == 260.6
    assert estimate.confidence >= 0.9


def test_experimental_lookup_is_canonicalized():
    # a non-canonical SMILES of glycerol must still resolve to the same entry
    estimate = resolve_melting_point("C(C(CO)O)O")
    assert estimate.source == "experimental"
    assert estimate.tm_k == 291.22


def test_experimental_confidence_exceeds_heuristic():
    exp = resolve_melting_point("OCCO")
    heur = resolve_melting_point("CCO")
    assert exp.confidence > heur.confidence


def test_qspr_used_when_table_misses_and_model_present(monkeypatch):
    monkeypatch.setattr(pr, "_qspr_model", lambda: _FakeQSPR(tm_k=400.0, std_k=8.0))
    est = resolve_melting_point("CCBr")  # not in the experimental table
    assert est.source == "qspr"
    assert est.tm_k == 400.0
    assert pr._HEURISTIC_CONFIDENCE < est.confidence < pr._EXPERIMENTAL_CONFIDENCE


def test_qspr_confidence_decreases_with_uncertainty(monkeypatch):
    monkeypatch.setattr(pr, "_qspr_model", lambda: _FakeQSPR(tm_k=400.0, std_k=5.0))
    sharp = resolve_melting_point("CCBr")
    monkeypatch.setattr(pr, "_qspr_model", lambda: _FakeQSPR(tm_k=400.0, std_k=60.0))
    fuzzy = resolve_melting_point("CCBr")
    assert sharp.confidence > fuzzy.confidence


def test_falls_back_to_heuristic_when_no_qspr_model():
    est = resolve_melting_point("CCBr")  # autouse fixture disabled QSPR
    assert est.source == "heuristic"


def test_qspr_confidence_lower_for_ionic_components(monkeypatch):
    monkeypatch.setattr(pr, "_qspr_model", lambda: _FakeQSPR(tm_k=400.0, std_k=8.0))
    neutral = resolve_melting_point("CCBr")
    # a quaternary-ammonium salt not in the experimental table: net-neutral but
    # carries charged atoms
    ionic = resolve_melting_point("CCCCCCCC[N+](C)(C)C.[Br-]")
    assert neutral.source == "qspr" and ionic.source == "qspr"
    assert ionic.confidence < neutral.confidence


def test_experimental_wins_over_qspr(monkeypatch):
    def _boom():
        raise AssertionError("QSPR should not be consulted on an experimental hit")

    monkeypatch.setattr(pr, "_qspr_model", _boom)
    est = resolve_melting_point("OCCO")
    assert est.source == "experimental"
