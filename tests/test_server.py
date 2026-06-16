"""Tests for FastAPI server name resolution integration."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from des_multi_agent.server import app

client = TestClient(app)


def _minimal_payload(**overrides) -> dict:
    base = {
        "component_a": "NC(N)=O",
        "n": 2,
        "checkpoint_path": "fake.pt",
    }
    base.update(overrides)
    return base


def test_search_resolves_molecule_name(monkeypatch):
    """POST /search with a molecule name for component_a resolves to SMILES."""
    received = {}

    def fake_run_search_report(component_a, **kwargs):
        received["component_a"] = component_a
        from des_multi_agent.orchestrator import SearchOutcome
        return SearchOutcome(
            results=[], annotated_results=[], candidate_proposals=[],
            candidate_reviews=[], brainstorm_candidates=[], explanation_notes=[],
            critique_notes=[], llm_warnings=[], contradiction_notes=[],
            viscosity_predictions=[], chemical_pattern_memory=None,
            chemistry_lesson_summary=None,
        )

    monkeypatch.setattr("des_multi_agent.server.run_search_report", fake_run_search_report)
    monkeypatch.setattr("des_multi_agent.server.format_report", lambda outcome, **kw: "")

    resp = client.post("/search", json=_minimal_payload(component_a="urea"))
    assert resp.status_code == 200
    assert received.get("component_a") == "NC(N)=O"


def test_search_returns_422_for_unknown_molecule_name():
    """POST /search with an unresolvable name returns 422."""
    resp = client.post("/search", json=_minimal_payload(component_a="not_a_real_molecule_xyz"))
    assert resp.status_code == 422
    assert "not_a_real_molecule_xyz" in resp.json()["detail"]
