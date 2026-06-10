"""Tests for the metal ion selectivity screening workflow."""
from __future__ import annotations

import pytest

from des_multi_agent.workflows.metal_binding_selectivity import (
    SelectivityResult,
    SelectivityScreenOutcome,
    _top_k_stable,
    run_metal_selectivity_screen,
)
from des_multi_agent.llm.schemas import CandidateBrainstorm, CandidateReview


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(smiles: str, log_k_target: float, log_k_competitor: float,
                 w_aff: float = 0.5, w_sel: float = 0.5) -> SelectivityResult:
    delta = log_k_target - log_k_competitor
    score = w_aff * log_k_target + w_sel * delta
    return SelectivityResult(
        ligand_smiles=smiles,
        log_k_target=log_k_target,
        log_k_competitor=log_k_competitor,
        delta_log_k=delta,
        composite_score=score,
        source="heuristic",
        source_id="test",
        rationale="test",
    )


# ---------------------------------------------------------------------------
# Dataclass + scoring formula
# ---------------------------------------------------------------------------

def test_selectivity_result_delta_log_k():
    r = _make_result("NCC(=O)O", log_k_target=10.0, log_k_competitor=6.0)
    assert abs(r.delta_log_k - 4.0) < 1e-9


def test_selectivity_result_composite_score_equal_weights():
    r = _make_result("NCC(=O)O", log_k_target=10.0, log_k_competitor=6.0,
                     w_aff=0.5, w_sel=0.5)
    # 0.5 * 10.0 + 0.5 * 4.0 = 7.0
    assert abs(r.composite_score - 7.0) < 1e-9


def test_selectivity_result_composite_score_affinity_only():
    r = _make_result("NCC(=O)O", log_k_target=10.0, log_k_competitor=6.0,
                     w_aff=1.0, w_sel=0.0)
    assert abs(r.composite_score - 10.0) < 1e-9


def test_selectivity_result_composite_score_selectivity_only():
    r = _make_result("NCC(=O)O", log_k_target=10.0, log_k_competitor=6.0,
                     w_aff=0.0, w_sel=1.0)
    assert abs(r.composite_score - 4.0) < 1e-9
