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


# ---------------------------------------------------------------------------
# _top_k_stable
# ---------------------------------------------------------------------------

def test_top_k_stable_identical():
    results = [_make_result(s, float(i), 0.0) for i, s in enumerate(["A", "B", "C", "D", "E"])]
    assert _top_k_stable(results, results, k=5)


def test_top_k_stable_different():
    r1 = [_make_result(s, float(i), 0.0) for i, s in enumerate(["A", "B", "C", "D", "E"])]
    r2 = [_make_result(s, float(i), 0.0) for i, s in enumerate(["A", "B", "C", "D", "F"])]
    assert not _top_k_stable(r1, r2, k=5)


# ---------------------------------------------------------------------------
# run_metal_selectivity_screen — no LLM
# ---------------------------------------------------------------------------

def test_run_screen_no_llm_returns_outcome():
    outcome = run_metal_selectivity_screen("Cu2+", "Zn2+", n=5, n_cycles=1)
    assert isinstance(outcome, SelectivityScreenOutcome)
    assert outcome.target_metal == "Cu2+"
    assert outcome.competitor_metal == "Zn2+"
    assert len(outcome.results) > 0
    assert outcome.llm_brainstorm == []
    assert outcome.llm_candidate_reviews == []


def test_run_screen_results_sorted_by_composite_score():
    outcome = run_metal_selectivity_screen("Cu2+", "Zn2+", n=10, n_cycles=1)
    scores = [r.composite_score for r in outcome.results]
    assert scores == sorted(scores, reverse=True)


def test_run_screen_delta_log_k_is_difference():
    outcome = run_metal_selectivity_screen("Cu2+", "Zn2+", n=5, n_cycles=1)
    for r in outcome.results:
        assert abs(r.delta_log_k - (r.log_k_target - r.log_k_competitor)) < 1e-9


def test_run_screen_composite_score_formula():
    outcome = run_metal_selectivity_screen(
        "Cu2+", "Zn2+", n=5, n_cycles=1, w_affinity=0.5, w_selectivity=0.5
    )
    for r in outcome.results:
        expected = 0.5 * r.log_k_target + 0.5 * r.delta_log_k
        assert abs(r.composite_score - expected) < 1e-9


def test_run_screen_no_duplicate_smiles():
    outcome = run_metal_selectivity_screen("Cu2+", "Zn2+", n=20, n_cycles=1)
    smiles_list = [r.ligand_smiles for r in outcome.results]
    assert len(smiles_list) == len(set(smiles_list))


def test_run_screen_multi_cycle_no_llm():
    outcome = run_metal_selectivity_screen("Cu2+", "Zn2+", n=5, n_cycles=3)
    assert outcome.n_cycles == 3
    assert len(outcome.results) > 0
