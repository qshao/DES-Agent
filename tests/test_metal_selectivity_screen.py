"""Tests for the metal ion selectivity screening workflow."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock

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
    # Inject a SMILES already in the heuristic library via mock LLM to exercise deduplication
    mock_llm = MagicMock()
    mock_llm.brainstorm_ligands_selectivity.return_value = [
        # "NCC(=O)O" is the first heuristic entry; dedup must filter it out
        CandidateBrainstorm(smiles="NCC(=O)O", rationale="duplicate", family="aminoacid"),
        CandidateBrainstorm(smiles="c1ccnc(-c2ccccn2)c1", rationale="bipy", family="polypyridyl"),
    ]
    mock_llm.review_ligand.return_value = MagicMock(
        smiles="c1ccnc(-c2ccccn2)c1", decision="keep", confidence=0.9, rationale="ok", notes=[],
    )
    outcome = run_metal_selectivity_screen("Cu2+", "Zn2+", n=5, llm_provider=mock_llm, n_cycles=1)
    smiles_list = [r.ligand_smiles for r in outcome.results]
    assert len(smiles_list) == len(set(smiles_list))


def test_run_screen_multi_cycle_no_llm_completes_gracefully():
    # Without LLM, cycles 2+ regenerate the same heuristic candidates (all already seen).
    # The workflow breaks early but preserves results from cycle 1.
    outcome_1 = run_metal_selectivity_screen("Cu2+", "Zn2+", n=5, n_cycles=1)
    outcome_3 = run_metal_selectivity_screen("Cu2+", "Zn2+", n=5, n_cycles=3)
    # Same unique candidates scored regardless of n_cycles (heuristic library exhausted after cycle 1)
    assert outcome_1.n_screened == outcome_3.n_screened
    assert len(outcome_3.results) > 0


# ---------------------------------------------------------------------------
# LLM prompt
# ---------------------------------------------------------------------------

from des_multi_agent.llm.prompts import ligand_selectivity_brainstorm_prompt
from des_multi_agent.llm.schemas import LigandFamily


def test_selectivity_brainstorm_prompt_contains_both_metals():
    prompt = ligand_selectivity_brainstorm_prompt("Cu2+", "Zn2+", None, "context")
    assert "Cu2+" in prompt
    assert "Zn2+" in prompt


def test_selectivity_brainstorm_prompt_contains_smiles_instruction():
    prompt = ligand_selectivity_brainstorm_prompt("Fe3+", "Ca2+", None, "context")
    assert "smiles" in prompt.lower()


def test_selectivity_brainstorm_prompt_with_families_includes_family_names():
    families = [LigandFamily(name="catecholates", rationale="bidentate O-donors", coordination_mode="bidentate O,O")]
    prompt = ligand_selectivity_brainstorm_prompt("Cu2+", "Zn2+", None, "context", families=families)
    assert "catecholates" in prompt
    assert "bidentate O,O" in prompt


# ---------------------------------------------------------------------------
# run_metal_selectivity_screen — with mock LLM
# ---------------------------------------------------------------------------

def test_run_screen_with_llm_brainstorm_called():
    mock_llm = MagicMock()
    mock_llm.brainstorm_ligands_selectivity.return_value = [
        CandidateBrainstorm(smiles="c1ccnc(-c2ccccn2)c1", rationale="bidentate N,N", family="polypyridyl"),
    ]
    mock_llm.review_ligand.return_value = MagicMock(
        smiles="c1ccnc(-c2ccccn2)c1", decision="keep", confidence=0.9,
        rationale="good chelator", notes=[],
    )

    outcome = run_metal_selectivity_screen("Cu2+", "Zn2+", n=5, llm_provider=mock_llm, n_cycles=1)
    assert mock_llm.brainstorm_ligands_selectivity.called
    call_args = mock_llm.brainstorm_ligands_selectivity.call_args
    assert "Cu2+" in str(call_args)
    assert "Zn2+" in str(call_args)


def test_run_screen_llm_brainstorm_failure_adds_warning():
    mock_llm = MagicMock()
    mock_llm.brainstorm_ligands_selectivity.side_effect = RuntimeError("LLM down")

    outcome = run_metal_selectivity_screen("Cu2+", "Zn2+", n=5, llm_provider=mock_llm, n_cycles=1)
    assert len(outcome.warnings) > 0
    assert any("brainstorm" in w.lower() for w in outcome.warnings)


def test_run_screen_skips_invalid_llm_smiles():
    mock_llm = MagicMock()
    mock_llm.brainstorm_ligands_selectivity.return_value = [
        CandidateBrainstorm(smiles="NOT_A_SMILES", rationale="bad", family="test"),
        CandidateBrainstorm(smiles="NCC(=O)O", rationale="glycine", family="aminoacid"),
    ]
    mock_llm.review_ligand.return_value = MagicMock(
        smiles="NCC(=O)O", decision="keep", confidence=0.8, rationale="ok", notes=[],
    )

    outcome = run_metal_selectivity_screen("Cu2+", "Zn2+", n=5, llm_provider=mock_llm, n_cycles=1)
    from rdkit import Chem
    for r in outcome.results:
        assert Chem.MolFromSmiles(r.ligand_smiles) is not None


# ---------------------------------------------------------------------------
# Report format
# ---------------------------------------------------------------------------

def test_format_metal_selectivity_report_contains_headers():
    from des_multi_agent.reporting import format_metal_selectivity_report
    outcome = run_metal_selectivity_screen("Cu2+", "Zn2+", n=3, n_cycles=1)
    report = format_metal_selectivity_report(outcome)
    assert "Cu2+" in report
    assert "Zn2+" in report
    assert "delta_log_k" in report
    assert "score" in report
    assert "log_k_target" in report
    assert "log_k_competitor" in report


def test_format_metal_selectivity_report_no_results():
    from des_multi_agent.reporting import format_metal_selectivity_report
    outcome = SelectivityScreenOutcome(
        target_metal="Cu2+", competitor_metal="Zn2+",
        results=[], n_screened=0, n_cycles=1,
    )
    report = format_metal_selectivity_report(outcome)
    assert "Cu2+" in report
    assert "none" in report.lower()


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------

def test_cli_metal_selectivity_routes_correctly(monkeypatch, capsys):
    """--workflow metal-selectivity without LLM should call run_metal_selectivity_screen."""
    import des_multi_agent.cli as cli_module
    import des_multi_agent.workflows.metal_binding_selectivity as sel_module

    fake_outcome = SelectivityScreenOutcome(
        target_metal="Cu2+", competitor_metal="Zn2+",
        results=[], n_screened=5, n_cycles=1,
    )
    monkeypatch.setattr(sel_module, "run_metal_selectivity_screen", lambda **kw: fake_outcome)
    monkeypatch.setattr(cli_module, "run_metal_selectivity_screen", lambda **kw: fake_outcome)

    cli_module.main([
        "--workflow", "metal-selectivity",
        "--target-metal-ion", "Cu2+",
        "--competitor-metal-ion", "Zn2+",
        "--n", "5",
    ])
    out = capsys.readouterr().out
    assert "Metal Selectivity Screen" in out or "summary:" in out.lower()


def test_cli_metal_binding_single_pair_unchanged_by_selectivity(monkeypatch, capsys):
    """Existing --workflow metal-binding --ligand-smiles path is not broken."""
    import des_multi_agent.cli as cli_module

    class _FakeOutcome:
        metal_ion = "Cu2+"
        ligand_smiles = "NCCN"
        prediction = type("P", (), {
            "value": 5.5, "units": "log K",
            "model_name": "mock", "source": "mock", "warnings": ()
        })()
        warnings = ()

    monkeypatch.setattr(cli_module, "run_metal_binding_workflow", lambda *a, **kw: _FakeOutcome())
    monkeypatch.setattr(cli_module, "format_metal_binding_report", lambda o: "SINGLE PAIR REPORT")

    cli_module.main([
        "--workflow", "metal-binding",
        "--metal-ion", "Cu2+",
        "--ligand-smiles", "NCCN",
    ])
    out = capsys.readouterr().out
    assert "SINGLE PAIR REPORT" in out


from des_multi_agent.workflows.metal_binding_selectivity import _build_selectivity_context


def test_build_context_includes_des_hints_when_provided():
    ctx = _build_selectivity_context(
        "Cu2+", "Zn2+", [], 2, 0.5, 0.5,
        des_compatible_hints=["NCC(=O)O"],
        des_incompatible_hints=["c1ccncc1"],
    )
    assert "formed DES" in ctx
    assert "NCC(=O)O" in ctx
    assert "NOT form DES" in ctx
    assert "c1ccncc1" in ctx


def test_build_context_omits_hint_sections_when_none():
    ctx = _build_selectivity_context("Cu2+", "Zn2+", [], 1, 0.5, 0.5)
    assert "formed DES" not in ctx
    assert "NOT form DES" not in ctx


def test_stability_rule_blend_discriminates_ni_over_co():
    # with the Irving-Williams rule blend, Ni2+ is favoured over Co2+ for chelators
    out = run_metal_selectivity_screen("Ni2+", "Co2+", n=6, n_cycles=1, stability_rule_weight=1.0)
    assert out.results
    assert all(r.delta_log_k > 0 for r in out.results)
    # invariant still holds under the blend
    for r in out.results:
        assert abs(r.delta_log_k - (r.log_k_target - r.log_k_competitor)) < 1e-9


def test_stability_rule_weight_zero_runs():
    out = run_metal_selectivity_screen("Ni2+", "Co2+", n=4, n_cycles=1, stability_rule_weight=0.0)
    assert out.results
