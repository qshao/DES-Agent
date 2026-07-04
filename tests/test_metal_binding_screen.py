"""Tests for the metal-binding ligand screening (LLM + iteration) features."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from des_multi_agent.candidate_generation_ligand import generate_ligand_candidates
from des_multi_agent.llm.parser import parse_ligand_families
from des_multi_agent.llm.prompts import (
    ligand_brainstorm_prompt,
    ligand_family_selection_prompt,
    ligand_review_prompt,
)
from des_multi_agent.llm.schemas import CandidateBrainstorm, CandidateReview, LigandFamily
from des_multi_agent.workflows.metal_binding_screen import (
    LigandScreenResult,
    MetalBindingScreenOutcome,
    _top_k_stable,
    run_metal_binding_screen,
)


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------

def test_generate_ligand_candidates_returns_n():
    candidates = generate_ligand_candidates("Cu2+", 5)
    assert len(candidates) == 5


def test_generate_ligand_candidates_all_valid_smiles():
    from rdkit import Chem
    candidates = generate_ligand_candidates("Fe3+", 20)
    for c in candidates:
        assert Chem.MolFromSmiles(c.smiles) is not None, f"Invalid SMILES: {c.smiles}"


def test_generate_ligand_candidates_no_duplicates():
    candidates = generate_ligand_candidates("Zn2+", 20)
    smiles_set = {c.smiles for c in candidates}
    assert len(smiles_set) == len(candidates)


def test_generate_ligand_candidates_source():
    candidates = generate_ligand_candidates("Cu2+", 3)
    for c in candidates:
        assert c.source == "heuristic"
        assert c.source_id == "rule-based-ligand-library"


# ---------------------------------------------------------------------------
# LLM prompts
# ---------------------------------------------------------------------------

def test_ligand_family_selection_prompt_contains_metal():
    prompt = ligand_family_selection_prompt("Cu2+", None, "context")
    assert "Cu2+" in prompt
    assert "coordination" in prompt.lower()


def test_ligand_brainstorm_prompt_contains_metal():
    prompt = ligand_brainstorm_prompt("Fe3+", None, "context")
    assert "Fe3+" in prompt
    assert "smiles" in prompt.lower()


def test_ligand_brainstorm_prompt_with_families():
    families = [LigandFamily(name="catecholates", rationale="bidentate O-donors", coordination_mode="bidentate O,O")]
    prompt = ligand_brainstorm_prompt("Cu2+", None, "context", families=families)
    assert "catecholates" in prompt
    assert "bidentate O,O" in prompt


def test_ligand_review_prompt_contains_metal_and_smiles():
    prompt = ligand_review_prompt("Zn2+", "NCC(=O)O", "context")
    assert "Zn2+" in prompt
    assert "NCC(=O)O" in prompt
    assert "decision" in prompt.lower()


# ---------------------------------------------------------------------------
# LigandFamily parser
# ---------------------------------------------------------------------------

def test_parse_ligand_families_valid():
    raw = '[{"name": "catecholates", "rationale": "O-donors", "coordination_mode": "bidentate O,O"}]'
    result = parse_ligand_families(raw)
    assert len(result) == 1
    assert result[0].name == "catecholates"
    assert result[0].coordination_mode == "bidentate O,O"


def test_parse_ligand_families_missing_coordination_mode_defaults():
    raw = '[{"name": "amines", "rationale": "N-donors"}]'
    result = parse_ligand_families(raw)
    assert len(result) == 1
    assert result[0].coordination_mode == "unspecified"


def test_parse_ligand_families_empty_list():
    assert parse_ligand_families("[]") == []


def test_parse_ligand_families_invalid_json():
    with pytest.raises(ValueError):
        parse_ligand_families("not json")


# ---------------------------------------------------------------------------
# _top_k_stable convergence check
# ---------------------------------------------------------------------------

def _make_result(smiles: str, log_k: float) -> LigandScreenResult:
    from des_multi_agent.predictors.stability_constants import StabilityConstantPrediction
    pred = StabilityConstantPrediction(
        task="stability_constant", value=log_k, units="log K",
        model_name="mock", source="mock", warnings=(),
        metadata={}, metal_ion="Cu2+", ligand=smiles,
    )
    return LigandScreenResult(
        metal_ion="Cu2+", ligand_smiles=smiles, prediction=pred,
        log_k=log_k, source="heuristic", source_id="test", rationale="test",
    )


def test_top_k_stable_identical():
    results = [_make_result(s, i) for i, s in enumerate(["A", "B", "C", "D", "E"])]
    assert _top_k_stable(results, results, k=5)


def test_top_k_stable_different():
    r1 = [_make_result(s, i) for i, s in enumerate(["A", "B", "C", "D", "E"])]
    r2 = [_make_result(s, i) for i, s in enumerate(["A", "B", "C", "D", "F"])]
    assert not _top_k_stable(r1, r2, k=5)


# ---------------------------------------------------------------------------
# run_metal_binding_screen — no LLM
# ---------------------------------------------------------------------------

def test_run_metal_binding_screen_no_llm():
    outcome = run_metal_binding_screen("Cu2+", n=5, n_cycles=1)
    assert isinstance(outcome, MetalBindingScreenOutcome)
    assert len(outcome.results) > 0
    assert outcome.metal_ion == "Cu2+"
    assert outcome.llm_brainstorm == []
    assert outcome.llm_candidate_reviews == []


def test_run_metal_binding_screen_results_sorted_by_log_k():
    outcome = run_metal_binding_screen("Cu2+", n=10, n_cycles=1)
    log_ks = [r.log_k for r in outcome.results]
    assert log_ks == sorted(log_ks, reverse=True)


def test_run_metal_binding_screen_multi_cycle_no_llm():
    outcome = run_metal_binding_screen("Cu2+", n=5, n_cycles=3)
    assert outcome.n_cycles == 3
    assert len(outcome.results) > 0


# ---------------------------------------------------------------------------
# run_metal_binding_screen — with mock LLM
# ---------------------------------------------------------------------------

def test_run_metal_binding_screen_with_llm_brainstorm():
    mock_llm = MagicMock()
    mock_llm.brainstorm_ligands.return_value = [
        CandidateBrainstorm(smiles="c1ccnc(-c2ccccn2)c1", rationale="bidentate N,N", family="polypyridyl"),
    ]
    mock_llm.review_ligand.return_value = MagicMock(
        smiles="c1ccnc(-c2ccccn2)c1", decision="keep", confidence=0.9,
        rationale="good chelator", notes=[],
    )

    outcome = run_metal_binding_screen("Cu2+", n=5, llm_provider=mock_llm, n_cycles=1)
    assert mock_llm.brainstorm_ligands.called
    assert len(outcome.llm_brainstorm) >= 1


def test_run_metal_binding_screen_llm_brainstorm_failure_does_not_crash():
    mock_llm = MagicMock()
    mock_llm.brainstorm_ligands.side_effect = RuntimeError("LLM down")

    outcome = run_metal_binding_screen("Cu2+", n=5, llm_provider=mock_llm, n_cycles=1)
    assert len(outcome.warnings) > 0
    assert any("brainstorm" in w.lower() for w in outcome.warnings)


def test_run_metal_binding_screen_skips_invalid_llm_smiles():
    mock_llm = MagicMock()
    mock_llm.brainstorm_ligands.return_value = [
        CandidateBrainstorm(smiles="NOT_A_SMILES", rationale="bad", family="test"),
        CandidateBrainstorm(smiles="NCC(=O)O", rationale="glycine", family="aminoacid"),
    ]
    mock_llm.review_ligand.return_value = MagicMock(
        smiles="NCC(=O)O", decision="keep", confidence=0.8, rationale="ok", notes=[],
    )

    outcome = run_metal_binding_screen("Cu2+", n=5, llm_provider=mock_llm, n_cycles=1)
    for r in outcome.results:
        from rdkit import Chem
        assert Chem.MolFromSmiles(r.ligand_smiles) is not None


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def test_format_metal_binding_screen_report_basic():
    from des_multi_agent.reporting import format_metal_binding_screen_report
    outcome = run_metal_binding_screen("Cu2+", n=3, n_cycles=1)
    report = format_metal_binding_screen_report(outcome)
    assert "Cu2+" in report
    assert "log_k" in report
    assert "ligand" in report


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------

def test_cli_metal_binding_screen_no_ligand_smiles(monkeypatch, capsys):
    """No --ligand-smiles → screening mode."""
    import des_multi_agent.cli as cli_module
    import des_multi_agent.workflows.metal_binding_screen as screen_module

    fake_outcome = MetalBindingScreenOutcome(
        metal_ion="Cu2+", results=[], n_screened=5, n_cycles=1,
    )
    monkeypatch.setattr(screen_module, "run_metal_binding_screen", lambda **kw: fake_outcome)
    monkeypatch.setattr(cli_module, "run_metal_binding_screen", lambda **kw: fake_outcome)

    cli_module.main([
        "--workflow", "metal-binding",
        "--metal-ion", "Cu2+",
        "--n", "5",
    ])
    out = capsys.readouterr().out
    assert "Metal Binding Screen" in out or "summary:" in out


def test_cli_metal_binding_single_pair_still_works(monkeypatch, capsys):
    """--ligand-smiles given → single-pair mode unchanged."""
    import des_multi_agent.cli as cli_module
    import des_multi_agent.workflows.metal_binding as mb_module

    class _FakeOutcome:
        metal_ion = "Cu2+"
        ligand_smiles = "NCCN"
        prediction = type("P", (), {"value": 5.5, "units": "log K",
                                     "model_name": "mock", "source": "mock", "warnings": ()})()
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


# ---------------------------------------------------------------------------
# Phase 5: claim_verdicts field (grounding)
# ---------------------------------------------------------------------------

def test_run_metal_binding_screen_claim_verdicts_is_list():
    """claim_verdicts field exists and is a list on MetalBindingScreenOutcome."""
    outcome = MetalBindingScreenOutcome(metal_ion="Cu2+", results=[], n_screened=0, n_cycles=1)
    assert isinstance(outcome.claim_verdicts, list)


def test_run_metal_binding_screen_claim_verdicts_populated_with_llm():
    """When LLM provides brainstorms with rationale, claim_verdicts is populated."""
    mock_llm = MagicMock()
    mock_llm.brainstorm_ligands.return_value = [
        CandidateBrainstorm(
            smiles="NCC(=O)O",
            rationale="bidentate N,O chelator",
            family="aminoacid",
        ),
    ]
    mock_llm.review_ligand.return_value = MagicMock(
        smiles="NCC(=O)O", decision="keep", confidence=0.9,
        rationale="good chelator", notes=[],
    )
    outcome = run_metal_binding_screen("Cu2+", n=3, llm_provider=mock_llm, n_cycles=1)
    assert isinstance(outcome.claim_verdicts, list)
    assert len(outcome.claim_verdicts) >= 1


def test_metal_screen_llm_reviews_run_concurrently():
    import threading

    from des_multi_agent.concurrency import run_concurrent
    from des_multi_agent.predictors.stability_constants import StabilityConstantPrediction
    from des_multi_agent.workflows.metal_binding_screen import LigandScreenResult

    n_ligands = 3
    barrier = threading.Barrier(n_ligands, timeout=2.0)

    class BarrierProvider:
        def review_ligand(self, metal_ion, ligand_smiles, context):
            barrier.wait()
            return CandidateReview(
                smiles=ligand_smiles, decision="keep", confidence=0.9, rationale="ok", notes=[],
            )

    def _cycle_result(smiles: str) -> LigandScreenResult:
        prediction = StabilityConstantPrediction(
            task="log_k", value=8.0, units="log_k", model_name="heuristic",
            source="heuristic", warnings=(), metadata={}, metal_ion="Cu2+", ligand=smiles,
        )
        return LigandScreenResult(
            metal_ion="Cu2+", ligand_smiles=smiles, prediction=prediction,
            log_k=8.0, source="heuristic", source_id="rule", rationale="demo",
        )

    cycle_results = [_cycle_result(f"C{i}") for i in range(n_ligands)]
    llm_provider = BarrierProvider()
    context = "context"
    all_reviews: list[CandidateReview] = []
    all_warnings: list[str] = []

    review_results = run_concurrent(cycle_results, lambda r: llm_provider.review_ligand("Cu2+", r.ligand_smiles, context))
    for r, res in zip(cycle_results, review_results):
        if res.error is not None:
            all_warnings.append(f"LLM review failed for {r.ligand_smiles}: {res.error}")
            continue
        all_reviews.append(res.value)

    assert len(all_reviews) == n_ligands
    assert all_warnings == []
