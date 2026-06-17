"""Tests for pH-aware coordination grounding wired into metal-binding workflows."""
from __future__ import annotations


def test_metal_binding_screen_accepts_binding_ph():
    import inspect
    from des_multi_agent.workflows.metal_binding_screen import run_metal_binding_screen
    sig = inspect.signature(run_metal_binding_screen)
    assert "binding_pH" in sig.parameters
    assert sig.parameters["binding_pH"].default == 7.0


def test_metal_selectivity_screen_accepts_binding_ph():
    import inspect
    from des_multi_agent.workflows.metal_binding_selectivity import run_metal_selectivity_screen
    sig = inspect.signature(run_metal_selectivity_screen)
    assert "binding_pH" in sig.parameters
    assert sig.parameters["binding_pH"].default == 7.0


def test_selectivity_screen_grounds_coordination_for_brainstormed_ligands():
    """With an LLM that brainstorms a ligand carrying a coordination rationale,
    the selectivity outcome's claim_verdicts include a coordination verdict."""
    from des_multi_agent.llm.schemas import CandidateBrainstorm, CandidateReview
    from des_multi_agent.workflows.metal_binding_selectivity import run_metal_selectivity_screen

    class _FakeLLM:
        def brainstorm_ligands_selectivity(self, target, competitor, constraints, context):
            return [CandidateBrainstorm(smiles="NCC(=O)O", rationale="bidentate N,O-chelator", family="amino acid")]
        def review_ligand(self, metal_ion, ligand_smiles, context):
            return CandidateReview(smiles=ligand_smiles, decision="keep", confidence=0.8, rationale="ok", notes=[])

    outcome = run_metal_selectivity_screen(
        "Cu2+", "Zn2+", n=3, llm_provider=_FakeLLM(), n_cycles=1, binding_pH=7.0,
    )
    assert isinstance(outcome.claim_verdicts, list)
    assert any(getattr(v, "claim", "") == "bidentate N,O-chelator" for v in outcome.claim_verdicts)
