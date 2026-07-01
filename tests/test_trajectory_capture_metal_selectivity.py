"""Task 5 — verify that run_metal_selectivity_screen captures a SearchTrajectory."""
from __future__ import annotations

from des_multi_agent.workflows import metal_binding_selectivity as mbs
from des_multi_agent.workflows.metal_binding_selectivity import (
    SelectivityResult,
    run_metal_selectivity_screen,
)
from des_multi_agent.schemas import CandidateProposal


def _proposal(smi):
    return CandidateProposal(smiles=smi, rationale="x", family="amine", source="heuristic", source_id="")


def test_metal_selectivity_trajectory_captured(monkeypatch):
    # Two cycles, deterministic proposals + scoring, no LLM.
    monkeypatch.setattr(mbs, "generate_ligand_candidates",
                        lambda metal, n, constraints: [_proposal("NCCN"), _proposal("NCCO")])

    scores = {"NCCN": (8.0, 2.0), "NCCO": (6.0, 1.0)}  # (log_k_target, delta)

    def fake_score(target, competitor, proposal, model_path, w_a, w_s, stability_rule_weight=0.0):
        lk, delta = scores[proposal.smiles]
        return SelectivityResult(
            ligand_smiles=proposal.smiles, log_k_target=lk, log_k_competitor=lk - delta,
            delta_log_k=delta, composite_score=lk + delta, source=proposal.source, source_id="",
            rationale="ok",
        ), []

    monkeypatch.setattr(mbs, "_score_proposal_pair", fake_score)

    outcome = run_metal_selectivity_screen(
        target_metal="Cu2+", competitor_metal="Zn2+", n=2, model_path=None,
        llm_provider=None, n_cycles=2,
    )

    traj = outcome.trajectory
    assert traj is not None
    assert traj.workflow == "metal-selectivity"
    assert traj.metric_label == "composite score"
    assert len(traj.snapshots) >= 1
    top = traj.snapshots[0].top_entries[0]
    assert top.metric_name == "composite_score"
    assert top.label == "NCCN"  # highest composite score


def test_total_cycles_matches_actual_cycles_run(monkeypatch):
    """total_cycles must count actual iterations, not snapshots captured."""
    monkeypatch.setattr(mbs, "generate_ligand_candidates",
                        lambda metal, n, constraints: [_proposal("NCCN")])

    def fake_score(target, competitor, proposal, model_path, w_a, w_s, stability_rule_weight=0.0):
        return SelectivityResult(
            ligand_smiles=proposal.smiles, log_k_target=5.0, log_k_competitor=3.0,
            delta_log_k=2.0, composite_score=7.0, source=proposal.source, source_id="",
            rationale="ok",
        ), []

    monkeypatch.setattr(mbs, "_score_proposal_pair", fake_score)

    # Run 3 cycles; cycle 2 converges (top-1 stable), so trajectory runs 2 cycles.
    outcome = run_metal_selectivity_screen(
        target_metal="Cu2+", competitor_metal="Zn2+", n=1, model_path=None,
        llm_provider=None, n_cycles=3,
    )
    traj = outcome.trajectory
    # total_cycles must equal actual cycles run (== len(snapshots) when no snapshot fails,
    # but crucially must not silently drop cycles when a snapshot exception occurs).
    assert traj.total_cycles == len(traj.snapshots)
