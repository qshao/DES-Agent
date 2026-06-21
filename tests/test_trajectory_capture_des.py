from dataclasses import dataclass

from des_multi_agent import multi_cycle
from des_multi_agent.evaluation import DesResult
from des_multi_agent.llm.schemas import CandidateBrainstorm
from des_multi_agent.prediction import CurvePrediction
from des_multi_agent.uncertainty import AnnotatedResult, MinimumTmUncertainty


def _curve(smi_b, tm):
    return CurvePrediction(
        smiles_a="CCO", smiles_b=smi_b, ratios=[0.5], tm_pred_k=[tm],
        t1_k=271.0, t2_k=300.0, checkpoint_path="ckpt.pt",
    )


def _result(smi_b, tm):
    return DesResult(curve=_curve(smi_b, tm), absolute_pass=True, relative_pass=True,
                     is_des=True, rationale="ok", min_tm_k=tm)


def _annotated(res):
    unc = MinimumTmUncertainty(
        component_a="CCO", component_b=res.curve.smiles_b, repeated_values=(),
        mean_tm_k=res.min_tm_k, std_tm_k=0.5, min_tm_k=res.min_tm_k, max_tm_k=res.min_tm_k,
        trust_score=0.85, uncertainty_flag="low", explanation="", checkpoint_path="ckpt.pt",
        config_path="x",
    )
    return AnnotatedResult(result=res, uncertainty=unc, trust_score=0.85, ranking_score=1.0)


@dataclass
class _FakeOutcome:
    results: list
    annotated_results: list
    brainstorm_candidates: list
    llm_warnings: list
    chemical_pattern_memory: object = None
    chemistry_lesson_summary: object = None


def _make_fake(cycle_results):
    """Return a fake run_search_report producing canned cycles in sequence."""
    seq = iter(cycle_results)

    def fake(**kwargs):
        results = next(seq)
        brainstorm = [CandidateBrainstorm(smiles=r.curve.smiles_b, rationale="x", family="diol")
                      for r in results]
        return _FakeOutcome(
            results=results,
            annotated_results=[_annotated(r) for r in results],
            brainstorm_candidates=brainstorm,
            llm_warnings=[],
        )

    return fake


def test_des_trajectory_captured(monkeypatch):
    cycle1 = [_result("OCCO", 201.8), _result("O", 225.0)]
    cycle2 = [_result("OCCO", 201.8), _result("OCC(O)CO", 221.0)]
    monkeypatch.setattr(multi_cycle, "run_search_report", _make_fake([cycle1, cycle2]))

    outcome = multi_cycle.run_multi_cycle_search(
        component_a="CCO", n=2, checkpoint_path="ckpt.pt", n_cycles=2, top_k_convergence=5,
    )

    traj = outcome.trajectory
    assert traj is not None
    assert traj.workflow == "des"
    assert traj.metric_label == "min Tm (K)"
    assert len(traj.snapshots) == 2
    # cycle 2 dropped water, gained glycerol (display names resolved)
    s2 = traj.snapshots[1]
    assert any("glycerol" in lbl or "OCC(O)CO" in lbl for lbl in s2.new_entrants)
    assert any("water" in lbl or lbl.endswith("(O)") for lbl in s2.dropouts)
    # final summary present, family ledger non-empty
    assert traj.final_summary
    assert traj.snapshots[0].family_ledger.get("diol") == 2
