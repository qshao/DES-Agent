"""Integration tests for how partner-reality grading shapes the report.

Covers two report-fidelity guarantees in the orchestrator grounding block:
  #1  A candidate dropped by reality grading leaves no stale claim verdict or
      warning behind — only the reality verdict that explains the drop remains.
  #2  A non-complementary novel partner is flagged once, not twice: the
      DES-plausibility "contradicted" verdict already covers it, so the
      redundant reality "demote" verdict and its warning are suppressed.
"""
from des_multi_agent import orchestrator
from des_multi_agent.evaluation import DesResult
from des_multi_agent.llm.schemas import CandidateBrainstorm
from des_multi_agent.prediction import CurvePrediction
from des_multi_agent.schemas import MeltingPointEstimate

_COMPONENT_A = "CCCC"               # butane — no H-bond capability
_DEMOTE = "CCCC(CC)CCCC"           # 4-ethyloctane — novel, no complementarity
_DROP = "CCCC[Si](C)(C)C"          # silane — disallowed element → structural drop


class _PartnerFakeLLM:
    def review_candidate(self, component_a, candidate_smiles, context):
        from des_multi_agent.llm.schemas import CandidateReview

        return CandidateReview(
            smiles=candidate_smiles, decision="keep", confidence=0.9,
            rationale="ok", notes=[],
        )

    def brainstorm_candidates(self, component_a, constraints, context, **kwargs):
        return [
            CandidateBrainstorm(smiles=_DEMOTE, rationale="novel", family="alkane"),
            CandidateBrainstorm(smiles=_DROP, rationale="novel", family="alkane"),
        ]

    def generate_explanations(self, results, context):
        return []

    def critique_results(self, results, context):
        return []

    def detect_contradictions(self, results, context, facts_block=""):
        return []

    def assess_candidate_chemistry(self, candidate_smiles, context, memory_notes=None):
        return []

    def suggest_next_steps(self, context, memory_notes=None):
        return []


def _run(monkeypatch):
    monkeypatch.setattr(orchestrator, "build_llm_provider", lambda cfg, request_fn=None: _PartnerFakeLLM())
    monkeypatch.setattr(orchestrator, "generate_candidates", lambda component_a, n, constraints=None: [])
    monkeypatch.setattr(orchestrator, "filter_candidates", lambda component_a, candidates: candidates)
    monkeypatch.setattr(
        orchestrator,
        "resolve_melting_point",
        lambda component, override_k=None: MeltingPointEstimate(
            component=component, tm_k=300.0, source="heuristic", confidence=0.5
        ),
    )
    monkeypatch.setattr(
        orchestrator,
        "predict_curve",
        lambda component_a, component_b, t1_k=300.0, t2_k=300.0, checkpoint_path="ckpt.pt", config_path="x": CurvePrediction(
            smiles_a=component_a, smiles_b=component_b, ratios=[0.1],
            tm_pred_k=[250.0], t1_k=t1_k, t2_k=t2_k, checkpoint_path=checkpoint_path,
        ),
    )
    monkeypatch.setattr(
        orchestrator,
        "classify_des",
        lambda curve, thresholds: DesResult(
            curve=curve, absolute_pass=True, relative_pass=True, is_des=True,
            rationale="ok", min_tm_k=250.0,
        ),
    )
    monkeypatch.setattr(orchestrator, "rank_results", lambda results: results)

    return orchestrator.run_search_report(
        component_a=_COMPONENT_A,
        n=2,
        checkpoint_path="ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt",
        llm_cfg={
            "enabled": True, "provider": "ollama", "model_name": "llama3.1",
            "api_base_url": "http://localhost:11434",
        },
    )


def test_dropped_candidate_leaves_no_stale_verdict_or_warning(monkeypatch):
    outcome = _run(monkeypatch)

    # The silane is dropped from results entirely.
    result_smiles = [r.curve.smiles_b for r in outcome.results]
    assert _DROP not in result_smiles

    # No DES-plausibility / family verdict survives for the dropped molecule —
    # only its reality verdict (which explains the drop) is allowed to mention it.
    stale = [
        v for v in outcome.claim_verdicts
        if _DROP in v.claim and getattr(v, "disposition", None) is None
    ]
    assert stale == [], f"stale grounding verdicts for dropped candidate: {[v.claim for v in stale]}"

    # The reality verdict explaining the drop is retained.
    reality = [v for v in outcome.claim_verdicts if v.claim == f"partner reality: {_DROP}"]
    assert len(reality) == 1 and reality[0].status == "novel_implausible"

    # Warnings: the drop is announced, but no [GROUNDING] line lingers for it.
    assert any(w.startswith("[REALITY]") and _DROP in w for w in outcome.llm_warnings)
    assert not any(w.startswith("[GROUNDING]") and _DROP in w for w in outcome.llm_warnings)


def test_noncomplementary_partner_flagged_once_not_twice(monkeypatch):
    outcome = _run(monkeypatch)

    # The demoted molecule survives in results (demote, not drop).
    assert _DEMOTE in [r.curve.smiles_b for r in outcome.results]

    # Exactly one verdict flags it as bad — the plausibility "contradicted"
    # one; the redundant reality "demote" verdict is suppressed.
    flagged = [
        v for v in outcome.claim_verdicts
        if _DEMOTE in v.claim and v.status in ("contradicted", "novel_implausible")
    ]
    assert len(flagged) == 1, f"expected single flag, got {[v.claim for v in flagged]}"
    assert getattr(flagged[0], "disposition", None) is None  # it's the GroundingVerdict

    # And only one warning mentions it — no duplicate [REALITY] demote line.
    demote_warnings = [w for w in outcome.llm_warnings if _DEMOTE in w]
    assert len(demote_warnings) == 1
    assert demote_warnings[0].startswith("[GROUNDING]")
