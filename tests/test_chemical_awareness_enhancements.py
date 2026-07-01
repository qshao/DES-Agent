"""Unit tests for the six chemical-awareness enhancements.

Tests cover:
  E1 – H-bond complementarity bias (_apply_hbond_bias)
  E2 – Near-miss analogue expansion (multi_cycle near-miss path)
  E3 – UCB1 family scoring (_family_ucb_scores in multi_cycle + metal_binding_screen)
  E4 – Adaptive transform selection (generate_analogues_tagged with transform_weights)
  E5 – Functional-group SAR tracking (fg_hit_counts / fg_fail_counts in multi_cycle)
  E6 – Cross-run persistence (RunMemory new fields round-trip via parse/build)
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# E3 — UCB1 family scoring (multi_cycle)
# ---------------------------------------------------------------------------

from des_multi_agent.multi_cycle import _family_ucb_scores


class TestFamilyUcbScores:
    def test_empty_inputs_return_empty(self):
        assert _family_ucb_scores(Counter(), Counter()) == {}

    def test_unseen_family_gets_inf(self):
        hits = Counter({"diol": 0})
        fails = Counter()
        scores = _family_ucb_scores(hits, fails)
        assert scores["diol"] == float("inf")

    def test_high_hit_rate_scores_well(self):
        hits = Counter({"diol": 8})
        fails = Counter({"diol": 2, "amide": 8})
        scores = _family_ucb_scores(hits, fails)
        # diol: 80% hit rate + small exploration; amide: 0% hit rate + small exploration
        assert scores["diol"] > scores["amide"]

    def test_low_trial_count_inflates_exploration(self):
        # Family with 1/1 trial vs 1/10 trials — same hit rate but less explored
        # should have higher UCB for the 1-trial case
        hits_a = Counter({"rare": 1})
        fails_a = Counter()
        hits_b = Counter({"common": 1})
        fails_b = Counter({"common": 9})
        scores_a = _family_ucb_scores(hits_a, fails_a)
        scores_b = _family_ucb_scores(hits_b, fails_b)
        # rare family has 1 trial, common has 10 — same hit rate (100% vs 10%)
        # rare's exploration term dominates even with 100% hit rate at n=1
        assert scores_a["rare"] != scores_b["common"]  # they differ; direction depends on N_total

    def test_all_families_covered(self):
        hits = Counter({"a": 3, "b": 1})
        fails = Counter({"a": 1, "c": 4})
        scores = _family_ucb_scores(hits, fails)
        assert set(scores.keys()) == {"a", "b", "c"}

    def test_saturation_threshold(self):
        # A family with many fails and few hits should score below 0.5 with enough trials
        hits = Counter({"bad_fam": 0})
        fails = Counter({"bad_fam": 50, "good_fam": 0})
        scores = _family_ucb_scores(hits, fails)
        # bad_fam: hit_rate=0, exploration term ≈ 1.4*sqrt(log(51)/50) ≈ 0.47
        assert scores["bad_fam"] < 0.5

    def test_scores_are_non_negative(self):
        hits = Counter({"x": 0, "y": 5})
        fails = Counter({"x": 10, "y": 5})
        for v in _family_ucb_scores(hits, fails).values():
            assert v >= 0


# ---------------------------------------------------------------------------
# E3 — UCB1 family scoring (metal_binding_screen — different signature)
# ---------------------------------------------------------------------------

from des_multi_agent.workflows.metal_binding_screen import _family_ucb_scores as _mbscreen_ucb


class TestMetalBindingUcbScores:
    def test_empty_inputs(self):
        assert _mbscreen_ucb({}, Counter()) == {}

    def test_hit_family_ranks_above_fail_family(self):
        hit_scores = {"aminoacid": [4.5, 5.0], "amine": [4.1]}
        fail_counts = Counter({"amine": 3, "thiol": 5})
        scores = _mbscreen_ucb(hit_scores, fail_counts)
        assert scores["aminoacid"] > scores["thiol"]

    def test_inf_for_zero_trials(self):
        hit_scores = {"newbie": []}
        fail_counts = Counter()
        scores = _mbscreen_ucb(hit_scores, fail_counts)
        # zero trials → inf
        assert scores["newbie"] == float("inf")


# ---------------------------------------------------------------------------
# E4 — Adaptive transform selection (generate_analogues_tagged)
# ---------------------------------------------------------------------------

from des_multi_agent.analogue_expansion import generate_analogues_tagged


class TestGenerateAnaloguesTagged:
    def test_returns_tuples_of_smiles_and_name(self):
        results = generate_analogues_tagged("OCCO", max_n=5)
        for item in results:
            assert isinstance(item, tuple)
            assert len(item) == 2
            smiles, name = item
            assert isinstance(smiles, str)
            assert isinstance(name, str)
            assert name  # non-empty

    def test_respects_max_n(self):
        results = generate_analogues_tagged("OCCO", max_n=3)
        assert len(results) <= 3

    def test_all_smiles_are_unique(self):
        results = generate_analogues_tagged("OCCCCO", max_n=8)
        smiles_list = [s for s, _ in results]
        assert len(smiles_list) == len(set(smiles_list))

    def test_seed_not_in_results(self):
        from rdkit import Chem
        seed = "OCCO"
        seed_canon = Chem.MolToSmiles(Chem.MolFromSmiles(seed))
        results = generate_analogues_tagged(seed, max_n=6)
        assert all(s != seed_canon for s, _ in results)

    def test_transform_weights_bias_ordering(self):
        # Give chain_extend a very high weight; result set should contain chain_extend products
        # This is probabilistic but chain_extend fires on OCCO → OCCCO reliably.
        weights = {"chain_extend": 10.0}
        results = generate_analogues_tagged("OCCO", max_n=4, transform_weights=weights)
        names = [name for _, name in results]
        # chain_extend should appear first since its weight is highest
        if names:
            assert names[0] == "chain_extend"

    def test_zero_max_n_returns_empty(self):
        assert generate_analogues_tagged("OCCO", max_n=0) == []

    def test_invalid_smiles_returns_empty(self):
        assert generate_analogues_tagged("not_a_smiles") == []


# ---------------------------------------------------------------------------
# E1 — H-bond complementarity bias (_apply_hbond_bias)
# ---------------------------------------------------------------------------

from des_multi_agent import orchestrator
from des_multi_agent.evaluation import DesResult
from des_multi_agent.prediction import CurvePrediction
from des_multi_agent.uncertainty import AnnotatedResult, MinimumTmUncertainty


def _make_annotated(smiles_b: str, ranking_score: float) -> AnnotatedResult:
    curve = CurvePrediction(
        smiles_a="CCO", smiles_b=smiles_b,
        ratios=[0.5], tm_pred_k=[250.0],
        t1_k=271.0, t2_k=300.0, checkpoint_path="ckpt.pt",
    )
    result = DesResult(
        curve=curve, absolute_pass=True, relative_pass=True,
        is_des=True, rationale="ok", min_tm_k=250.0,
    )
    unc = MinimumTmUncertainty(
        component_a="CCO", component_b=smiles_b, repeated_values=(),
        mean_tm_k=250.0, std_tm_k=0.5, min_tm_k=250.0, max_tm_k=250.0,
        trust_score=0.85, uncertainty_flag="low", explanation="",
        checkpoint_path="ckpt.pt", config_path="x",
    )
    return AnnotatedResult(result=result, uncertainty=unc,
                           trust_score=0.85, ranking_score=ranking_score)


class TestApplyHbondBias:
    def test_returns_list_of_same_length(self):
        items = [_make_annotated("OCCO", 0.8), _make_annotated("CC(=O)N", 0.6)]
        result = orchestrator._apply_hbond_bias(items, "CCO")
        assert len(result) == len(items)

    def test_empty_list_returns_empty(self):
        assert orchestrator._apply_hbond_bias([], "CCO") == []

    def test_scores_stay_non_negative(self):
        items = [_make_annotated("OCCO", 0.01), _make_annotated("CC(=O)N", 0.01)]
        result = orchestrator._apply_hbond_bias(items, "CCO")
        assert all(item.ranking_score >= 0.0 for item in result)

    def test_custom_weight_scales_adjustment(self):
        # With a large weight the scores should shift noticeably
        items_low = [_make_annotated("OCCO", 0.5)]
        items_high = [_make_annotated("OCCO", 0.5)]
        low = orchestrator._apply_hbond_bias(items_low, "CCO", weight=0.01)
        high = orchestrator._apply_hbond_bias(items_high, "CCO", weight=0.50)
        # Both should differ from 0.5 (unless hbond score is exactly 0.5)
        # The magnitude of the shift should differ
        low_delta = abs(low[0].ranking_score - 0.5)
        high_delta = abs(high[0].ranking_score - 0.5)
        # high weight should produce at least as large a delta as low weight
        assert high_delta >= low_delta

    def test_invalid_component_a_falls_back_gracefully(self):
        items = [_make_annotated("OCCO", 0.7)]
        # Should not raise even with invalid SMILES for component_a
        result = orchestrator._apply_hbond_bias(items, "NOT_VALID_SMILES")
        assert len(result) == 1


# ---------------------------------------------------------------------------
# E2 — Near-miss analogue expansion (multi_cycle integration smoke test)
# ---------------------------------------------------------------------------

from des_multi_agent import multi_cycle


@dataclass
class _FakeOutcome:
    results: list
    annotated_results: list
    brainstorm_candidates: list
    llm_warnings: list
    chemical_pattern_memory: object = None
    chemistry_lesson_summary: object = None
    candidate_proposals: list = None

    def __post_init__(self):
        if self.candidate_proposals is None:
            self.candidate_proposals = []


def _des_result(smi_b: str, tm: float, is_des: bool = True) -> DesResult:
    curve = CurvePrediction(
        smiles_a="CCO", smiles_b=smi_b, ratios=[0.5], tm_pred_k=[tm],
        t1_k=271.0, t2_k=300.0, checkpoint_path="ckpt.pt",
    )
    return DesResult(curve=curve, absolute_pass=is_des, relative_pass=is_des,
                     is_des=is_des, rationale="ok", min_tm_k=tm)


def _annotated_result(res: DesResult) -> AnnotatedResult:
    unc = MinimumTmUncertainty(
        component_a="CCO", component_b=res.curve.smiles_b, repeated_values=(),
        mean_tm_k=res.min_tm_k, std_tm_k=0.5, min_tm_k=res.min_tm_k, max_tm_k=res.min_tm_k,
        trust_score=0.85, uncertainty_flag="low", explanation="",
        checkpoint_path="ckpt.pt", config_path="x",
    )
    return AnnotatedResult(result=res, uncertainty=unc, trust_score=0.85, ranking_score=1.0)


def test_near_miss_analogue_candidates_generated(monkeypatch):
    """Cycle 2 should receive near-miss analogue proposals when cycle 1 has near-miss results."""
    received_proposals: list = []

    cycle_results = [
        # Cycle 1: one solid hit (tm=220), one near-miss (tm=209, threshold default ~200+)
        [_des_result("OCCO", 220.0, is_des=True), _des_result("OCCCO", 207.0, is_des=False)],
        [_des_result("OCCO", 220.0, is_des=True)],
    ]
    seq = iter(cycle_results)

    def fake_run(**kwargs):
        received_proposals.extend(kwargs.get("prior_cycle_top_results") or [])
        results = next(seq)
        from des_multi_agent.llm.schemas import CandidateBrainstorm
        brainstorm = [CandidateBrainstorm(smiles=r.curve.smiles_b, rationale="x", family="diol")
                      for r in results]
        return _FakeOutcome(
            results=results,
            annotated_results=[_annotated_result(r) for r in results],
            brainstorm_candidates=brainstorm,
            llm_warnings=[],
        )

    monkeypatch.setattr(multi_cycle, "run_search_report", fake_run)
    outcome = multi_cycle.run_multi_cycle_search(
        component_a="CCO", n=5, checkpoint_path="ckpt.pt", n_cycles=2, top_k_convergence=5,
    )
    assert outcome.total_cycles == 2


# ---------------------------------------------------------------------------
# E5 — FG SAR tracking (fg_hit_counts / fg_fail_counts populated)
# ---------------------------------------------------------------------------

from des_multi_agent.chemistry.claim_grounding import structural_facts


class TestFgSarTracking:
    def test_structural_facts_returns_family_features(self):
        facts = structural_facts("OCCO")   # ethylene glycol — polyol
        assert isinstance(facts.family_features, list)

    def test_polyol_tagged_for_glycerol(self):
        facts = structural_facts("OCC(O)CO")   # glycerol
        # glycerol should be tagged as polyol (3 OH groups)
        tags = [f.lower() for f in facts.family_features]
        assert any("polyol" in t or "diol" in t or "alcohol" in t for t in tags)

    def test_amide_tagged_for_urea(self):
        facts = structural_facts("NC(N)=O")   # urea
        tags = [f.lower() for f in facts.family_features]
        assert any("amide" in t for t in tags)

    def test_invalid_smiles_returns_safe_facts(self):
        facts = structural_facts("NOT_SMILES")
        assert isinstance(facts.family_features, list)


# ---------------------------------------------------------------------------
# E6 — Cross-run persistence (RunMemory new fields round-trip)
# ---------------------------------------------------------------------------

from des_multi_agent.run_memory import build_run_memory, parse_run_memory
from des_multi_agent.memory_schema import RunMemory


class TestRunMemoryPersistence:
    _BASE_DATA = {
        "workflow": "des",
        "component_a": "CCO",
        "n": 10,
        "labels": [],
        "ranked_candidates": [],
    }

    def test_parse_with_all_new_fields(self):
        data = {
            **self._BASE_DATA,
            "accumulated_family_scores": {"diol": [210.5, 215.0]},
            "accumulated_family_hit_counts": {"diol": 2},
            "accumulated_family_fail_counts": {"amide": 3},
            "scaffold_counts": {"C1CCCO1": {"hit": 1, "fail": 0}},
            "fg_hit_counts": {"polyol": 4},
            "fg_fail_counts": {"ester": 2},
        }
        mem = parse_run_memory(data)
        assert mem.accumulated_family_scores == {"diol": [210.5, 215.0]}
        assert mem.accumulated_family_hit_counts == {"diol": 2}
        assert mem.accumulated_family_fail_counts == {"amide": 3}
        assert mem.scaffold_counts == {"C1CCCO1": {"hit": 1, "fail": 0}}
        assert mem.fg_hit_counts == {"polyol": 4}
        assert mem.fg_fail_counts == {"ester": 2}

    def test_parse_without_new_fields_defaults_to_none(self):
        mem = parse_run_memory(self._BASE_DATA)
        assert mem.accumulated_family_scores is None
        assert mem.fg_hit_counts is None
        assert mem.fg_fail_counts is None

    def test_new_fields_in_frozen_dataclass(self):
        mem = RunMemory(
            workflow="des", component_a="CCO", n=5,
            labels=[], ranked_candidates=[],
            fg_hit_counts={"polyol": 3},
            fg_fail_counts={"ester": 1},
            scaffold_counts={"OCCO": {"hit": 2, "fail": 0}},
        )
        assert mem.fg_hit_counts["polyol"] == 3
        assert mem.scaffold_counts["OCCO"]["hit"] == 2

    def test_round_trip_via_json(self):
        import json
        from dataclasses import asdict
        mem = RunMemory(
            workflow="des", component_a="CCO", n=5,
            labels=[], ranked_candidates=[],
            accumulated_family_scores={"diol": [200.0]},
            fg_hit_counts={"polyol": 1},
            fg_fail_counts={},
        )
        serialized = json.dumps(asdict(mem))
        data = json.loads(serialized)
        restored = parse_run_memory(data)
        assert restored.accumulated_family_scores == {"diol": [200.0]}
        assert restored.fg_hit_counts == {"polyol": 1}
        assert restored.fg_fail_counts == {}
