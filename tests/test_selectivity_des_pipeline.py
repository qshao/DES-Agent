from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from des_multi_agent.workflows.selectivity_des_pipeline import (
    LigandDesResult,
    SelectivityDesPipelineOutcome,
    _bridge_filter,
)
from des_multi_agent.workflows.metal_binding_selectivity import (
    SelectivityResult,
    SelectivityScreenOutcome,
)


def _sel_result(smiles: str, delta: float = 1.0, score: float = 5.0) -> SelectivityResult:
    return SelectivityResult(
        ligand_smiles=smiles,
        log_k_target=10.0,
        log_k_competitor=10.0 - delta,
        delta_log_k=delta,
        composite_score=score,
        source="heuristic",
        source_id="",
        rationale="",
    )


def _sel_outcome(smiles_list: list[str]) -> SelectivityScreenOutcome:
    return SelectivityScreenOutcome(
        target_metal="Cu2+",
        competitor_metal="Zn2+",
        results=[_sel_result(s) for s in smiles_list],
        n_screened=len(smiles_list),
        n_cycles=1,
    )


# --- LigandDesResult ---

def test_ligand_des_result_des_compatible_true_when_any_is_des():
    dr = MagicMock()
    dr.is_des = True
    ldr = LigandDesResult(
        ligand=_sel_result("NCC(=O)O"),
        des_results=[dr],
        n_des_screened=5,
        des_compatible=True,
    )
    assert ldr.des_compatible is True


def test_ligand_des_result_des_compatible_false_when_no_des():
    dr = MagicMock()
    dr.is_des = False
    ldr = LigandDesResult(
        ligand=_sel_result("NCC(=O)O"),
        des_results=[dr],
        n_des_screened=5,
        des_compatible=False,
    )
    assert ldr.des_compatible is False


# --- _bridge_filter ---

def test_bridge_filter_keeps_ligands_above_threshold():
    results = [_sel_result("AAA", delta=1.0), _sel_result("BBB", delta=-0.1)]
    warnings: list[str] = []
    out = _bridge_filter(results, min_delta_log_k=0.5, top_n=3, warnings=warnings)
    assert len(out) == 1
    assert out[0].ligand_smiles == "AAA"
    assert warnings == []


def test_bridge_filter_respects_top_n_cap():
    results = [_sel_result(f"S{i}", delta=float(i + 1)) for i in range(5)]
    warnings: list[str] = []
    out = _bridge_filter(results, min_delta_log_k=0.0, top_n=2, warnings=warnings)
    assert len(out) == 2


def test_bridge_filter_fallback_when_all_below_threshold():
    results = [_sel_result("AAA", delta=-0.5)]
    warnings: list[str] = []
    out = _bridge_filter(results, min_delta_log_k=1.0, top_n=3, warnings=warnings)
    assert out[0].ligand_smiles == "AAA"
    assert len(warnings) == 1
    assert "unconditionally" in warnings[0]


def test_bridge_filter_empty_results_returns_empty():
    warnings: list[str] = []
    out = _bridge_filter([], min_delta_log_k=0.0, top_n=3, warnings=warnings)
    assert out == []
