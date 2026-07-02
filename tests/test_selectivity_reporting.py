"""Tests for multi-off-target report formatting."""
from __future__ import annotations

from des_multi_agent.reporting import format_metal_selectivity_report
from des_multi_agent.workflows.metal_binding_selectivity import (
    SelectivityResult,
    SelectivityScreenOutcome,
)


def _result(smiles, log_k_target, log_k_competitors, worst_metal):
    worst_val = log_k_competitors[worst_metal]
    return SelectivityResult(
        ligand_smiles=smiles, log_k_target=log_k_target, log_k_competitor=worst_val,
        delta_log_k=log_k_target - worst_val, composite_score=log_k_target - worst_val,
        source="heuristic", source_id="s", rationale="r",
        log_k_competitors=log_k_competitors, worst_competitor_metal=worst_metal,
    )


def test_report_single_competitor_byte_identical_header():
    outcome = SelectivityScreenOutcome(
        target_metal="Cu2+", competitor_metals=["Zn2+"],
        results=[_result("NCCN", 10.0, {"Zn2+": 6.0}, "Zn2+")],
        n_screened=1, n_cycles=1,
    )
    report = format_metal_selectivity_report(outcome)
    assert "=== Metal Selectivity Screen: Cu2+ over Zn2+ ===" in report
    assert "off_target_breakdown" not in report


def test_report_multi_competitor_header_lists_all_off_targets():
    outcome = SelectivityScreenOutcome(
        target_metal="Cu2+", competitor_metals=["Zn2+", "Fe3+"],
        results=[_result("NCCN", 10.0, {"Zn2+": 6.0, "Fe3+": 9.0}, "Fe3+")],
        n_screened=1, n_cycles=1,
    )
    report = format_metal_selectivity_report(outcome)
    assert "=== Metal Selectivity Screen: Cu2+ over Zn2+, Fe3+ ===" in report


def test_report_multi_competitor_breakdown_column_marks_worst_case():
    outcome = SelectivityScreenOutcome(
        target_metal="Cu2+", competitor_metals=["Zn2+", "Fe3+"],
        results=[_result("NCCN", 10.0, {"Zn2+": 6.0, "Fe3+": 9.0}, "Fe3+")],
        n_screened=1, n_cycles=1,
    )
    report = format_metal_selectivity_report(outcome)
    assert "off_target_breakdown" in report
    assert "Zn2+=6.00" in report
    assert "Fe3+=9.00*" in report
