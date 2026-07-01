"""Tests for CLI DFT flags and report rendering with DFT results."""
from __future__ import annotations
import pytest
from unittest.mock import MagicMock, patch
from des_multi_agent.chemistry.dft_validator import DFTResult
from des_multi_agent.workflows.metal_binding_selectivity import SelectivityResult, SelectivityScreenOutcome
from des_multi_agent.reporting import format_metal_selectivity_report


def _make_outcome(dft_results: dict | None = None) -> SelectivityScreenOutcome:
    r = SelectivityResult(
        ligand_smiles="NCCN", log_k_target=5.5, log_k_competitor=4.0,
        delta_log_k=1.5, composite_score=0.85, source="test", source_id="", rationale="good",
    )
    return SelectivityScreenOutcome(
        target_metal="Cu2+", competitor_metals=["Zn2+"],
        results=[r], n_screened=1, n_cycles=1,
        dft_results=dft_results or {},
    )


class TestReportWithoutDFT:
    def test_no_dft_columns_when_no_dft_results(self):
        report = format_metal_selectivity_report(_make_outcome())
        assert "dft_homo_ev" not in report
        assert "DFT validation" not in report

    def test_report_renders_without_error(self):
        report = format_metal_selectivity_report(_make_outcome())
        assert "Cu2+" in report
        assert "NCCN" in report


class TestReportWithDFT:
    def _outcome_with_dft(self):
        dft = DFTResult(smiles="NCCN", success=True, homo_ev=-8.51,
                        homo_lumo_gap_ev=5.12, donor_charges=[-0.31, -0.29])
        return _make_outcome({"NCCN": dft})

    def test_homo_ev_appears_in_report(self):
        report = format_metal_selectivity_report(self._outcome_with_dft())
        assert "-8.51" in report

    def test_dft_summary_block_present(self):
        report = format_metal_selectivity_report(self._outcome_with_dft())
        assert "DFT validation" in report or "B3LYP" in report

    def test_failed_dft_shows_dash_not_crash(self):
        dft_fail = DFTResult(smiles="NCCN", success=False, error="SCF fail")
        outcome = _make_outcome({"NCCN": dft_fail})
        report = format_metal_selectivity_report(outcome)
        assert "—" in report or "FAILED" in report

    def test_non_nominated_row_shows_dash(self):
        r2 = SelectivityResult(
            ligand_smiles="NCC(=O)O", log_k_target=4.0, log_k_competitor=3.5,
            delta_log_k=0.5, composite_score=0.70, source="test", source_id="", rationale="ok",
        )
        dft = DFTResult(smiles="NCCN", success=True, homo_ev=-8.5,
                        homo_lumo_gap_ev=5.0, donor_charges=[-0.3])
        outcome = _make_outcome({"NCCN": dft})
        outcome = SelectivityScreenOutcome(
            target_metal="Cu2+", competitor_metals=["Zn2+"],
            results=[outcome.results[0], r2], n_screened=2, n_cycles=1,
            dft_results={"NCCN": dft},
        )
        report = format_metal_selectivity_report(outcome)
        assert "—" in report   # NCC(=O)O has no DFT result


class TestCLIDFTFlags:
    def test_dft_validate_flag_exists(self):
        from des_multi_agent.cli import build_parser
        parser = build_parser()
        # parse a valid metal-selectivity command with --dft-validate
        args = parser.parse_args([
            "--workflow", "metal-selectivity",
            "--target-metal-ion", "Cu2+",
            "--competitor-metal-ion", "Zn2+",
            "--dft-validate",
            "--dft-top-n", "2",
        ])
        assert args.dft_validate is True
        assert args.dft_top_n == 2

    def test_dft_top_n_default_is_3(self):
        from des_multi_agent.cli import build_parser
        parser = build_parser()
        args = parser.parse_args([
            "--workflow", "metal-selectivity",
            "--target-metal-ion", "Cu2+",
            "--competitor-metal-ion", "Zn2+",
        ])
        assert args.dft_top_n == 3
