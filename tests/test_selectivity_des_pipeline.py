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


from unittest.mock import patch, call
from des_multi_agent.workflows.selectivity_des_pipeline import run_selectivity_des_pipeline


def _make_multi_cycle_outcome(is_des: bool):
    """Return a minimal MultiCycleOutcome mock."""
    dr = MagicMock()
    dr.is_des = is_des
    dr.min_tm_k = 280.0
    dr.eutectic_ratio_b = 0.5
    dr.rationale = "test"
    dr.curve = MagicMock()
    dr.curve.smiles_b = "CCO"

    search_outcome = MagicMock()
    search_outcome.results = [dr]

    cycle_delta = MagicMock()
    cycle_delta.n_screened = 5

    mco = MagicMock()
    mco.final_outcome = search_outcome
    mco.cycle_deltas = [cycle_delta]
    return mco


@patch("des_multi_agent.workflows.selectivity_des_pipeline.run_multi_cycle_search")
@patch("des_multi_agent.workflows.selectivity_des_pipeline.run_metal_selectivity_screen")
def test_pipeline_returns_outcome_with_correct_shape(mock_sel, mock_des):
    mock_sel.return_value = _sel_outcome(["NCC(=O)O", "NCCN"])
    mock_des.return_value = _make_multi_cycle_outcome(is_des=True)
    outcome = run_selectivity_des_pipeline(
        target_metal="Cu2+",
        competitor_metal="Zn2+",
        checkpoint_path="/fake/ckpt.pt",
        n_outer_cycles=1,
        top_ligands=2,
    )
    assert outcome.target_metal == "Cu2+"
    assert outcome.competitor_metal == "Zn2+"
    assert len(outcome.ligand_des_results) == 2
    assert outcome.n_outer_cycles_run == 1
    assert not outcome.converged


@patch("des_multi_agent.workflows.selectivity_des_pipeline.run_multi_cycle_search")
@patch("des_multi_agent.workflows.selectivity_des_pipeline.run_metal_selectivity_screen")
def test_pipeline_converges_when_des_compatible_set_stable(mock_sel, mock_des):
    mock_sel.return_value = _sel_outcome(["NCC(=O)O"])
    mock_des.return_value = _make_multi_cycle_outcome(is_des=True)
    outcome = run_selectivity_des_pipeline(
        target_metal="Cu2+",
        competitor_metal="Zn2+",
        checkpoint_path="/fake/ckpt.pt",
        n_outer_cycles=3,
        top_ligands=1,
    )
    assert outcome.converged
    assert outcome.n_outer_cycles_run == 2  # stable after pass 2


@patch("des_multi_agent.workflows.selectivity_des_pipeline.run_multi_cycle_search")
@patch("des_multi_agent.workflows.selectivity_des_pipeline.run_metal_selectivity_screen")
def test_pipeline_runs_all_outer_cycles_when_set_changes(mock_sel, mock_des):
    # Alternate DES compatibility so the set never stabilises
    mock_sel.side_effect = [
        _sel_outcome(["NCC(=O)O"]),
        _sel_outcome(["NCCN"]),
    ]
    mock_des.return_value = _make_multi_cycle_outcome(is_des=True)
    outcome = run_selectivity_des_pipeline(
        target_metal="Cu2+",
        competitor_metal="Zn2+",
        checkpoint_path="/fake/ckpt.pt",
        n_outer_cycles=2,
        top_ligands=1,
    )
    assert not outcome.converged
    assert outcome.n_outer_cycles_run == 2


@patch("des_multi_agent.workflows.selectivity_des_pipeline.run_multi_cycle_search")
@patch("des_multi_agent.workflows.selectivity_des_pipeline.run_metal_selectivity_screen")
def test_pipeline_des_failure_adds_warning_and_continues(mock_sel, mock_des):
    mock_sel.return_value = _sel_outcome(["NCC(=O)O", "NCCN"])
    mock_des.side_effect = [
        RuntimeError("model unavailable"),
        _make_multi_cycle_outcome(is_des=True),
    ]
    outcome = run_selectivity_des_pipeline(
        target_metal="Cu2+",
        competitor_metal="Zn2+",
        checkpoint_path="/fake/ckpt.pt",
        n_outer_cycles=1,
        top_ligands=2,
    )
    assert any("DES search failed" in w for w in outcome.warnings)
    assert len(outcome.ligand_des_results) == 2
    assert outcome.ligand_des_results[0].des_compatible is False
    assert outcome.ligand_des_results[1].des_compatible is True


@patch("des_multi_agent.workflows.selectivity_des_pipeline.run_multi_cycle_search")
@patch("des_multi_agent.workflows.selectivity_des_pipeline.run_metal_selectivity_screen")
def test_pipeline_passes_des_hints_on_second_outer_cycle(mock_sel, mock_des):
    mock_sel.return_value = _sel_outcome(["NCC(=O)O"])
    mock_des.return_value = _make_multi_cycle_outcome(is_des=True)
    run_selectivity_des_pipeline(
        target_metal="Cu2+",
        competitor_metal="Zn2+",
        checkpoint_path="/fake/ckpt.pt",
        n_outer_cycles=2,
        top_ligands=1,
    )
    # outer cycle 1: no hints yet
    first_call_kwargs = mock_sel.call_args_list[0][1]
    assert first_call_kwargs.get("des_compatible_hints") is None
    # outer cycle 2: compatible hint present (set was {"NCC(=O)O"})
    second_call_kwargs = mock_sel.call_args_list[1][1]
    assert "NCC(=O)O" in second_call_kwargs.get("des_compatible_hints", [])


from des_multi_agent.reporting import format_selectivity_des_report


def _make_pipeline_outcome(des_compatible: bool = True) -> SelectivityDesPipelineOutcome:
    dr = MagicMock()
    dr.is_des = des_compatible
    dr.min_tm_k = 287.1
    dr.eutectic_ratio_b = 0.33
    dr.rationale = "min Tm=287.1 K"
    dr.curve = MagicMock()
    dr.curve.smiles_b = "CC(=O)NCCO"

    ligand = _sel_result("NCC(=O)O", delta=1.0, score=7.5)
    ldr = LigandDesResult(
        ligand=ligand,
        des_results=[dr] if des_compatible else [],
        n_des_screened=10,
        des_compatible=des_compatible,
    )
    sel_out = _sel_outcome(["NCC(=O)O", "NCCN"])
    return SelectivityDesPipelineOutcome(
        target_metal="Cu2+",
        competitor_metal="Zn2+",
        selectivity_outcome=sel_out,
        ligand_des_results=[ldr],
        n_outer_cycles_run=2,
        converged=True,
        warnings=[],
    )


def test_report_contains_section_1_header():
    report = format_selectivity_des_report(_make_pipeline_outcome())
    assert "Section 1: Selectivity Results" in report


def test_report_contains_section_2_header():
    report = format_selectivity_des_report(_make_pipeline_outcome())
    assert "Section 2: DES Partners" in report


def test_report_section_1_has_des_compatible_column():
    report = format_selectivity_des_report(_make_pipeline_outcome(des_compatible=True))
    assert "des_compatible" in report
    assert "| yes" in report


def test_report_section_2_shows_des_partner_when_compatible():
    report = format_selectivity_des_report(_make_pipeline_outcome(des_compatible=True))
    assert "CC(=O)NCCO" in report
    assert "DES-compatible: YES" in report


def test_report_section_2_shows_no_partners_when_incompatible():
    report = format_selectivity_des_report(_make_pipeline_outcome(des_compatible=False))
    assert "No DES partners found" in report
    assert "DES-compatible: NO" in report


def test_report_pipeline_outcome_none_selectivity_does_not_crash():
    outcome = SelectivityDesPipelineOutcome(
        target_metal="Cu2+",
        competitor_metal="Zn2+",
        selectivity_outcome=None,
        ligand_des_results=[],
        n_outer_cycles_run=0,
        converged=False,
    )
    report = format_selectivity_des_report(outcome)
    assert "Selectivity-DES Pipeline" in report


@patch("des_multi_agent.workflows.selectivity_des_pipeline.run_multi_cycle_search")
@patch("des_multi_agent.workflows.selectivity_des_pipeline.run_metal_selectivity_screen")
def test_pipeline_passes_des_incompatible_hints_on_second_outer_cycle(mock_sel, mock_des):
    mock_sel.return_value = _sel_outcome(["NCC(=O)O", "NCCN"])
    # First ligand is DES-compatible, second is not
    mock_des.side_effect = [
        _make_multi_cycle_outcome(is_des=True),   # NCC(=O)O — compatible
        _make_multi_cycle_outcome(is_des=False),  # NCCN — incompatible
        _make_multi_cycle_outcome(is_des=True),   # NCC(=O)O cycle 2
        _make_multi_cycle_outcome(is_des=False),  # NCCN cycle 2
    ]
    run_selectivity_des_pipeline(
        target_metal="Cu2+",
        competitor_metal="Zn2+",
        checkpoint_path="/fake/ckpt.pt",
        n_outer_cycles=2,
        top_ligands=2,
    )
    second_call_kwargs = mock_sel.call_args_list[1][1]
    assert "NCCN" in second_call_kwargs.get("des_incompatible_hints", [])
    assert "NCC(=O)O" in second_call_kwargs.get("des_compatible_hints", [])


from unittest.mock import patch as _patch
from des_multi_agent.cli import build_parser, main as cli_main


def test_cli_selectivity_des_routes_to_pipeline(tmp_path):
    fake_ckpt = tmp_path / "ckpt.pt"
    fake_ckpt.write_text("x")
    fake_outcome = _make_pipeline_outcome()
    with _patch(
        "des_multi_agent.cli.run_selectivity_des_pipeline",
        return_value=fake_outcome,
    ) as mock_run, _patch("des_multi_agent.cli.format_selectivity_des_report", return_value="REPORT"):
        cli_main([
            "--workflow", "selectivity-des",
            "--target-metal-ion", "Cu2+",
            "--competitor-metal-ion", "Zn2+",
            "--checkpoint-path", str(fake_ckpt),
        ])
    mock_run.assert_called_once()
    kwargs = mock_run.call_args[1]
    assert kwargs["target_metal"] == "Cu2+"
    assert kwargs["competitor_metal"] == "Zn2+"


def test_cli_selectivity_des_requires_target_metal_ion():
    with pytest.raises(SystemExit):
        cli_main([
            "--workflow", "selectivity-des",
            "--competitor-metal-ion", "Zn2+",
            "--checkpoint-path", "/fake/ckpt.pt",
        ])


def test_cli_selectivity_des_requires_checkpoint_path():
    with pytest.raises(SystemExit):
        cli_main([
            "--workflow", "selectivity-des",
            "--target-metal-ion", "Cu2+",
            "--competitor-metal-ion", "Zn2+",
        ])


def test_cli_metal_selectivity_workflow_unchanged():
    """Existing metal-selectivity workflow must still parse without error."""
    parser = build_parser()
    args = parser.parse_args([
        "--workflow", "metal-selectivity",
        "--target-metal-ion", "Cu2+",
        "--competitor-metal-ion", "Zn2+",
    ])
    assert args.workflow == "metal-selectivity"
