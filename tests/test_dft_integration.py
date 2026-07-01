"""Integration tests for the DFT stage wired into run_metal_selectivity_screen.

All heavy deps are mocked: compute_dft_properties returns a controlled DFTResult,
predict_log_k returns a mock object with .value = 5.0.
"""
from __future__ import annotations
import inspect
from unittest.mock import patch, MagicMock

from des_multi_agent.chemistry.dft_validator import DFTResult
from des_multi_agent.workflows.metal_binding_selectivity import (
    SelectivityScreenOutcome,
    run_metal_selectivity_screen,
)


FAKE_DFT_SUCCESS = DFTResult(
    smiles="NCCN", success=True, homo_ev=-8.5, homo_lumo_gap_ev=5.1,
    donor_charges=[-0.3, -0.3],
)
FAKE_DFT_FAIL = DFTResult(smiles="c1ccncc1", success=False, error="no donor atoms")


def _mock_log_k(*args, **kwargs):
    """Drop-in for predict_log_k — returns an object with .value = 5.0."""
    return MagicMock(value=5.0)


class TestSelectivityScreenOutcomeHasDFTField:
    def test_dft_results_field_exists_and_defaults_empty(self):
        outcome = SelectivityScreenOutcome(
            target_metal="Cu2+",
            competitor_metal="Zn2+",
            results=[],
            n_screened=0,
            n_cycles=1,
        )
        assert hasattr(outcome, "dft_results")
        assert outcome.dft_results == {}


class TestRunMetalSelectivityScreenDFTParam:
    def test_accepts_dft_validate_param(self):
        """run_metal_selectivity_screen must accept dft_validate and dft_top_n."""
        sig = inspect.signature(run_metal_selectivity_screen)
        assert "dft_validate" in sig.parameters
        assert "dft_top_n" in sig.parameters
        assert sig.parameters["dft_validate"].default is False
        assert sig.parameters["dft_top_n"].default == 3


class TestDFTStageWiring:
    """Verifies the DFT stage by mocking predict_log_k and compute_dft_properties."""

    def _run(self, dft_validate: bool, fake_dft_result: DFTResult | None = None,
             dft_top_n: int = 1) -> SelectivityScreenOutcome:
        """Run selectivity screen with only the heavy deps mocked."""
        chosen = fake_dft_result if fake_dft_result is not None else FAKE_DFT_SUCCESS

        with (
            patch(
                "des_multi_agent.workflows.metal_binding_selectivity.predict_log_k",
                side_effect=_mock_log_k,
            ),
            patch(
                "des_multi_agent.chemistry.dft_validator.compute_dft_properties",
                side_effect=lambda smi: chosen,
            ),
        ):
            return run_metal_selectivity_screen(
                target_metal="Cu2+",
                competitor_metal="Zn2+",
                n=3,
                model_path=None,
                llm_provider=None,
                n_cycles=1,
                dft_validate=dft_validate,
                dft_top_n=dft_top_n,
            )

    def test_dft_results_populated_on_validate(self):
        outcome = self._run(dft_validate=True, fake_dft_result=FAKE_DFT_SUCCESS)
        assert isinstance(outcome.dft_results, dict)
        # At least one entry should be present when DFT succeeds
        assert len(outcome.dft_results) > 0

    def test_dft_false_leaves_dft_results_empty(self):
        outcome = self._run(dft_validate=False)
        assert outcome.dft_results == {}

    def test_dft_failure_adds_warning(self):
        outcome = self._run(dft_validate=True, fake_dft_result=FAKE_DFT_FAIL)
        dft_warnings = [w for w in outcome.warnings if "[DFT]" in w]
        assert len(dft_warnings) > 0, "Expected at least one [DFT] warning"
        assert any("Warning" in w for w in dft_warnings)
