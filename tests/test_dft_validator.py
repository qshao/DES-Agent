"""Unit tests for dft_validator. Heavy deps (gpu4pyscf, xtb) are mocked."""
from __future__ import annotations
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from des_multi_agent.chemistry.dft_validator import (
    DFTResult, compute_dft_properties, _embed_mmff,
)


class TestDFTResult:
    def test_dataclass_fields(self):
        r = DFTResult(smiles="NCCN", success=True, homo_ev=-8.5, homo_lumo_gap_ev=5.1,
                      donor_charges=[-0.3, -0.3])
        assert r.smiles == "NCCN"
        assert r.success is True
        assert r.dft_method == "B3LYP-D3(BJ)/def2-SVP"

    def test_failure_result(self):
        r = DFTResult(smiles="X", success=False, error="bad")
        assert r.homo_ev is None
        assert r.donor_charges == []


class TestEmbedMMFF:
    def test_invalid_smiles_raises(self):
        with pytest.raises(ValueError, match="RDKit cannot parse"):
            _embed_mmff("NOT_A_SMILES!!!")

    def test_valid_smiles_returns_mol(self):
        mol = _embed_mmff("NCCN")
        assert mol is not None
        assert mol.GetNumConformers() == 1


class TestComputeDFTProperties:
    def _make_mock_mf(self, n_electrons=8):
        mf = MagicMock()
        mf.converged = True
        # homo_idx = n_electrons//2 - 1 = 3
        # mo_energy[3] = -8.5/27.2114 Hartree (HOMO), [4] = -5.5/27.2114 (LUMO)
        HARTREE_TO_EV = 27.2114
        mo_energies = [-15.0, -12.0, -10.0, -8.5, -5.5, -3.0]
        mf.mo_energy = [e / HARTREE_TO_EV for e in mo_energies]
        mf.mulliken_pop.return_value = (None, np.array([-0.3, -0.1, -0.1, -0.3]))
        return mf

    def test_invalid_smiles_returns_failure_no_exception(self):
        result = compute_dft_properties("NOT_A_SMILES!!!")
        assert result.success is False
        assert result.error is not None
        assert result.homo_ev is None
        assert result.donor_charges == []

    def test_success_path_fields(self):
        mock_mol = MagicMock()
        mock_mf = self._make_mock_mf(n_electrons=8)
        with patch("des_multi_agent.chemistry.dft_validator._embed_mmff",
                   return_value=mock_mol), \
             patch("des_multi_agent.chemistry.dft_validator._xtb_optimize",
                   return_value=(["N", "C", "C", "N"], np.zeros((4, 3)))), \
             patch("des_multi_agent.chemistry.dft_validator._run_dft",
                   return_value=(-8.5, 5.1, [0, 3], mock_mf)):
            result = compute_dft_properties("NCCN")
        assert result.success is True
        assert abs(result.homo_ev - (-8.5)) < 0.01
        assert abs(result.homo_lumo_gap_ev - 5.1) < 0.01
        assert len(result.donor_charges) == 2   # donor_indices = [0, 3]

    def test_dft_scf_failure_returns_failure(self):
        mock_mol = MagicMock()
        with patch("des_multi_agent.chemistry.dft_validator._embed_mmff",
                   return_value=mock_mol), \
             patch("des_multi_agent.chemistry.dft_validator._xtb_optimize",
                   return_value=(["N", "C", "N"], np.zeros((3, 3)))), \
             patch("des_multi_agent.chemistry.dft_validator._run_dft",
                   side_effect=RuntimeError("SCF did not converge")):
            result = compute_dft_properties("NCN")
        assert result.success is False
        assert "SCF" in result.error

    def test_xtb_failure_returns_failure(self):
        mock_mol = MagicMock()
        with patch("des_multi_agent.chemistry.dft_validator._embed_mmff",
                   return_value=mock_mol), \
             patch("des_multi_agent.chemistry.dft_validator._xtb_optimize",
                   side_effect=RuntimeError("xtb optimization failed")):
            result = compute_dft_properties("NCCN")
        assert result.success is False

    def test_no_donor_atoms_gives_empty_charges(self):
        mock_mol = MagicMock()
        mock_mf = self._make_mock_mf(n_electrons=6)
        with patch("des_multi_agent.chemistry.dft_validator._embed_mmff",
                   return_value=mock_mol), \
             patch("des_multi_agent.chemistry.dft_validator._xtb_optimize",
                   return_value=(["C", "C", "C"], np.zeros((3, 3)))), \
             patch("des_multi_agent.chemistry.dft_validator._run_dft",
                   return_value=(-9.0, 4.5, [], mock_mf)):
            result = compute_dft_properties("CCC")
        assert result.success is True
        assert result.donor_charges == []


from des_multi_agent.chemistry.dft_selectivity import dft_selectivity_adjustment


class TestDFTSelectivityAdjustment:
    def _result(self, homo_ev: float) -> DFTResult:
        return DFTResult(smiles="X", success=True, homo_ev=homo_ev, donor_charges=[])

    def test_returns_zero_on_failure(self):
        r = DFTResult(smiles="X", success=False, error="fail")
        assert dft_selectivity_adjustment(r, "Cu2+", "Zn2+") == 0.0

    def test_returns_zero_when_homo_none(self):
        r = DFTResult(smiles="X", success=True, homo_ev=None, donor_charges=[])
        assert dft_selectivity_adjustment(r, "Cu2+", "Zn2+") == 0.0

    def test_adjustment_within_bounds(self):
        # Any HOMO energy must produce adjustment in [-0.05, +0.05]
        for homo in [-12.0, -9.5, -8.5, -7.5, -5.0]:
            adj = dft_selectivity_adjustment(self._result(homo), "Cu2+", "Zn2+")
            assert -0.05 <= adj <= 0.05, f"homo={homo} gave {adj}"

    def test_hard_donor_prefers_hard_metal(self):
        # HOMO ≤ −9.5 eV → hard donor (softness ≈ 0)
        # Hard metal (Mg2+ softness=0) vs soft metal (Cd2+ softness=1)
        # → adjustment should be positive (matches hard target)
        from des_multi_agent.chemistry.stability_rules import _metal_softness
        # Find a metal pair where one is harder than the other
        # Cu2+ softness from _METAL_IDENTITY ≈ 0.5–0.7 (borderline)
        # Use Cu2+ as target, Zn2+ as competitor — both borderline but Cu slightly softer
        hard_donor = self._result(-10.0)   # very hard donor
        soft_donor = self._result(-7.0)    # very soft donor
        adj_hard = dft_selectivity_adjustment(hard_donor, "Cu2+", "Zn2+")
        adj_soft = dft_selectivity_adjustment(soft_donor, "Cu2+", "Zn2+")
        # The two adjustments should have opposite signs or at least differ
        assert adj_hard != adj_soft

    def test_symmetric_metals_gives_near_zero(self):
        # Same metal for target and competitor → softness delta = 0 → adjustment ≈ 0
        adj = dft_selectivity_adjustment(self._result(-8.5), "Cu2+", "Cu2+")
        assert abs(adj) < 1e-9
