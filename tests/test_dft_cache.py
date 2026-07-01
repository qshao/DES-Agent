"""Unit tests for dft_cache. compute_dft_properties is mocked; no gpu4pyscf/xtb required."""
from __future__ import annotations

from unittest.mock import patch

from des_multi_agent.chemistry.dft_cache import cached_compute_dft_properties
from des_multi_agent.chemistry.dft_validator import DFTResult


def _success(smiles: str) -> DFTResult:
    return DFTResult(smiles=smiles, success=True, homo_ev=-8.5, homo_lumo_gap_ev=5.1,
                      donor_charges=[-0.3, -0.3])


class TestCacheMissThenHit:
    def test_second_call_is_cache_hit(self, tmp_path):
        cache_path = tmp_path / "dft.sqlite3"
        with patch("des_multi_agent.chemistry.dft_cache.compute_dft_properties",
                   return_value=_success("NCCN")) as mock_compute:
            r1 = cached_compute_dft_properties("NCCN", pH=7.0, cache_path=cache_path)
            r2 = cached_compute_dft_properties("NCCN", pH=7.0, cache_path=cache_path)

        assert mock_compute.call_count == 1
        assert r1.from_cache is False
        assert r2.from_cache is True
        assert r2.homo_ev == r1.homo_ev


class TestFailuresNotCached:
    def test_failure_not_cached(self, tmp_path):
        cache_path = tmp_path / "dft.sqlite3"
        failure = DFTResult(smiles="X", success=False, error="SCF did not converge")
        with patch("des_multi_agent.chemistry.dft_cache.compute_dft_properties",
                   return_value=failure) as mock_compute:
            cached_compute_dft_properties("X", pH=7.0, cache_path=cache_path)
            cached_compute_dft_properties("X", pH=7.0, cache_path=cache_path)

        assert mock_compute.call_count == 2   # never cached -> recomputed both times


class TestCacheKeyIsSpeciesSmiles:
    def test_two_spellings_of_same_species_share_cache_entry(self, tmp_path):
        cache_path = tmp_path / "dft.sqlite3"
        # "CC(=O)O" and "OC(C)=O" are the same molecule (acetic acid) written
        # differently -> dominant_species canonicalizes both to one species_smiles.
        with patch("des_multi_agent.chemistry.dft_cache.compute_dft_properties",
                   return_value=_success("CC(=O)O")) as mock_compute:
            cached_compute_dft_properties("CC(=O)O", pH=7.4, cache_path=cache_path)
            cached_compute_dft_properties("OC(C)=O", pH=7.4, cache_path=cache_path)

        assert mock_compute.call_count == 1

    def test_same_input_different_ph_gives_separate_entries(self, tmp_path):
        cache_path = tmp_path / "dft.sqlite3"
        with patch("des_multi_agent.chemistry.dft_cache.compute_dft_properties",
                   return_value=_success("CC(=O)O")) as mock_compute:
            cached_compute_dft_properties("CC(=O)O", pH=2.0, cache_path=cache_path)   # neutral
            cached_compute_dft_properties("CC(=O)O", pH=7.4, cache_path=cache_path)   # deprotonated

        assert mock_compute.call_count == 2   # different species_smiles -> no false hit


class TestCacheFailureFallback:
    def test_corrupt_cache_file_falls_back_to_direct_call(self, tmp_path):
        cache_path = tmp_path / "dft.sqlite3"
        cache_path.write_text("not a valid sqlite file")
        with patch("des_multi_agent.chemistry.dft_cache.compute_dft_properties",
                   return_value=_success("NCCN")) as mock_compute:
            result = cached_compute_dft_properties("NCCN", pH=7.0, cache_path=cache_path)

        assert result.success is True
        assert mock_compute.call_count == 1


class TestCacheFileCreation:
    def test_cache_db_created_in_missing_parent_dir(self, tmp_path):
        cache_path = tmp_path / "nested" / "dir" / "dft.sqlite3"
        with patch("des_multi_agent.chemistry.dft_cache.compute_dft_properties",
                   return_value=_success("NCCN")):
            cached_compute_dft_properties("NCCN", pH=7.0, cache_path=cache_path)
        assert cache_path.exists()


class TestCacheHitReflectsCurrentCaller:
    def test_cache_hit_updates_smiles_and_ph_to_current_call(self, tmp_path):
        """Two spellings of same molecule share cache entry, but smiles/pH
        reflect the current caller, not the original writer."""
        cache_path = tmp_path / "dft.sqlite3"
        # First call with "CC(=O)O", second call with "OC(C)=O" (same molecule, different spelling)
        with patch("des_multi_agent.chemistry.dft_cache.compute_dft_properties",
                   return_value=_success("CC(=O)O")) as mock_compute:
            r1 = cached_compute_dft_properties("CC(=O)O", pH=7.4, cache_path=cache_path)
            r2 = cached_compute_dft_properties("OC(C)=O", pH=7.4, cache_path=cache_path)

        assert mock_compute.call_count == 1  # both share cache entry
        assert r1.from_cache is False
        assert r2.from_cache is True
        # r1's smiles reflects its own input
        assert r1.smiles == "CC(=O)O"
        # r2's smiles reflects ITS OWN input, not the original writer's
        assert r2.smiles == "OC(C)=O"
        # But the species_smiles (cache key) should remain canonical
        assert r2.species_smiles == r1.species_smiles
