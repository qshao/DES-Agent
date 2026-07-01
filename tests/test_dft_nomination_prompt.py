"""Tests for DFT nomination prompt and fallback nomination."""
from __future__ import annotations
import pytest
from unittest.mock import MagicMock
from des_multi_agent.llm.prompts import dft_nomination_prompt
from des_multi_agent.llm.base import nominate_for_dft_fallback


def _make_candidate(smiles: str, delta: float, score: float):
    c = MagicMock()
    c.ligand_smiles = smiles
    c.delta_log_k = delta
    c.composite_score = score
    return c


CANDIDATES = [
    _make_candidate("NCCN", 1.5, 0.90),
    _make_candidate("NCC(=O)O", 1.1, 0.80),
    _make_candidate("c1ccncc1", 0.4, 0.65),
    _make_candidate("NCCCCN", 0.2, 0.55),
]


class TestDFTNominationPrompt:
    def test_prompt_contains_target_metal(self):
        p = dft_nomination_prompt(CANDIDATES, "Cu2+", "Zn2+")
        assert "Cu2+" in p

    def test_prompt_contains_competitor_metal(self):
        p = dft_nomination_prompt(CANDIDATES, "Cu2+", "Zn2+")
        assert "Zn2+" in p

    def test_prompt_contains_all_smiles(self):
        p = dft_nomination_prompt(CANDIDATES, "Cu2+", "Zn2+")
        for c in CANDIDATES:
            assert c.ligand_smiles in p

    def test_prompt_contains_delta_log_k(self):
        p = dft_nomination_prompt(CANDIDATES, "Cu2+", "Zn2+")
        assert "1.50" in p or "ΔlogK" in p

    def test_top_n_in_prompt(self):
        p = dft_nomination_prompt(CANDIDATES, "Cu2+", "Zn2+", top_n=2)
        assert "1–2" in p or "2" in p

    def test_json_instruction_in_prompt(self):
        p = dft_nomination_prompt(CANDIDATES, "Cu2+", "Zn2+")
        assert "JSON" in p


class TestNominateForDFTFallback:
    def test_returns_top_n_by_score(self):
        result = nominate_for_dft_fallback(CANDIDATES, top_n=2)
        assert result == ["NCCN", "NCC(=O)O"]

    def test_respects_top_n_cap(self):
        result = nominate_for_dft_fallback(CANDIDATES, top_n=1)
        assert result == ["NCCN"]

    def test_returns_all_if_fewer_than_n(self):
        result = nominate_for_dft_fallback(CANDIDATES[:2], top_n=5)
        assert len(result) == 2

    def test_empty_candidates_returns_empty(self):
        result = nominate_for_dft_fallback([], top_n=3)
        assert result == []
