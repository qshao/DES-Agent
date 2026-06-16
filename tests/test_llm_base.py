"""Tests for structural facts injection in BaseLLMProvider methods (Phase 3)."""
from __future__ import annotations

import pytest
import des_multi_agent.llm.base as base_mod


class FakeProvider(base_mod.BaseLLMProvider):
    """Minimal stub that captures the prompt passed to _request."""

    def _request(self, prompt: str) -> str:
        self._last_prompt = prompt
        return self._fake_response

    def extract_text(self, raw: str) -> str:
        return raw

    request_profile = None  # not needed for tests


def _make_provider(fake_response: str) -> FakeProvider:
    provider = FakeProvider.__new__(FakeProvider)
    provider._fake_response = fake_response
    return provider


def test_review_candidate_injects_facts_block(monkeypatch):
    """_request receives a prompt containing 'Computed facts:' when candidate SMILES is valid."""
    captured = {}

    class CapturingProvider(base_mod.BaseLLMProvider):
        def _request(self, prompt):
            captured["prompt"] = prompt
            return (
                '{"smiles": "NC(N)=O", "decision": "keep", "confidence": 0.9, '
                '"rationale": "ok", "notes": []}'
            )

        def extract_text(self, raw):
            return raw

        request_profile = None

    provider = CapturingProvider.__new__(CapturingProvider)
    provider.review_candidate("C[N+](C)(C)CCO.[Cl-]", "NC(N)=O", "ctx")
    assert "Computed facts:" in captured["prompt"]


def test_review_candidate_facts_block_before_candidate_line(monkeypatch):
    """Computed facts appear before the 'Candidate:' line in the prompt."""
    captured = {}

    class CapturingProvider(base_mod.BaseLLMProvider):
        def _request(self, prompt):
            captured["prompt"] = prompt
            return (
                '{"smiles": "NC(N)=O", "decision": "keep", "confidence": 0.9, '
                '"rationale": "ok", "notes": []}'
            )

        def extract_text(self, raw):
            return raw

        request_profile = None

    provider = CapturingProvider.__new__(CapturingProvider)
    provider.review_candidate("C[N+](C)(C)CCO.[Cl-]", "NC(N)=O", "ctx")
    prompt = captured["prompt"]
    assert prompt.index("Computed facts:") < prompt.index("Candidate:")


def test_assess_candidate_chemistry_injects_facts_block():
    """assess_candidate_chemistry passes structural facts to the prompt."""
    captured = {}

    class CapturingProvider(base_mod.BaseLLMProvider):
        def _request(self, prompt):
            captured["prompt"] = prompt
            return '[{"smiles":"NC(N)=O","decision":"keep","confidence":0.9,"rationale":"ok","warnings":[]}]'

        def extract_text(self, raw):
            return raw

        request_profile = None

    provider = CapturingProvider.__new__(CapturingProvider)
    provider.assess_candidate_chemistry("NC(N)=O", "ctx")
    assert "Computed facts:" in captured["prompt"]


def test_detect_contradictions_no_facts_block():
    """detect_contradictions passes facts_block='' (empty) as specified."""
    captured = {}

    class CapturingProvider(base_mod.BaseLLMProvider):
        def _request(self, prompt):
            captured["prompt"] = prompt
            return "[]"

        def extract_text(self, raw):
            return raw

        request_profile = None

    provider = CapturingProvider.__new__(CapturingProvider)
    provider.detect_contradictions([], "ctx")
    # Per spec: facts_block="" for contradictions; no computed facts injected
    assert "Computed facts:" not in captured["prompt"]
