
from __future__ import annotations

from abc import ABC, abstractmethod

from ..evaluation import DesResult
from .schemas import CandidateBrainstorm, CandidateReview, ContradictionNote, CritiqueNote, ExplanationNote


class LLMProvider(ABC):
    @abstractmethod
    def route_request(self, request: str, normalized=None) -> str:
        raise NotImplementedError

    @abstractmethod
    def review_candidate(self, component_a: str, candidate_smiles: str, context: str) -> CandidateReview:
        raise NotImplementedError

    @abstractmethod
    def brainstorm_candidates(
        self,
        component_a: str,
        constraints: dict | None,
        context: str,
    ) -> list[CandidateBrainstorm]:
        raise NotImplementedError

    @abstractmethod
    def generate_explanations(self, results: list[DesResult], context: str) -> list[ExplanationNote]:
        raise NotImplementedError

    @abstractmethod
    def critique_results(self, results: list[DesResult], context: str) -> list[CritiqueNote]:
        raise NotImplementedError

    @abstractmethod
    def detect_contradictions(self, results: list[DesResult], context: str) -> list[ContradictionNote]:
        raise NotImplementedError
