from __future__ import annotations

from abc import ABC, abstractmethod

from ..evaluation import DesResult
from .schemas import CandidateBrainstorm, CritiqueNote, ExplanationNote


class LLMProvider(ABC):
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
