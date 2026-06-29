
from __future__ import annotations

from abc import ABC, abstractmethod

from ..evaluation import DesResult
from .schemas import CandidateBrainstorm, CandidateFamily, CandidateReview, ChemistryAssessment, ChemistryNextStep, ContradictionNote, CritiqueNote, ExplanationNote


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
        *,
        max_families: int = 6,
        diversity_mode: str = "balanced",
        family_bias_strength: float = 0.5,
        prior_productive_families: dict[str, int] | None = None,
    ) -> list[CandidateBrainstorm]:
        raise NotImplementedError

    @abstractmethod
    def select_candidate_families(
        self,
        component_a: str,
        constraints: dict | None,
        context: str,
        *,
        max_families: int = 6,
        diversity_mode: str = "balanced",
        family_bias_strength: float = 0.5,
        prior_productive_families: dict[str, int] | None = None,
    ) -> list[CandidateFamily]:
        raise NotImplementedError

    @abstractmethod
    def generate_explanations(self, results: list[DesResult], context: str) -> list[ExplanationNote]:
        raise NotImplementedError

    @abstractmethod
    def critique_results(self, results: list[DesResult], context: str) -> list[CritiqueNote]:
        raise NotImplementedError

    @abstractmethod
    def detect_contradictions(self, results: list[DesResult], context: str, facts_block: str = "") -> list[ContradictionNote]:
        raise NotImplementedError

    @abstractmethod
    def assess_candidate_chemistry(
        self,
        candidate_smiles: str,
        context: str,
        memory_notes: list[str] | None = None,
    ) -> list[ChemistryAssessment]:
        raise NotImplementedError

    @abstractmethod
    def suggest_next_steps(
        self,
        context: str,
        memory_notes: list[str] | None = None,
    ) -> list[ChemistryNextStep]:
        raise NotImplementedError
