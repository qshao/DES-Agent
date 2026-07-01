
from __future__ import annotations

import os
import sys
from urllib.parse import urlencode, urlsplit, urlunsplit, parse_qsl

from ..evaluation import DesResult
from ..chemistry.claim_grounding import structural_facts
from ..chemistry.hbond import hbond_profile
from ..chemistry.partner_registry import known_ligand_menu as _known_ligand_menu, known_partner_menu
from .client import post_json_chat
from .parser import parse_candidate_brainstorms, parse_candidate_families, parse_candidate_review, parse_chemistry_assessments, parse_chemistry_next_steps, parse_contradiction_notes, parse_critique_notes, parse_explanation_notes, parse_ligand_families
from .prompts import candidate_brainstorm_prompt, candidate_review_prompt, chemistry_assessment_prompt, chemistry_next_step_prompt, contradiction_prompt, critique_prompt, explanation_prompt, family_selection_prompt, ligand_brainstorm_prompt, ligand_family_selection_prompt, ligand_review_prompt, ligand_selectivity_brainstorm_prompt
from ..task_router_prompts import task_router_prompt
from .provider import LLMProvider
from .schemas import CandidateBrainstorm, CandidateFamily, CandidateReview, ChemistryAssessment, ChemistryNextStep, ContradictionNote, CritiqueNote, ExplanationNote, LigandFamily
from .specs import RequestProfile
from .transport import RequestTransport


def nominate_for_dft_fallback(candidates: list, top_n: int) -> list[str]:
    """Return top-top_n SMILES by composite_score when no LLM is available."""
    sorted_cands = sorted(candidates, key=lambda r: r.composite_score, reverse=True)
    return [r.ligand_smiles for r in sorted_cands[:top_n]]


def _complementary_role(role: str) -> str:
    """Partner role that complements a component with the given H-bond role."""
    if role == "HBA":
        return "HBD"
    if role == "HBD":
        return "HBA"
    return "amphoteric"


class BaseLLMProvider(LLMProvider):
    request_profile: RequestProfile

    def __init__(
        self,
        *,
        model_name: str,
        api_base_url: str,
        api_key_env: str | None = None,
        max_candidates: int = 20,
        max_tokens: int = 512,
        temperature: float = 0.2,
        timeout_seconds: float = 30.0,
        diversity_mode: str = "balanced",
        max_families: int = 6,
        family_bias_strength: float = 0.5,
        request_fn=post_json_chat,
    ):
        self.model_name = model_name
        self.api_base_url = api_base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.max_candidates = max_candidates
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds
        self.diversity_mode = diversity_mode
        self.max_families = max_families
        self.family_bias_strength = family_bias_strength
        self.transport = RequestTransport(request_fn=request_fn, timeout_seconds=timeout_seconds)

    def _request(self, prompt: str) -> str:
        api_key = os.getenv(self.api_key_env) if self.api_key_env else None
        raw = self.transport.post_json(
            self.request_url(api_key),
            self.build_payload(prompt),
            api_key=api_key,
            include_api_key_in_header=self.request_profile.api_key_in_header,
        )
        return self.extract_text(raw)

    def route_request(self, request: str, normalized=None) -> str:
        return self._request(task_router_prompt(request, normalized=normalized))

    def review_candidate(self, component_a: str, candidate_smiles: str, context: str) -> CandidateReview:
        facts_block = structural_facts(candidate_smiles).as_prompt_block()
        raw = self._request(candidate_review_prompt(component_a, candidate_smiles, context, facts_block=facts_block))
        review = parse_candidate_review(raw)
        return review

    def brainstorm_candidates(
        self,
        component_a: str,
        constraints: dict | None,
        context: str,
        *,
        max_families: int | None = None,
        diversity_mode: str | None = None,
        family_bias_strength: float | None = None,
        prior_productive_families: dict[str, int] | None = None,
    ) -> list[CandidateBrainstorm]:
        families: list[CandidateFamily] = []
        effective_max_families = getattr(self, "max_families", 6) if max_families is None else max_families
        effective_diversity_mode = getattr(self, "diversity_mode", "balanced") if diversity_mode is None else diversity_mode
        effective_family_bias_strength = (
            getattr(self, "family_bias_strength", 0.5) if family_bias_strength is None else family_bias_strength
        )
        try:
            families = self.select_candidate_families(
                component_a,
                constraints,
                context,
                max_families=effective_max_families,
                diversity_mode=effective_diversity_mode,
                family_bias_strength=effective_family_bias_strength,
                prior_productive_families=prior_productive_families,
            )
        except Exception as exc:
            print(f"family selection failed, falling back to single-stage brainstorm: {exc}", file=sys.stderr)
        facts_block = structural_facts(component_a).as_prompt_block()
        try:
            wanted = _complementary_role(hbond_profile(component_a).role)
            partner_menu = known_partner_menu(wanted, limit=30)
        except Exception:
            partner_menu = None
        raw = self._request(
            candidate_brainstorm_prompt(
                component_a,
                constraints,
                context,
                self.max_candidates,
                families,
                diversity_mode=effective_diversity_mode,
                family_bias_strength=effective_family_bias_strength,
                prior_productive_families=prior_productive_families,
                facts_block=facts_block,
                known_partner_menu=partner_menu,
            )
        )
        return parse_candidate_brainstorms(raw)[: self.max_candidates]

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
        raw = self._request(
            family_selection_prompt(
                component_a,
                constraints,
                context,
                max_families=max_families,
                diversity_mode=diversity_mode,
                family_bias_strength=family_bias_strength,
                prior_productive_families=prior_productive_families,
            )
        )
        return parse_candidate_families(raw)

    def generate_explanations(self, results: list[DesResult], context: str) -> list[ExplanationNote]:
        raw = self._request(explanation_prompt(results, context, len(results) or None))
        return parse_explanation_notes(raw)

    def critique_results(self, results: list[DesResult], context: str) -> list[CritiqueNote]:
        raw = self._request(critique_prompt(results, context, len(results) or None))
        return parse_critique_notes(raw)

    def detect_contradictions(self, results: list[DesResult], context: str, facts_block: str = "") -> list[ContradictionNote]:
        raw = self._request(contradiction_prompt(results, context, len(results) or None, facts_block=facts_block))
        return parse_contradiction_notes(raw)

    def assess_candidate_chemistry(
        self,
        candidate_smiles: str,
        context: str,
        memory_notes: list[str] | None = None,
    ) -> list[ChemistryAssessment]:
        facts_block = structural_facts(candidate_smiles).as_prompt_block()
        raw = self._request(chemistry_assessment_prompt(candidate_smiles, context, memory_notes, facts_block=facts_block))
        return parse_chemistry_assessments(raw)

    def suggest_next_steps(
        self,
        context: str,
        memory_notes: list[str] | None = None,
    ) -> list[ChemistryNextStep]:
        raw = self._request(chemistry_next_step_prompt(context, memory_notes))
        return parse_chemistry_next_steps(raw)

    # ------------------------------------------------------------------
    # Metal-binding ligand methods
    # ------------------------------------------------------------------

    def select_ligand_families(
        self, metal_ion: str, constraints: dict | None, context: str
    ) -> list[LigandFamily]:
        raw = self._request(ligand_family_selection_prompt(metal_ion, constraints, context))
        return parse_ligand_families(raw)

    def brainstorm_ligands(
        self, metal_ion: str, constraints: dict | None, context: str
    ) -> list[CandidateBrainstorm]:
        families: list[LigandFamily] = []
        try:
            families = self.select_ligand_families(metal_ion, constraints, context)
        except Exception as exc:
            print(f"ligand family selection failed, falling back to single-stage brainstorm: {exc}", file=sys.stderr)
        ligand_menu = _known_ligand_menu(metal_ion, limit=15)
        raw = self._request(
            ligand_brainstorm_prompt(metal_ion, constraints, context, self.max_candidates, families,
                                     known_ligand_menu=ligand_menu or None)
        )
        return parse_candidate_brainstorms(raw)[: self.max_candidates]

    def review_ligand(self, metal_ion: str, ligand_smiles: str, context: str) -> CandidateReview:
        raw = self._request(ligand_review_prompt(metal_ion, ligand_smiles, context))
        return parse_candidate_review(raw)

    def brainstorm_ligands_selectivity(
        self,
        target_metal: str,
        competitor_metal: str,
        constraints: dict | None,
        context: str,
    ) -> list[CandidateBrainstorm]:
        families: list[LigandFamily] = []
        try:
            families = self.select_ligand_families(target_metal, constraints, context)
        except Exception as exc:
            print(
                f"ligand family selection failed, falling back to single-stage brainstorm: {exc}",
                file=sys.stderr,
            )
        ligand_menu = _known_ligand_menu(target_metal, limit=15)
        raw = self._request(
            ligand_selectivity_brainstorm_prompt(
                target_metal, competitor_metal, constraints, context,
                self.max_candidates, families,
                known_ligand_menu=ligand_menu or None,
            )
        )
        return parse_candidate_brainstorms(raw)[: self.max_candidates]

    def nominate_for_dft(
        self,
        candidates: list,
        target_metal: str,
        competitor_metal: str,
        top_n: int = 3,
    ) -> list[str]:
        """Return SMILES of candidates nominated for DFT validation.

        Parses the LLM's JSON list response; falls back to top-N by composite_score
        on any parse failure.
        """
        import json
        from .prompts import dft_nomination_prompt

        valid_smiles = {r.ligand_smiles for r in candidates}
        raw = self._request(dft_nomination_prompt(candidates, target_metal, competitor_metal, top_n))
        try:
            nominated = json.loads(raw.strip())
            if not isinstance(nominated, list):
                raise ValueError("response is not a JSON list")
            filtered = [s for s in nominated if isinstance(s, str) and s in valid_smiles]
            if filtered:
                return filtered[:top_n]
        except Exception:
            pass
        return nominate_for_dft_fallback(candidates, top_n)

    def request_url(self, api_key: str | None) -> str:
        suffix = self.request_profile.path_template.format(model_name=self.model_name)
        url = f"{self.api_base_url}{suffix}"
        if api_key and self.request_profile.api_key_in_query:
            parts = list(urlsplit(url))
            query = dict(parse_qsl(parts[3]))
            query["key"] = api_key
            parts[3] = urlencode(query)
            return urlunsplit(parts)
        return url

    def build_payload(self, prompt: str) -> dict:
        style = self.request_profile.payload_style
        if style == "ollama":
            return {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "think": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                },
            }
        if style == "openai":
            return {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
        if style == "gemini":
            return {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": self.temperature,
                    "maxOutputTokens": self.max_tokens,
                },
            }
        raise ValueError(f"Unknown request payload style: {style}")

    def extract_text(self, raw: str) -> str:
        raise NotImplementedError
