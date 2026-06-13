from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from .chemistry_filter import canonicalize_smiles
from .schemas import CandidateProposal


_DEFAULT_MAX_SIMILARITY = 0.85
_DEFAULT_MAX_FAMILIES = 6
_DEFAULT_FAMILY_BIAS_STRENGTH = 0.5

_FAMILY_NEIGHBORS: dict[str, tuple[str, ...]] = {
    "polyol": ("amide", "urea", "carboxylic acid", "alcohol", "diol"),
    "alcohol": ("polyol", "diol", "amide", "urea"),
    "diol": ("polyol", "alcohol", "amide", "urea"),
    "short diol": ("polyol", "alcohol", "amide", "urea"),
    "amide": ("urea", "polyol", "carboxylic acid", "lactam"),
    "urea": ("amide", "polyol", "carboxylic acid", "cyclic urea"),
    "carboxylic acid": ("amide", "urea", "polyol", "alcohol"),
    "lactam": ("amide", "urea", "cyclic urea"),
    "cyclic urea": ("urea", "amide", "lactam"),
    "amine": ("amide", "urea", "alcohol"),
    "quaternary ammonium salt": ("choline-like", "ionic partner", "alcohol"),
    "choline-like": ("quaternary ammonium salt", "alcohol", "polyol"),
    "hydroxypyridine": ("polyol", "amide", "lactam"),
    "hydroxyamide": ("amide", "polyol", "urea"),
    "sulfoxide": ("sulfone", "amide", "urea"),
    "sulfone": ("sulfoxide", "amide", "urea"),
}




def _config_value(mapping: Mapping[str, object] | object | None, key: str, default: object) -> object:
    if mapping is None:
        return default
    if isinstance(mapping, Mapping):
        return mapping.get(key, default)
    return getattr(mapping, key, default)

@dataclass(frozen=True)
class ProposalDiversityConfig:
    max_similarity: float = _DEFAULT_MAX_SIMILARITY
    deduplicate_exact: bool = True
    deduplicate_near: bool = True
    family_fallback: bool = True
    per_family_budget: int | None = None
    diversity_mode: str = "balanced"
    max_families: int = _DEFAULT_MAX_FAMILIES
    family_bias_strength: float = _DEFAULT_FAMILY_BIAS_STRENGTH

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object] | object | None) -> "ProposalDiversityConfig":
        if mapping is None:
            return cls()
        per_family_budget = _config_value(mapping, "per_family_budget", None)
        return cls(
            max_similarity=float(_config_value(mapping, "max_similarity", _DEFAULT_MAX_SIMILARITY)),
            deduplicate_exact=bool(_config_value(mapping, "deduplicate_exact", True)),
            deduplicate_near=bool(_config_value(mapping, "deduplicate_near", True)),
            family_fallback=bool(_config_value(mapping, "family_fallback", True)),
            per_family_budget=None if per_family_budget is None else int(per_family_budget),
            diversity_mode=str(_config_value(mapping, "diversity_mode", "balanced")),
            max_families=int(_config_value(mapping, "max_families", _DEFAULT_MAX_FAMILIES)),
            family_bias_strength=float(_config_value(mapping, "family_bias_strength", _DEFAULT_FAMILY_BIAS_STRENGTH)),
        )


@dataclass(frozen=True)
class ProposalDiversityResult:
    accepted: list[CandidateProposal] = field(default_factory=list)
    suppressed: list[CandidateProposal] = field(default_factory=list)
    suggested_families: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

def _fingerprint_from_smiles(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=1, nBits=2048)


def _similarity(fp_a, fp_b) -> float:
    return float(DataStructs.TanimotoSimilarity(fp_a, fp_b))


def _unique_extend(target: list[str], values: Sequence[str]) -> None:
    seen = set(target)
    for value in values:
        cleaned = value.strip()
        if not cleaned or cleaned in seen:
            continue
        target.append(cleaned)
        seen.add(cleaned)


def _adjacent_families(family: str) -> list[str]:
    return list(_FAMILY_NEIGHBORS.get(family, ()))


def _suggest_families_from_collapse(
    family_counts: Counter[str],
    accepted_families: set[str],
    suppressed_families: set[str],
    diversity_mode: str,
) -> list[str]:
    if not family_counts:
        return []

    dominant_family, dominant_count = family_counts.most_common(1)[0]
    if dominant_count < 2 and not suppressed_families:
        return []
    if diversity_mode.strip().lower() == "exploit":
        return []

    suggestions: list[str] = []
    _unique_extend(suggestions, _adjacent_families(dominant_family))
    for family in sorted(suppressed_families):
        if family == dominant_family:
            continue
        _unique_extend(suggestions, _adjacent_families(family))
    if not suggestions:
        _unique_extend(suggestions, sorted(suppressed_families - accepted_families))
    return [family for family in suggestions if family not in accepted_families][:6]


def apply_proposal_diversity(
    component_a: str,
    proposals: Sequence[CandidateProposal],
    config: ProposalDiversityConfig | None = None,
) -> ProposalDiversityResult:
    effective_config = config or ProposalDiversityConfig()
    canonical_component_a = canonicalize_smiles(component_a)
    accepted: list[CandidateProposal] = []
    suppressed: list[CandidateProposal] = []
    notes: list[str] = []
    accepted_canonical: set[str] = set()
    accepted_fingerprints: list[object] = []
    accepted_family_counts: Counter[str] = Counter()
    proposal_family_counts: Counter[str] = Counter()
    suppressed_families: set[str] = set()

    for proposal in proposals:
        family = proposal.family.strip() if proposal.family else ""
        if family:
            proposal_family_counts[family] += 1

        smiles = proposal.smiles.strip()
        if not smiles:
            suppressed.append(proposal)
            notes.append("Suppressed empty proposal SMILES")
            continue

        try:
            canonical = canonicalize_smiles(smiles)
        except ValueError:
            suppressed.append(proposal)
            notes.append(f"Suppressed invalid proposal SMILES: {smiles}")
            continue

        if canonical == canonical_component_a:
            suppressed.append(proposal)
            suppressed_families.add(family)
            notes.append(f"Suppressed component-matching proposal: {smiles}")
            continue

        if effective_config.deduplicate_exact and canonical in accepted_canonical:
            suppressed.append(proposal)
            suppressed_families.add(family)
            notes.append(f"Suppressed exact duplicate proposal: {smiles}")
            continue

        if effective_config.deduplicate_near and accepted_fingerprints:
            candidate_fp = _fingerprint_from_smiles(canonical)
            if any(_similarity(candidate_fp, existing_fp) >= effective_config.max_similarity for existing_fp in accepted_fingerprints):
                suppressed.append(proposal)
                suppressed_families.add(family)
                notes.append(f"Suppressed near-duplicate proposal: {smiles}")
                continue
        else:
            candidate_fp = None

        if effective_config.per_family_budget is not None and family:
            if accepted_family_counts[family] >= effective_config.per_family_budget:
                suppressed.append(proposal)
                suppressed_families.add(family)
                notes.append(f"Suppressed family-budget overflow for {family}: {smiles}")
                continue

        accepted.append(proposal)
        accepted_canonical.add(canonical)
        accepted_family_counts[family] += 1
        if effective_config.deduplicate_near:
            if candidate_fp is None:
                candidate_fp = _fingerprint_from_smiles(canonical)
            accepted_fingerprints.append(candidate_fp)

    suggested_families = []
    if effective_config.family_fallback:
        suggested_families = _suggest_families_from_collapse(
            proposal_family_counts,
            {family for family in accepted_family_counts if family},
            suppressed_families,
            effective_config.diversity_mode,
        )

    return ProposalDiversityResult(
        accepted=accepted,
        suppressed=suppressed,
        suggested_families=suggested_families[: max(0, effective_config.max_families)],
        notes=notes,
    )
