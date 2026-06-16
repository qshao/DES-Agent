"""Deterministic chemistry grounding layer.

Provides:
  structural_facts(smiles)     → StructuralFacts
  ground_coordination(...)     → GroundingVerdict
  ground_selectivity(...)      → GroundingVerdict
  ground_family(...)           → GroundingVerdict
  ground_des_plausibility(..)  → GroundingVerdict

Zero LLM dependency.  Identical verdicts regardless of LLM backend.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from rdkit import Chem

from .claim_verification import verify_coordination_claim, verify_selectivity_claim
from .coordination import coordination_profile
from .hbond import des_hbond_complementarity, hbond_profile

# ---------------------------------------------------------------------------
# Metal normalisation helpers
# ---------------------------------------------------------------------------

# Common charge suffixes to try when bare symbol is given (e.g. "Cu" → "Cu2+")
_CHARGE_SUFFIXES = ("2+", "3+", "+", "2-", "-")


def _normalise_metal(symbol: str) -> str | None:
    """Return the canonical metal-ion key (e.g. "Cu2+") or None if unknown.

    Tries the symbol as-is first, then appends common charge suffixes.
    Imports _METAL_IDENTITY lazily to avoid circular imports.
    """
    from des_multi_agent.predictors.stability_constants import _METAL_IDENTITY

    key = symbol.strip()
    if key in _METAL_IDENTITY:
        return key
    for suffix in _CHARGE_SUFFIXES:
        candidate = key + suffix
        if candidate in _METAL_IDENTITY:
            return candidate
    return None  # unknown metal


# ---------------------------------------------------------------------------
# SMARTS family table
# ---------------------------------------------------------------------------

# Each entry: smarts_pattern, minimum_match_count
_FAMILY_SMARTS: dict[str, tuple[str, int]] = {
    "polyol":          ("[OX2H]", 2),      # ≥2 hydroxyl groups
    "amide":           ("C(=O)N", 1),
    "carboxylic acid": ("C(=O)[OH]", 1),
    "amine":           ("[NX3;H1,H2]", 1),
    "phenol":          ("c[OH]", 1),
}

# ---------------------------------------------------------------------------
# StructuralFacts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StructuralFacts:
    """Deterministic structural summary for a single SMILES string."""

    smiles: str                           # canonical SMILES or "unparseable"
    n_hbd: int
    n_hba: int
    hbond_role: str
    n_donor_atoms: int
    denticity: int
    donor_element_counts: dict[str, int]
    mean_donor_softness: float
    family_features: list[str]            # detected family tags, e.g. ["polyol"]

    def as_prompt_block(self) -> str:
        """Return a compact single-line fact string suitable for LLM injection."""
        donor_str = ", ".join(
            f"{count} {elem}"
            for elem, count in self.donor_element_counts.items()
        ) or "none"
        feat_str = str(self.family_features)
        return (
            f"computed facts: HBD={self.n_hbd}, HBA={self.n_hba}, "
            f"role={self.hbond_role}, donor atoms={donor_str}, "
            f"denticity={self.denticity}, features={feat_str}"
        )


def structural_facts(smiles: str) -> StructuralFacts:
    """Compute a deterministic structural summary for *smiles*.

    Never raises — returns a safe sentinel object on any error.
    """
    try:
        hb = hbond_profile(smiles)
        cp = coordination_profile(smiles)

        # Detect family features
        mol = Chem.MolFromSmiles(smiles)
        features: list[str] = []
        if mol is not None:
            for tag, (smarts, min_count) in _FAMILY_SMARTS.items():
                patt = Chem.MolFromSmarts(smarts)
                if patt is not None:
                    matches = mol.GetSubstructMatches(patt)
                    if len(matches) >= min_count:
                        features.append(tag)

        return StructuralFacts(
            smiles=hb.smiles,
            n_hbd=hb.n_hbd,
            n_hba=hb.n_hba,
            hbond_role=hb.role,
            n_donor_atoms=cp.n_donor_atoms,
            denticity=cp.denticity,
            donor_element_counts=dict(cp.donor_element_counts),
            mean_donor_softness=cp.mean_donor_softness,
            family_features=features,
        )
    except Exception:
        return StructuralFacts(
            smiles="unparseable",
            n_hbd=0,
            n_hba=0,
            hbond_role="none",
            n_donor_atoms=0,
            denticity=0,
            donor_element_counts={},
            mean_donor_softness=0.0,
            family_features=[],
        )


# ---------------------------------------------------------------------------
# GroundingVerdict
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GroundingVerdict:
    """Outcome of a single deterministic chemistry grounding check.

    Invariant: penalty == 0.0 when status is "verified" or "unverifiable";
    penalty > 0.0 when status is "contradicted".
    """

    claim: str
    status: str    # "verified" | "contradicted" | "unverifiable"
    detail: str
    penalty: float  # 0.0 for verified/unverifiable, 0.25 for contradicted

    def __post_init__(self) -> None:
        if self.status == "contradicted" and self.penalty == 0.0:
            raise ValueError("contradicted verdict must carry a non-zero penalty")
        if self.status in ("verified", "unverifiable") and self.penalty != 0.0:
            raise ValueError(f"status={self.status!r} must have penalty=0.0")


# ---------------------------------------------------------------------------
# ground_coordination
# ---------------------------------------------------------------------------

# Map claim_verification verdicts → grounding statuses
_COORD_VERDICT_MAP: dict[str, str] = {
    "ok":                "verified",
    "denticity_mismatch": "contradicted",
    "donor_mismatch":    "contradicted",
    "unparseable":       "unverifiable",
    "not_a_ligand":      "unverifiable",
}


def ground_coordination(smiles: str, claim_text: str) -> GroundingVerdict:
    """Verify a natural-language coordination claim against structural evidence.

    Uses :func:`verify_coordination_claim` from *claim_verification.py* and maps
    the result to a :class:`GroundingVerdict`.
    """
    cv = verify_coordination_claim(smiles, claim_text)
    status = _COORD_VERDICT_MAP.get(cv.verdict, "unverifiable")
    penalty = 0.25 if status == "contradicted" else 0.0
    detail = "; ".join(cv.notes) if cv.notes else cv.verdict
    return GroundingVerdict(
        claim=claim_text,
        status=status,
        detail=detail,
        penalty=penalty,
    )


# ---------------------------------------------------------------------------
# ground_selectivity
# ---------------------------------------------------------------------------


def ground_selectivity(
    target: str,
    competitor: str,
    smiles: str,
    claim_sign: str,
) -> GroundingVerdict:
    """Verify a metal-selectivity claim against rule-based ΔlogK evidence.

    *target* and *competitor* may be bare element symbols (e.g. ``"Cu"``) or
    full ion keys (e.g. ``"Cu2+"``).  Both are normalised against
    :data:`_METAL_IDENTITY` before the lookup; unknown metals return
    ``"unverifiable"``.

    *claim_sign* is one of:
      ``"target_selective"`` | ``"competitor_selective"`` | ``"neutral"``

    When the rule-based verdict is ``"neutral"`` but *claim_sign* is directional
    (or vice-versa), the result is ``"unverifiable"`` rather than ``"contradicted"``
    — the evidence is too weak to call a contradiction in either direction.
    """
    norm_target = _normalise_metal(target)
    norm_competitor = _normalise_metal(competitor)

    if norm_target is None or norm_competitor is None:
        unknown = target if norm_target is None else competitor
        return GroundingVerdict(
            claim=f"{target} > {competitor} for {smiles}",
            status="unverifiable",
            detail=f"Metal '{unknown}' not in deterministic stability table",
            penalty=0.0,
        )

    try:
        sv = verify_selectivity_claim(norm_target, norm_competitor, smiles)
    except Exception as exc:
        return GroundingVerdict(
            claim=f"{target} > {competitor} for {smiles}",
            status="unverifiable",
            detail=f"Selectivity computation failed: {exc}",
            penalty=0.0,
        )

    actual = sv.verdict   # "target_selective" | "competitor_selective" | "neutral"
    claim = f"{target} > {competitor} for {smiles}"

    if actual == claim_sign:
        return GroundingVerdict(
            claim=claim,
            status="verified",
            detail=f"ΔlogK={sv.delta_log_k:.3f}; actual={actual}",
            penalty=0.0,
        )

    # Contradicted only when both sides are non-neutral and they disagree
    if actual != "neutral" and claim_sign != "neutral":
        return GroundingVerdict(
            claim=claim,
            status="contradicted",
            detail=f"ΔlogK={sv.delta_log_k:.3f}; actual={actual} contradicts claimed={claim_sign}",
            penalty=0.25,
        )

    return GroundingVerdict(
        claim=claim,
        status="unverifiable",
        detail=f"ΔlogK={sv.delta_log_k:.3f}; actual={actual}; claim={claim_sign}",
        penalty=0.0,
    )


# ---------------------------------------------------------------------------
# ground_family
# ---------------------------------------------------------------------------


def ground_family(smiles: str, family_label: str) -> GroundingVerdict:
    """Check whether *smiles* belongs to the named chemical family.

    Uses a deterministic SMARTS table.  Unknown *family_label* values return
    ``"unverifiable"`` (no penalty).
    """
    key = family_label.lower().strip()
    claim = f"{smiles} is a {family_label}"

    if key not in _FAMILY_SMARTS:
        return GroundingVerdict(
            claim=claim,
            status="unverifiable",
            detail="family label not in deterministic table",
            penalty=0.0,
        )

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return GroundingVerdict(
            claim=claim,
            status="unverifiable",
            detail=f"Could not parse SMILES: {smiles!r}",
            penalty=0.0,
        )

    smarts, min_count = _FAMILY_SMARTS[key]
    patt = Chem.MolFromSmarts(smarts)
    actual_count = len(mol.GetSubstructMatches(patt)) if patt is not None else 0
    detail = f"found {actual_count} match(es) of '{smarts}'; need ≥{min_count}"

    if actual_count >= min_count:
        return GroundingVerdict(claim=claim, status="verified", detail=detail, penalty=0.0)
    return GroundingVerdict(claim=claim, status="contradicted", detail=detail, penalty=0.25)


# ---------------------------------------------------------------------------
# ground_des_plausibility
# ---------------------------------------------------------------------------


def ground_des_plausibility(component_a: str, candidate: str) -> GroundingVerdict:
    """Check whether a component pair is plausible as a DES via H-bond complementarity.

    Uses :func:`des_hbond_complementarity` and interprets the composite label:
    - ``"strong"`` or ``"moderate"`` → verified
    - ``"weak"``                      → unverifiable (insufficient evidence)
    - ``"none"``                      → contradicted (no DES formation expected)
    """
    claim = f"DES plausibility of {component_a!r} + {candidate!r}"

    try:
        hbc = des_hbond_complementarity(component_a, candidate)
    except Exception as exc:
        return GroundingVerdict(
            claim=claim,
            status="unverifiable",
            detail=f"H-bond computation failed: {exc}",
            penalty=0.0,
        )

    label = hbc.label
    detail = (
        f"H-bond complementarity: label={label}, "
        f"complementarity_score={hbc.complementarity_score:.3f}, "
        f"composite_score={hbc.composite_score:.3f}"
    )

    if label in ("strong", "moderate"):
        return GroundingVerdict(claim=claim, status="verified", detail=detail, penalty=0.0)
    if label == "none":
        return GroundingVerdict(
            claim=claim,
            status="contradicted",
            detail=f"H-bond complementarity label=none; no DES formation expected. {detail}",
            penalty=0.25,
        )
    # "weak" — some evidence but not enough to verify or contradict
    return GroundingVerdict(claim=claim, status="unverifiable", detail=detail, penalty=0.0)
