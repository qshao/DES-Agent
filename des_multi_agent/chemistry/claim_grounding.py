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
from .partner_registry import is_known, structural_sanity
from .protonation import dominant_species
from ..chemistry_filter import viability_check

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

# Each entry: (compiled SMARTS mol, minimum_match_count)
# Pre-compiled at module load to avoid recompilation on every structural_facts/ground_family call.
_FAMILY_SMARTS: dict[str, tuple[str, int]] = {
    "polyol":          ("[OX2H]", 2),      # ≥2 hydroxyl groups
    "amide":           ("C(=O)N", 1),
    "carboxylic acid": ("C(=O)[OH]", 1),
    "amine":           ("[NX3;H1,H2]", 1),
    "phenol":          ("c[OH]", 1),
}
_FAMILY_PATTERNS: dict[str, tuple[object, int]] = {
    tag: (Chem.MolFromSmarts(smarts), min_count)
    for tag, (smarts, min_count) in _FAMILY_SMARTS.items()
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
    net_charge: int = 0                   # net formal charge of the profiled species
    protonation_summary: str = ""         # non-empty only when a pH was applied

    def as_prompt_block(self) -> str:
        """Return a compact single-line fact string suitable for LLM injection."""
        donor_str = ", ".join(
            f"{count} {elem}"
            for elem, count in self.donor_element_counts.items()
        ) or "none"
        feat_str = str(self.family_features)
        block = (
            f"computed facts: HBD={self.n_hbd}, HBA={self.n_hba}, "
            f"role={self.hbond_role}, donor atoms={donor_str}, "
            f"denticity={self.denticity}, features={feat_str}"
        )
        if self.protonation_summary:
            block += f"; {self.protonation_summary}"
        return block


def structural_facts(smiles: str, pH: float | None = None) -> StructuralFacts:
    """Compute a deterministic structural summary for *smiles*.

    When *pH* is None (default) the molecule is profiled as drawn — byte-identical
    to the original behavior. When *pH* is a float, the dominant ionized species at
    that pH is profiled for H-bond and coordination counts (family features are
    always read from the as-drawn form). Never raises.
    """
    try:
        net_charge = 0
        protonation_summary = ""
        if pH is None:
            profiled = smiles
        else:
            res = dominant_species(smiles, pH)
            profiled = res.species_smiles
            net_charge = res.net_charge
            ionized = [g for g in res.groups if g.state != "neutral"]
            if ionized:
                parts = ", ".join(f"{g.group_name} {g.state}" for g in ionized)
                protonation_summary = f"species @ pH{pH:g}: net charge {net_charge:+d} ({parts})"
            else:
                protonation_summary = f"species @ pH{pH:g}: net charge {net_charge:+d} (no ionizable groups)"

        hb = hbond_profile(profiled)
        cp = coordination_profile(profiled)

        # Family features are read from the AS-DRAWN molecule.
        mol = Chem.MolFromSmiles(smiles)
        features: list[str] = []
        if mol is not None:
            for tag, (patt, min_count) in _FAMILY_PATTERNS.items():
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
            net_charge=net_charge,
            protonation_summary=protonation_summary,
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
# PartnerVerdict
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PartnerVerdict:
    """Reality grading of one proposed DES partner.

    status:      "known" | "novel_plausible" | "novel_implausible"
    disposition: "keep" | "demote" | "drop"
    Invariants:
      keep   ⟺ penalty == 0.0 and status in {known, novel_plausible}
      demote ⟹ penalty > 0.0 and status == novel_implausible
      drop   ⟹ penalty == 0.0 and status == novel_implausible
    """

    claim: str
    status: str
    detail: str
    penalty: float
    disposition: str
    candidate_smiles: str = ""   # the SMILES this verdict is about

    def __post_init__(self) -> None:
        if self.disposition == "keep":
            if self.penalty != 0.0 or self.status not in ("known", "novel_plausible"):
                raise ValueError("keep requires penalty=0.0 and a non-implausible status")
        elif self.disposition == "demote":
            if self.penalty <= 0.0 or self.status != "novel_implausible":
                raise ValueError("demote requires penalty>0.0 and status=novel_implausible")
        elif self.disposition == "drop":
            if self.penalty != 0.0 or self.status != "novel_implausible":
                raise ValueError("drop requires penalty=0.0 and status=novel_implausible")
        else:
            raise ValueError(f"unknown disposition: {self.disposition!r}")


def ground_partner_reality(component_a: str, candidate_smiles: str) -> PartnerVerdict:
    """Deterministically grade a proposed DES partner against reality.

    Order: invalid → drop; known → keep; bad structure → drop; reactive/toxic/
    too complex → drop; no H-bond complementarity → demote; otherwise keep.
    Never raises; on internal error returns a neutral keep.
    """
    claim = f"partner reality: {candidate_smiles}"
    try:
        mol = Chem.MolFromSmiles(candidate_smiles)
        if mol is None:
            return PartnerVerdict(claim, "novel_implausible", "invalid SMILES", 0.0, "drop", candidate_smiles)
        if is_known(candidate_smiles):
            return PartnerVerdict(claim, "known", "known/attested compound", 0.0, "keep", candidate_smiles)
        ok, reason = structural_sanity(candidate_smiles)
        if not ok:
            return PartnerVerdict(claim, "novel_implausible", reason, 0.0, "drop", candidate_smiles)
        ok_v, reason_v = viability_check(mol)
        if not ok_v:
            return PartnerVerdict(claim, "novel_implausible", reason_v, 0.0, "drop", candidate_smiles)
        label = des_hbond_complementarity(component_a, candidate_smiles).label
        if label == "none":
            return PartnerVerdict(
                claim, "novel_implausible",
                "no H-bond complementarity with component A", 0.25, "demote", candidate_smiles,
            )
        return PartnerVerdict(claim, "novel_plausible", f"novel; complementarity={label}", 0.0, "keep", candidate_smiles)
    except Exception:
        return PartnerVerdict(claim, "novel_plausible", "reality check skipped (internal error)", 0.0, "keep", candidate_smiles)


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


def ground_coordination(
    smiles: str, claim_text: str, pH: float | None = None
) -> GroundingVerdict:
    """Verify a natural-language coordination claim against structural evidence.

    Uses :func:`verify_coordination_claim` from *claim_verification.py* and maps
    the result to a :class:`GroundingVerdict`. When *pH* is provided, the claim
    is verified against the dominant ionized species at that pH.
    """
    try:
        target_smiles = smiles
        if pH is not None:
            target_smiles = dominant_species(smiles, pH).species_smiles
        cv = verify_coordination_claim(target_smiles, claim_text)
        status = _COORD_VERDICT_MAP.get(cv.verdict, "unverifiable")
        penalty = 0.25 if status == "contradicted" else 0.0
        detail = "; ".join(cv.notes) if cv.notes else cv.verdict
        return GroundingVerdict(
            claim=claim_text,
            status=status,
            detail=detail,
            penalty=penalty,
        )
    except Exception as exc:
        return GroundingVerdict(
            claim=claim_text,
            status="unverifiable",
            detail=f"coordination grounding failed: {exc}",
            penalty=0.0,
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

    if key not in _FAMILY_PATTERNS:
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

    patt, min_count = _FAMILY_PATTERNS[key]
    smarts = _FAMILY_SMARTS[key][0]
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
