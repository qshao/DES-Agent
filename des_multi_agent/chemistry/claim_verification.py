"""Cross-check LLM coordination and selectivity claims against structure-based evidence.

Two verifiers:
1. ``verify_coordination_claim`` — parses a coordination-mode string (e.g.
   "bidentate N,O-chelator") and compares the claimed denticity / donor atoms
   against ``coordination_profile()``.
2. ``verify_selectivity_claim`` — computes the rule-based ΔlogK
   (target − competitor) from ``selectivity_delta_log_k()`` and classifies
   whether the structural evidence supports or contradicts an LLM sign claim.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .coordination import coordination_profile
from .stability_rules import selectivity_delta_log_k

# ---------------------------------------------------------------------------
# Coordination-mode parsing
# ---------------------------------------------------------------------------

_DENTAL_MAP: dict[str, int] = {
    "mono": 1, "bi": 2, "tri": 3, "tetra": 4, "penta": 5, "hexa": 6,
}
_DENTAL_RE = re.compile(r"\b(mono|bi|tri|tetra|penta|hexa)dentate\b", re.I)
# Comma-separated donor run: "N,O" or "N,N,N" (requires at least one comma)
_COMMA_DONOR_RE = re.compile(r"[NOSP](?:,\s*[NOSP])+", re.I)

_DONOR_TONE_MAP: dict[str, int] = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
}


def _parse_dental(text: str) -> int | None:
    m = _DENTAL_RE.search(text)
    if m:
        return _DENTAL_MAP[m.group(1).lower()]
    for word, n in _DONOR_TONE_MAP.items():
        if word in text.lower():
            return n
    return None


def _parse_donors(text: str) -> list[str]:
    """Return a list of donor element symbols extracted from *text*.

    Donor letters (N/O/S/P) are only extracted from structured contexts to
    avoid matching them inside common words like "monodentate" or "solvent".
    """
    text_upper = text.upper()

    # Primary: extract from the suffix immediately after the dental keyword.
    # e.g. "bidentate N,O-chelator" → suffix="N,O-CHELATOR" → elem_part="N,O" → ["N","O"]
    dental_m = _DENTAL_RE.search(text_upper)
    if dental_m:
        suffix = text_upper[dental_m.end():].lstrip()
        # Grab up to the first non-element-list character (hyphen, space, paren)
        elem_part = re.split(r"[-(\s]", suffix)[0] if suffix else ""
        elements = [c for c in elem_part if c in "NOSP"]
        if elements:
            return elements

    # Fallback: comma-separated element list anywhere (requires comma as separator
    # so random letters inside words are not mistakenly captured).
    comma_m = _COMMA_DONOR_RE.search(text_upper)
    if comma_m:
        return [c for c in comma_m.group(0) if c in "NOSP"]

    return []


# ---------------------------------------------------------------------------
# Claim dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ClaimVerification:
    smiles: str
    claim_text: str
    claimed_denticity: int | None
    claimed_donors: list[str]
    actual_denticity: int
    actual_donor_counts: dict[str, int]
    verdict: str  # "ok" | "denticity_mismatch" | "donor_mismatch" | "not_a_ligand" | "unparseable"
    notes: list[str] = field(default_factory=list)

    @property
    def is_ok(self) -> bool:
        return self.verdict == "ok"


@dataclass(frozen=True)
class SelectivityVerification:
    target_metal: str
    competitor_metal: str
    ligand_smiles: str
    delta_log_k: float       # rule-based log K(target) - log K(competitor)
    verdict: str             # "target_selective" | "competitor_selective" | "neutral"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def verify_coordination_claim(smiles: str, claim_text: str) -> ClaimVerification:
    """Parse *claim_text* and check it against the actual coordination profile.

    Returns a :class:`ClaimVerification` with verdicts:
    - ``"ok"`` — denticity and donors match.
    - ``"denticity_mismatch"`` — donor atoms are present but denticity disagrees.
    - ``"donor_mismatch"`` — claimed donor element absent from the structure.
    - ``"not_a_ligand"`` — structure has no donor atoms at all.
    - ``"unparseable"`` — claim text contained no recognisable coordination info.
    """
    claimed_denticity = _parse_dental(claim_text)
    claimed_donors = _parse_donors(claim_text)

    if claimed_denticity is None and not claimed_donors:
        return ClaimVerification(
            smiles=smiles,
            claim_text=claim_text,
            claimed_denticity=None,
            claimed_donors=[],
            actual_denticity=0,
            actual_donor_counts={},
            verdict="unparseable",
            notes=["No denticity or donor-element information found in claim text"],
        )

    try:
        prof = coordination_profile(smiles)
    except Exception as exc:
        return ClaimVerification(
            smiles=smiles,
            claim_text=claim_text,
            claimed_denticity=claimed_denticity,
            claimed_donors=claimed_donors,
            actual_denticity=0,
            actual_donor_counts={},
            verdict="unparseable",
            notes=[f"Could not parse SMILES: {exc}"],
        )

    if prof.n_donor_atoms == 0:
        return ClaimVerification(
            smiles=smiles,
            claim_text=claim_text,
            claimed_denticity=claimed_denticity,
            claimed_donors=claimed_donors,
            actual_denticity=0,
            actual_donor_counts={},
            verdict="not_a_ligand",
            notes=["Structure has no donor atoms (N/O/S/P)"],
        )

    notes: list[str] = []
    verdict = "ok"

    # Check donor elements
    for elem in set(claimed_donors):
        if prof.donor_element_counts.get(elem, 0) == 0:
            verdict = "donor_mismatch"
            notes.append(f"Claimed donor '{elem}' not present in structure")

    # Check denticity (allow ±1 tolerance for ring-size ambiguity)
    if verdict == "ok" and claimed_denticity is not None:
        if abs(prof.denticity - claimed_denticity) > 1:
            verdict = "denticity_mismatch"
            notes.append(
                f"Claimed denticity {claimed_denticity} but structure has {prof.denticity} coordinating site(s)"
            )

    if verdict == "ok" and not notes:
        notes.append(
            f"Actual: denticity={prof.denticity}, donors={dict(prof.donor_element_counts)}"
        )

    return ClaimVerification(
        smiles=smiles,
        claim_text=claim_text,
        claimed_denticity=claimed_denticity,
        claimed_donors=claimed_donors,
        actual_denticity=prof.denticity,
        actual_donor_counts=dict(prof.donor_element_counts),
        verdict=verdict,
        notes=notes,
    )


def verify_selectivity_claim(
    target_metal: str,
    competitor_metal: str,
    ligand_smiles: str,
    *,
    neutral_threshold: float = 0.1,
) -> SelectivityVerification:
    """Compute rule-based ΔlogK and classify selectivity direction.

    ``neutral_threshold`` (in log-K units) is the minimum |ΔlogK| needed to
    call a result selective; smaller values are labelled ``"neutral"``.
    """
    delta = selectivity_delta_log_k(target_metal, competitor_metal, ligand_smiles)
    if abs(delta) < neutral_threshold:
        verdict = "neutral"
    elif delta > 0:
        verdict = "target_selective"
    else:
        verdict = "competitor_selective"
    return SelectivityVerification(
        target_metal=target_metal,
        competitor_metal=competitor_metal,
        ligand_smiles=ligand_smiles,
        delta_log_k=delta,
        verdict=verdict,
    )


def batch_verify_coordination(
    smiles_claim_pairs: list[tuple[str, str]],
) -> list[ClaimVerification]:
    """Verify a batch of (smiles, claim_text) pairs; silently skip on error."""
    results: list[ClaimVerification] = []
    for smiles, claim in smiles_claim_pairs:
        try:
            results.append(verify_coordination_claim(smiles, claim))
        except Exception:
            pass
    return results
