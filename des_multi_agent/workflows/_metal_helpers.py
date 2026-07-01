"""Shared helpers for metal-binding workflow files."""
from __future__ import annotations


def _apply_ligand_reality_gate(
    metal_ion: str,
    brainstorms: list,
    proposals: list,
    all_warnings: list,
) -> list:
    """Drop proposals that fail ground_ligand_reality for *metal_ion*.

    Builds a canonical-SMILES drop set so the filter matches the canonical
    SMILES stored in *proposals* by _deduplicate_proposals, regardless of
    whether the LLM returned a non-canonical form.
    """
    from ..chemistry.claim_grounding import ground_ligand_reality as _ground_lig
    from ..chemistry_filter import canonicalize_smiles as _canon

    drop_smiles: set[str] = set()
    for b in brainstorms:
        try:
            rv = _ground_lig(metal_ion, b.smiles)
            if rv.disposition == "drop":
                drop_smiles.add(_canon(b.smiles) or b.smiles)
                all_warnings.append(
                    f"[GROUNDING] Ligand dropped (reality): {b.smiles}: {rv.detail}"
                )
        except Exception:
            pass
    if drop_smiles:
        return [p for p in proposals if p.smiles not in drop_smiles]
    return proposals
