"""Structural analogue expansion for DES and ligand candidates.

Generates close structural analogues of a seed SMILES using a small set of
RDKit reaction transforms.  Each transform applies one minimal chemical
change (chain extension, chain shortening, or bioisostere swap).  Products
are screened through viability_check so no reactive or over-complex analogues
enter the prediction pipeline.

Designed for both DES partner search (generate_analogues of top-scoring
component B) and metal-binding ligand search (generate_analogues of
top-scoring ligands).  Returns at most max_n unique valid SMILES per call.
"""
from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import AllChem

from .chemistry_filter import viability_check


# ---------------------------------------------------------------------------
# Reaction transforms — precompiled at module load
# ---------------------------------------------------------------------------

_TRANSFORM_SMARTS: list[tuple[str, str]] = [
    # Insert one CH2 into an aliphatic (non-ring) C–C bond (chain homologation)
    ("[CX4H2;!R:1]-[CX4;!R:2]>>[CX4H2;!R:1]-C-[CX4;!R:2]", "chain_extend"),
    # Remove one CH2 from a non-ring aliphatic chain (chain shortening)
    ("[CX4;!R:1]-[CX4H2;!R:2]-[CX4;!R:3]>>[CX4;!R:1]-[CX4;!R:3]", "chain_shorten"),
    # Hydroxyl → primary amine bioisostere (keeps H-bond capacity)
    ("[CX4:1]-[OX2H1:2]>>[CX4:1]-[NH2]", "oh_to_nh2"),
    # Primary amine → hydroxyl bioisostere
    ("[CX4:1]-[NX3H2:2]>>[CX4:1]-[OH]", "nh2_to_oh"),
    # N-methylation of primary amine (modifies basicity)
    ("[CX4:1]-[NX3H2:2]>>[CX4:1]-[NX3H1:2]C", "n_methyl"),
]

_TRANSFORMS: list[tuple[object, str]] = []
for _smarts, _name in _TRANSFORM_SMARTS:
    try:
        _rxn = AllChem.ReactionFromSmarts(_smarts)
        if _rxn is not None:
            _TRANSFORMS.append((_rxn, _name))
    except Exception as _exc:
        import sys as _sys
        print(f"[analogue_expansion] WARNING: transform {_name!r} failed to compile: {_exc}", file=_sys.stderr)

if len(_TRANSFORMS) < len(_TRANSFORM_SMARTS):
    import sys as _sys
    print(
        f"[analogue_expansion] WARNING: only {len(_TRANSFORMS)}/{len(_TRANSFORM_SMARTS)} transforms loaded",
        file=_sys.stderr,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_analogues_tagged(
    smiles: str,
    max_n: int = 5,
    transform_weights: dict[str, float] | None = None,
) -> list[tuple[str, str]]:
    """Return up to *max_n* (analogue_smiles, transform_name) pairs.

    *transform_weights* maps transform name → weight (higher = tried first).
    Transforms not in the dict receive weight 1.0.  This lets callers that
    track per-transform hit rates bias future calls toward productive transforms.

    Each analogue SMILES is distinct from the input and all prior results,
    has ≥3 heavy atoms, and passes viability_check.  Never raises.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    try:
        seed_canonical = Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return []

    # Re-order transforms by descending weight for adaptive selection.
    if transform_weights:
        ordered = sorted(
            _TRANSFORMS,
            key=lambda t: transform_weights.get(t[1], 1.0),
            reverse=True,
        )
    else:
        ordered = list(_TRANSFORMS)

    seen: set[str] = {seed_canonical}
    results: list[tuple[str, str]] = []

    for rxn, name in ordered:
        if len(results) >= max_n:
            break
        try:
            product_sets = rxn.RunReactants((mol,))
        except Exception:
            continue

        batch: list[str] = []
        for product_tuple in product_sets:
            if not product_tuple:
                continue
            prod = product_tuple[0]
            try:
                Chem.SanitizeMol(prod)
                canon = Chem.MolToSmiles(prod, canonical=True)
            except Exception:
                continue
            if not canon or canon in seen:
                continue
            if prod.GetNumAtoms() < 3:
                continue
            ok, _ = viability_check(prod)
            if not ok:
                continue
            seen.add(canon)
            batch.append(canon)

        for canon in sorted(batch):
            if len(results) >= max_n:
                break
            results.append((canon, name))

    return results


def generate_analogues(
    smiles: str,
    max_n: int = 5,
    transform_weights: dict[str, float] | None = None,
) -> list[str]:
    """Return up to *max_n* structural analogues of *smiles*.

    Convenience wrapper around :func:`generate_analogues_tagged` that drops
    the transform attribution.  Pass *transform_weights* to bias toward
    historically productive transforms.
    """
    return [smi for smi, _ in generate_analogues_tagged(smiles, max_n, transform_weights)]
