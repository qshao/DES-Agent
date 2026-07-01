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
    # Insert one CH2 into an aliphatic C–C bond (chain homologation)
    ("[CX4H2:1]-[CX4:2]>>[CX4H2:1]-C-[CX4:2]", "chain_extend"),
    # Remove one CH2 from an aliphatic chain (chain shortening)
    ("[CX4:1]-[CX4H2:2]-[CX4:3]>>[CX4:1]-[CX4:3]", "chain_shorten"),
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
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_analogues(smiles: str, max_n: int = 5) -> list[str]:
    """Return up to *max_n* structural analogues of *smiles*.

    Each returned SMILES is:
    - distinct from the input (different canonical SMILES)
    - distinct from all other returned SMILES
    - at least 3 heavy atoms
    - passing viability_check (no reactive groups, reasonable complexity)

    Never raises; returns an empty list on invalid input or if no valid
    analogues can be generated.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    try:
        seed_canonical = Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return []

    seen: set[str] = {seed_canonical}
    results: list[str] = []

    for rxn, _ in _TRANSFORMS:
        if len(results) >= max_n:
            break
        try:
            product_sets = rxn.RunReactants((mol,))
        except Exception:
            continue

        # Collect unique valid products across all match positions
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
            if canon not in seen:
                seen.add(canon)
                batch.append(canon)

        # Sort for determinism, then append up to remaining budget
        for canon in sorted(batch):
            if len(results) >= max_n:
                break
            results.append(canon)

    return results
