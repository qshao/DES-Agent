"""Resolve SMILES strings to human-readable compound names.

The local dictionary covers ~60 common DES-relevant compounds (HBAs, HBDs,
and neutral co-solvents).  When a SMILES is not found locally and
``pubchem=True`` is passed, a single synchronous PubChem REST call is made;
the result is cached in-process for the session.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Raw local dictionary  (RDKit will canonicalise these keys at load time)
# ---------------------------------------------------------------------------

_RAW_NAMES: dict[str, str] = {
    # --- water & simple alcohols ---
    "O": "water",
    "CO": "methanol",
    "CCO": "ethanol",
    "CCCO": "1-propanol",
    "CC(C)O": "isopropanol",
    "CCCCO": "1-butanol",
    # --- diols / polyols ---
    "OCCO": "ethylene glycol",
    "CC(O)CO": "1,2-propanediol",
    "OCCCO": "1,3-propanediol",
    "OCC(O)CO": "glycerol",
    "OCC(O)C(O)CO": "erythritol",
    "OCC(O)C(O)C(O)CO": "xylitol",
    "OCC(O)C(O)C(O)C(O)CO": "sorbitol",
    "OCC(O)C(O)C(O)C(O)C(O)CO": "mannitol",
    # --- carboxylic acids ---
    "OC=O": "formic acid",
    "CC(=O)O": "acetic acid",
    "CCC(=O)O": "propionic acid",
    "CCCC(=O)O": "butyric acid",
    "OC(=O)C(=O)O": "oxalic acid",
    "OC(=O)CC(=O)O": "malonic acid",
    "OC(=O)CCC(=O)O": "succinic acid",
    "OC(=O)CCCC(=O)O": "glutaric acid",
    "OC(=O)CCCCC(=O)O": "adipic acid",
    "OC(=O)c1ccccc1": "benzoic acid",
    "CC(O)C(=O)O": "lactic acid",
    "OC(=O)C(O)CC(=O)O": "malic acid",
    "OC(C(=O)O)C(O)C(=O)O": "tartaric acid",
    "OC(=O)CC(O)(CC(=O)O)C(=O)O": "citric acid",
    "CC(=O)CCC(=O)O": "levulinic acid",
    # --- amino acids & amides ---
    "NCC(=O)O": "glycine",
    "NC(CO)C(=O)O": "serine",
    "CC(O)C(N)C(=O)O": "threonine",
    "NC(=O)CC(N)C(=O)O": "asparagine",
    "NC(=O)CCC(N)C(=O)O": "glutamine",
    "OC(=O)C1CCCN1": "proline",
    "CC(N)=O": "acetamide",
    "CNC(C)=O": "N-methylacetamide",
    "NC(N)=O": "urea",
    # --- phenolic & aromatic ---
    "Oc1ccccc1": "phenol",
    "Oc1cccc(O)c1": "resorcinol",
    "Oc1ccc(O)cc1": "hydroquinone",
    "Oc1ccccc1O": "catechol",
    "OC(=O)c1ccc(O)cc1": "p-hydroxybenzoic acid",
    # --- nitrogen-containing HBAs ---
    "OCC[N+](C)(C)C.[Cl-]": "choline chloride",
    "OCC[N+](C)(C)C": "choline",
    "C[N+](C)(C)CC(=O)[O-]": "betaine",
    "OC(CC([O-])=O)C[N+](C)(C)C": "carnitine",
    "O=C1CCCCCN1": "caprolactam",
    "CCN(CC)CC(=O)Nc1ccccc1C": "lidocaine",
    # --- sulfur / other ---
    "CS(C)=O": "dimethyl sulfoxide",
    # --- aldehydes ---
    "O=Cc1ccco1": "furfural",
    # --- saccharides (simplified canonical forms) ---
    "OCC1OC(O)C(O)C(O)C1O": "glucose",
    "OCC1OC(O)(CO)C(O)C1O": "fructose",
    "OCC1OC(O)C(O)C1O": "ribose",
}

# ---------------------------------------------------------------------------
# Build canonical → name lookup at import time (requires RDKit)
# ---------------------------------------------------------------------------

def _build_canonical_dict() -> dict[str, str]:
    try:
        from rdkit import Chem
    except ImportError:
        return {}
    result: dict[str, str] = {}
    for raw_smiles, name in _RAW_NAMES.items():
        try:
            mol = Chem.MolFromSmiles(raw_smiles)
            if mol is None:
                continue
            canonical = Chem.MolToSmiles(mol)
            result[canonical] = name
        except Exception:
            continue
    return result


_CANONICAL_TO_NAME: dict[str, str] = _build_canonical_dict()

_PUBCHEM_CACHE: dict[str, str | None] = {}
_PUBCHEM_CACHE_MAXSIZE = 512


def _canonicalize(smiles: str) -> str | None:
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol) if mol is not None else None
    except Exception:
        return None


def _pubchem_lookup(smiles: str) -> str | None:
    """Single blocking PubChem REST call; returns IUPAC name or None."""
    import urllib.request
    import urllib.parse
    import urllib.error

    encoded = urllib.parse.quote(smiles, safe="")
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{encoded}/property/IUPACName/TXT"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            return resp.read().decode("utf-8").strip().splitlines()[0]
    except Exception:
        return None


def resolve_name(smiles: str, *, pubchem: bool = False) -> str | None:
    """Return a human-readable name for *smiles*, or ``None`` if unknown.

    Checks the local dictionary first (instant).  When *pubchem* is ``True``
    and the local lookup misses, makes a single PubChem REST call and caches
    the result for the session.
    """
    canonical = _canonicalize(smiles)
    if canonical is None:
        return None
    name = _CANONICAL_TO_NAME.get(canonical)
    if name is not None:
        return name
    if not pubchem:
        return None
    if canonical in _PUBCHEM_CACHE:
        return _PUBCHEM_CACHE[canonical]
    fetched = _pubchem_lookup(canonical)
    if len(_PUBCHEM_CACHE) >= _PUBCHEM_CACHE_MAXSIZE:
        _PUBCHEM_CACHE.pop(next(iter(_PUBCHEM_CACHE)))
    _PUBCHEM_CACHE[canonical] = fetched
    return fetched


def display_name(smiles: str, *, pubchem: bool = False) -> str:
    """Return ``"name (SMILES)"`` if name is known, else just the SMILES."""
    name = resolve_name(smiles, pubchem=pubchem)
    if name:
        return f"{name} ({smiles})"
    return smiles
