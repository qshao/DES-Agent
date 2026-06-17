"""Reality anchoring for DES partner proposals.

Known-set membership (real, attested compounds), a role-tagged anchor menu,
and a structural-sanity gate. Offline + deterministic. Never raises into the
proposal/prompt path.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from rdkit import Chem

from .hbond import hbond_profile

_ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts"
_COMMON_NAMES_PATH = _ARTIFACTS / "molecule_names" / "common_names.json"
_EXPERIMENTAL_PATH = _ARTIFACTS / "melting_points" / "experimental.json"


def _inchikey(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return Chem.MolToInchiKey(mol)
    except Exception:
        return None


@lru_cache(maxsize=1)
def known_inchikeys() -> frozenset[str]:
    """Canonical InChIKey set from common_names.json ∪ experimental.json.

    Recomputes the InChIKey from each stored SMILES so membership uses the same
    key form as incoming proposals. Missing/invalid artifacts degrade gracefully
    to whatever loads.
    """
    keys: set[str] = set()
    try:
        data = json.loads(_COMMON_NAMES_PATH.read_text(encoding="utf-8"))
        for entry in data.get("entries", []):
            k = _inchikey(entry["smiles"])
            if k is not None:
                keys.add(k)
    except Exception:
        pass
    try:
        data = json.loads(_EXPERIMENTAL_PATH.read_text(encoding="utf-8"))
        for record in data.get("entries", {}).values():
            k = _inchikey(record["smiles"])
            if k is not None:
                keys.add(k)
    except Exception:
        pass
    return frozenset(keys)


def is_known(smiles: str) -> bool:
    """True if the canonical InChIKey of `smiles` is in the known set.

    Returns False on unparseable SMILES (never raises).
    """
    k = _inchikey(smiles)
    return k is not None and k in known_inchikeys()
