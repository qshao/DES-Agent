"""Offline molecule name → SMILES resolution.

Looks up common molecule names and synonyms from a bundled dictionary.
Also accepts and canonicalises valid SMILES strings directly.
No network access required.
"""
from __future__ import annotations

import difflib
import json
from pathlib import Path

from rdkit import Chem

_DICT_PATH = Path(__file__).resolve().parents[2] / "artifacts" / "molecule_names" / "common_names.json"


def _normalise(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _build_index() -> tuple[dict[str, str], list[dict]]:
    data = json.loads(_DICT_PATH.read_text(encoding="utf-8"))
    index: dict[str, str] = {}
    entries = data["entries"]
    for entry in entries:
        for name in entry["names"]:
            index[_normalise(name)] = entry["smiles"]
    return index, entries


try:
    _NAME_INDEX, _ENTRIES = _build_index()
except Exception:
    _NAME_INDEX, _ENTRIES = {}, []


def resolve_name(text: str) -> str | None:
    """Return canonical SMILES if *text* is a known name/synonym, else None.

    Does not check whether *text* is a valid SMILES — call resolve_to_smiles
    for the combined pass-through-or-lookup behaviour.
    """
    return _NAME_INDEX.get(_normalise(text))


def resolve_to_smiles(text: str) -> str:
    """Return canonical SMILES for *text*, which may be a SMILES or a name.

    Resolution order:
      1. If *text* is a valid SMILES, canonicalise and return it.
      2. If *text* matches a known name or synonym, return the dictionary SMILES.
      3. Raise ValueError with a user-friendly message including a 'did you mean'
         suggestion when the input is close to a known name.
    """
    if not text or not text.strip():
        raise ValueError(
            "Unknown molecule: ''"
            "\n  → Run 'des-agent list-molecules' to see all supported names."
            "\n  → If you have a SMILES string, pass that directly instead."
        )

    mol = Chem.MolFromSmiles(text)
    if mol is not None:
        return Chem.MolToSmiles(mol)

    key = _normalise(text)
    if key in _NAME_INDEX:
        return _NAME_INDEX[key]

    suggestion = ""
    close = difflib.get_close_matches(key, _NAME_INDEX.keys(), n=1, cutoff=0.75)
    if close:
        matched_smiles = _NAME_INDEX[close[0]]
        suggestion = f"\n  → Did you mean {close[0]!r}?  (SMILES: {matched_smiles})"

    raise ValueError(
        f"Unknown molecule: {text!r}"
        f"{suggestion}"
        f"\n  → Run 'des-agent list-molecules' to see all supported names."
        f"\n  → If you have a SMILES string, pass that directly instead."
    )


def list_molecules() -> list[dict]:
    """Return all dictionary entries sorted by role then canonical name.

    Each dict has keys: smiles, canonical_name, synonyms, role.
    """
    result = []
    for entry in _ENTRIES:
        result.append({
            "smiles": entry["smiles"],
            "canonical_name": entry["names"][0],
            "synonyms": entry["names"][1:],
            "role": entry["role"],
        })
    return sorted(result, key=lambda e: (e["role"], e["canonical_name"]))
