"""Reality anchoring for DES partner proposals.

Known-set membership (real, attested compounds), a role-tagged anchor menu,
and a structural-sanity gate. Offline + deterministic. Never raises into the
proposal/prompt path.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Descriptors

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
    import warnings
    try:
        data = json.loads(_COMMON_NAMES_PATH.read_text(encoding="utf-8"))
        for entry in data.get("entries", []):
            k = _inchikey(entry["smiles"])
            if k is not None:
                keys.add(k)
    except Exception as exc:
        warnings.warn(
            f"partner_registry: could not load {_COMMON_NAMES_PATH}: {exc}; "
            "is_known() will under-report known partners",
            RuntimeWarning,
            stacklevel=2,
        )
    try:
        data = json.loads(_EXPERIMENTAL_PATH.read_text(encoding="utf-8"))
        for record in data.get("entries", {}).values():
            k = _inchikey(record["smiles"])
            if k is not None:
                keys.add(k)
    except Exception as exc:
        warnings.warn(
            f"partner_registry: could not load {_EXPERIMENTAL_PATH}: {exc}; "
            "is_known() will under-report known partners",
            RuntimeWarning,
            stacklevel=2,
        )
    return frozenset(keys)


def is_known(smiles: str) -> bool:
    """True if the canonical InChIKey of `smiles` is in the known set.

    Returns False on unparseable SMILES (never raises).
    """
    k = _inchikey(smiles)
    return k is not None and k in known_inchikeys()


_ALLOWED_ELEMENTS = {"H", "C", "N", "O", "S", "P", "F", "Cl", "Br", "I"}


def structural_sanity(smiles: str) -> tuple[bool, str]:
    """Deterministic 'is this a sane small molecule' check.

    Fails when: unparseable; any atom outside the allowed-element set; any
    radical electrons; molecular weight outside the open interval (40, 400).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False, "invalid SMILES"
    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in _ALLOWED_ELEMENTS:
            return False, f"disallowed element: {atom.GetSymbol()}"
        if atom.GetNumRadicalElectrons() > 0:
            return False, "radical species"
    mw = Descriptors.MolWt(mol)
    if not (40.0 < mw < 400.0):
        return False, f"molecular weight out of range: {mw:.1f}"
    return True, ""


@dataclass(frozen=True)
class MenuEntry:
    smiles: str
    display_name: str   # curated name, or the SMILES for auto-tagged entries
    role: str           # "HBD" | "HBA" | "amphoteric"


def _serves(entry_role: str, wanted: str) -> bool:
    return (
        entry_role == wanted
        or entry_role == "amphoteric"
        or wanted == "amphoteric"
    )


@lru_cache(maxsize=1)
def _all_menu_entries() -> tuple[MenuEntry, ...]:
    """Curated registry entries first, then auto-role-tagged experimental
    compounds, deduped by InChIKey. Built once and cached."""
    entries: list[MenuEntry] = []
    seen: set[str] = set()

    # Curated registry: trust the stored role tag; keep H-bonders only.
    try:
        data = json.loads(_COMMON_NAMES_PATH.read_text(encoding="utf-8"))
        for entry in data.get("entries", []):
            role = entry.get("role", "")
            if role not in ("HBD", "HBA", "amphoteric"):
                continue
            k = _inchikey(entry["smiles"])
            if k is None or k in seen:
                continue
            seen.add(k)
            name = entry["names"][0] if entry.get("names") else entry["smiles"]
            entries.append(MenuEntry(entry["smiles"], name, role))
    except Exception:
        pass

    # Experimental compounds: derive role from the H-bond profiler.
    try:
        data = json.loads(_EXPERIMENTAL_PATH.read_text(encoding="utf-8"))
        for record in data.get("entries", {}).values():
            try:
                smi = record["smiles"]
                k = _inchikey(smi)
                if k is None or k in seen:
                    continue
                role = hbond_profile(smi).role
                if role not in ("HBD", "HBA", "amphoteric"):
                    continue
                seen.add(k)
                entries.append(MenuEntry(smi, smi, role))
            except Exception:
                continue
    except Exception:
        pass

    return tuple(entries)


def _coord_role_summary(prof) -> str:
    """Compact coordination description for the ligand menu."""
    if prof.n_donor_atoms == 0:
        return "no donors"
    elems = ",".join(prof.donor_site_elements[:3])
    if prof.denticity <= 1:
        return f"monodentate ({elems})"
    if prof.denticity == 2:
        return f"bidentate ({elems})"
    return f"polydentate ({elems})"


@lru_cache(maxsize=32)
def _scored_ligand_entries(metal_ion: str) -> tuple[MenuEntry, ...]:
    """Registry molecules with ≥1 donor atom, sorted by predicted log K for *metal_ion*.

    Computed once per metal ion and cached. Uses the internal
    ``_rule_based_log_k_from_profile`` to avoid computing the coordination
    profile twice.
    """
    from .coordination import coordination_profile
    from .stability_rules import _rule_based_log_k_from_profile

    scored: list[tuple[float, MenuEntry]] = []
    for e in _all_menu_entries():
        try:
            prof = coordination_profile(e.smiles)
            if prof.n_donor_atoms == 0:
                continue
            log_k = _rule_based_log_k_from_profile(metal_ion, prof)
            role = _coord_role_summary(prof)
            scored.append((log_k, MenuEntry(e.smiles, e.display_name, role)))
        except Exception:
            continue
    scored.sort(key=lambda t: t[0], reverse=True)
    return tuple(e for _, e in scored)


def known_ligand_menu(metal_ion: str, limit: int = 15) -> list[MenuEntry]:
    """Top *limit* registry molecules sorted by predicted log K for *metal_ion*.

    Each entry's ``role`` field describes the coordination mode (e.g.
    ``"bidentate (N,O)"``). Returns an empty list on any error (never raises).
    """
    try:
        return list(_scored_ligand_entries(metal_ion)[:limit])
    except Exception:
        return []


def known_partner_menu(role: str, limit: int = 30) -> list[MenuEntry]:
    """Menu entries that can serve the wanted partner role `role`.

    An entry serves `role` when its role equals `role`, or either side is
    "amphoteric". Curated entries precede auto-tagged ones; capped at `limit`.
    """
    out: list[MenuEntry] = []
    for e in _all_menu_entries():
        if _serves(e.role, role):
            out.append(e)
            if len(out) >= limit:
                break
    return out
