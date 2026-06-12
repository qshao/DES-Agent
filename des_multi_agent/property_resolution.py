from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

from .schemas import MeltingPointEstimate

# Confidence assigned to each source, highest-trust first.
_EXPERIMENTAL_CONFIDENCE = 0.95
_HEURISTIC_CONFIDENCE = 0.35
# QSPR confidence is derived from the ensemble spread and bounded strictly
# between the heuristic and experimental layers.
_QSPR_CONFIDENCE_MAX = 0.85
_QSPR_CONFIDENCE_MIN = 0.40
_QSPR_STD_SCALE_K = 40.0  # ensemble std (K) at which confidence hits the floor

_EXPERIMENTAL_TABLE_PATH = (
    Path(__file__).resolve().parents[1] / "artifacts" / "melting_points" / "experimental.json"
)


@lru_cache(maxsize=1)
def _experimental_table() -> dict[str, dict]:
    """Load the InChIKey-keyed experimental melting-point table.

    Returns an empty mapping when the artifact is absent so the resolver
    degrades gracefully to prediction/heuristic layers.
    """
    try:
        with open(_EXPERIMENTAL_TABLE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return dict(data.get("entries", {}))
    except (FileNotFoundError, ValueError):
        return {}


def _lookup_experimental(mol) -> float | None:
    table = _experimental_table()
    if not table:
        return None
    key = Chem.MolToInchiKey(mol)
    entry = table.get(key)
    if entry is None:
        return None
    return float(entry["tm_k"])


def _resolve_mp_device() -> str:
    """Device for the QSPR melting-point model.

    Defaults to ``cpu`` (the model is tiny) so it stays off the GPU even when the
    DES eutectic stage runs on ``cuda`` via ``--ml-device`` — keeping the GPU free
    for a local LLM. Override with ``DES_MP_DEVICE``.
    """
    return os.environ.get("DES_MP_DEVICE", "cpu")


@lru_cache(maxsize=1)
def _qspr_model():
    """Lazily load the QSPR melting-point ensemble (cached for the process).

    Returns ``None`` when disabled via ``DES_DISABLE_QSPR`` or when the artifact
    or its dependencies are unavailable, so resolution degrades to the heuristic.
    """
    if os.environ.get("DES_DISABLE_QSPR"):
        return None
    try:
        from .predictors.melting_point import load_qspr_model

        return load_qspr_model(device=_resolve_mp_device())
    except Exception:
        return None


def clear_resolver_caches() -> None:
    """Reset the cached experimental table and QSPR model.

    Useful in long-lived processes when an artifact is regenerated or
    ``DES_DISABLE_QSPR`` / ``DES_MP_DEVICE`` change mid-run. Guarded so it is a
    no-op if either function has been monkeypatched without an lru cache.
    """
    for fn in (_experimental_table, _qspr_model):
        getattr(fn, "cache_clear", lambda: None)()


def _qspr_confidence(std_k: float) -> float:
    frac = min(max(std_k, 0.0), _QSPR_STD_SCALE_K) / _QSPR_STD_SCALE_K
    return _QSPR_CONFIDENCE_MAX - frac * (_QSPR_CONFIDENCE_MAX - _QSPR_CONFIDENCE_MIN)


def _heuristic_tm_k(mol) -> float:
    heavy_atoms = float(mol.GetNumHeavyAtoms())
    ring_count = float(rdMolDescriptors.CalcNumRings(mol))
    tpsa = float(rdMolDescriptors.CalcTPSA(mol))
    hbd = float(Lipinski.NumHDonors(mol))
    hba = float(Lipinski.NumHAcceptors(mol))
    logp = float(Crippen.MolLogP(mol))
    mol_wt = float(Descriptors.MolWt(mol))
    formal_charge = float(sum(atom.GetFormalCharge() for atom in mol.GetAtoms()))

    tm_k = (
        245.0
        + 6.5 * heavy_atoms
        + 4.5 * ring_count
        + 0.04 * tpsa
        + 3.0 * hbd
        + 1.5 * hba
        - 2.5 * logp
        + 0.03 * mol_wt
        - 5.0 * abs(formal_charge)
    )
    return max(150.0, float(tm_k))


def resolve_melting_point(component: str, override_k: float | None = None) -> MeltingPointEstimate:
    """Resolve a pure-component melting point, most-trusted source first.

    Resolution order:
      1. explicit override (confidence 1.0)
      2. experimental lookup table keyed by InChIKey (confidence 0.95)
      3. RDKit descriptor heuristic (confidence 0.35)
    """
    if override_k is not None:
        return MeltingPointEstimate(
            component=component,
            tm_k=float(override_k),
            source="override",
            confidence=1.0,
        )

    mol = Chem.MolFromSmiles(component)
    if mol is None:
        raise ValueError("Invalid component SMILES")

    experimental = _lookup_experimental(mol)
    if experimental is not None:
        return MeltingPointEstimate(
            component=component,
            tm_k=experimental,
            source="experimental",
            confidence=_EXPERIMENTAL_CONFIDENCE,
        )

    model = _qspr_model()
    if model is not None:
        try:
            pred = model.predict(component)
            return MeltingPointEstimate(
                component=component,
                tm_k=float(pred.tm_k),
                source="qspr",
                confidence=_qspr_confidence(float(pred.std_k)),
            )
        except Exception:
            pass  # fall through to the heuristic layer

    return MeltingPointEstimate(
        component=component,
        tm_k=_heuristic_tm_k(mol),
        source="heuristic",
        confidence=_HEURISTIC_CONFIDENCE,
    )
