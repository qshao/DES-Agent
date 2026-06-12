from __future__ import annotations

import json
import os
import sys
from functools import lru_cache
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

from .schemas import MeltingPointEstimate

# Messages already emitted, so degradation is surfaced once rather than silently
# or repeatedly.
_WARNED_ONCE: set[str] = set()


def _warn_once(message: str) -> None:
    if message in _WARNED_ONCE:
        return
    _WARNED_ONCE.add(message)
    print(f"[WARNING] {message}", file=sys.stderr, flush=True)

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


@lru_cache(maxsize=1)
def _tautomer_enumerator():
    from rdkit.Chem.MolStandardize import rdMolStandardize

    return rdMolStandardize.TautomerEnumerator()


def canonical_inchikey(mol) -> str:
    """InChIKey of the canonical tautomer.

    Standard InChI already merges most mobile-H tautomers; this additionally
    collapses hard cases (e.g. 1,3-dicarbonyl keto/enol). Falls back to the plain
    InChIKey if tautomer canonicalization fails.
    """
    try:
        canon = _tautomer_enumerator().Canonicalize(mol)
    except Exception:
        canon = mol
    return Chem.MolToInchiKey(canon)


def _lookup_experimental(mol) -> float | None:
    table = _experimental_table()
    if not table:
        return None
    # Fast path: plain standard InChIKey (covers all non-tautomeric inputs and
    # the tautomers InChI already normalizes). Only canonicalize on a miss.
    entry = table.get(Chem.MolToInchiKey(mol))
    if entry is None:
        entry = table.get(canonical_inchikey(mol))
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
    except Exception as exc:  # pragma: no cover - exercised via warn-once test
        _warn_once(f"QSPR melting-point model unavailable ({exc}); using heuristic fallback")
        return None


def clear_resolver_caches() -> None:
    """Reset the cached experimental table and QSPR model.

    Useful in long-lived processes when an artifact is regenerated or
    ``DES_DISABLE_QSPR`` / ``DES_MP_DEVICE`` change mid-run. Guarded so it is a
    no-op if either function has been monkeypatched without an lru cache.
    """
    for fn in (_experimental_table, _qspr_model):
        getattr(fn, "cache_clear", lambda: None)()


# The QSPR is trained predominantly on neutral organics; ionic species
# (quaternary-ammonium HBAs, carboxylate salts) extrapolate poorly, so their
# confidence is discounted.
_QSPR_IONIC_PENALTY = 0.75


def _is_ionic(mol) -> bool:
    return any(atom.GetFormalCharge() != 0 for atom in mol.GetAtoms())


def _qspr_confidence(std_k: float, scale_k: float | None = None, ionic: bool = False) -> float:
    """Map ensemble spread to a confidence.

    ``scale_k`` is the ensemble std (K) at which confidence hits the floor. When
    the model provides a data-calibrated scale (90th-pct std on a held-out set)
    it replaces the rough default, making the confidence calibrated rather than
    tied to a magic constant.
    """
    scale = scale_k if scale_k and scale_k > 0 else _QSPR_STD_SCALE_K
    frac = min(max(std_k, 0.0), scale) / scale
    conf = _QSPR_CONFIDENCE_MAX - frac * (_QSPR_CONFIDENCE_MAX - _QSPR_CONFIDENCE_MIN)
    if ionic:
        conf *= _QSPR_IONIC_PENALTY
    return conf


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
            scale_k = getattr(model, "std_scale_k", None)
            return MeltingPointEstimate(
                component=component,
                tm_k=float(pred.tm_k),
                source="qspr",
                confidence=_qspr_confidence(float(pred.std_k), scale_k=scale_k, ionic=_is_ionic(mol)),
            )
        except Exception as exc:
            _warn_once(f"QSPR prediction failed ({exc}); using heuristic fallback")

    return MeltingPointEstimate(
        component=component,
        tm_k=_heuristic_tm_k(mol),
        source="heuristic",
        confidence=_HEURISTIC_CONFIDENCE,
    )
