from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

from .artifacts import resolve_artifact
from .base import clamp, default_warning_tuple, linear_predict, load_artifact_payload, parse_local_model


@dataclass(frozen=True)
class StabilityConstantPrediction:
    task: str
    value: float
    units: str
    model_name: str
    source: str
    warnings: tuple[str, ...]
    metadata: dict[str, object]
    metal_ion: str
    ligand: str


# (group, period, d_electrons, hsab_softness 0=hard→1=soft)
_METAL_IDENTITY: dict[str, tuple[int, int, int, float]] = {
    "Li+":  (1,  2, 0, 0.0),  "Na+": (1, 3, 0, 0.0),  "K+":  (1, 4, 0, 0.0),
    "Mg2+": (2,  3, 0, 0.0),  "Ca2+": (2, 4, 0, 0.0),  "Ba2+": (2, 6, 0, 0.0),
    "Al3+": (13, 3, 0, 0.0),
    "Mn2+": (7,  4, 5, 0.2),  "Mn3+": (7, 4, 4, 0.2),
    "Fe2+": (8,  4, 6, 0.2),  "Fe3+": (8, 4, 5, 0.1),
    "Co2+": (9,  4, 7, 0.3),  "Co3+": (9, 4, 6, 0.2),
    "Ni2+": (10, 4, 8, 0.3),
    "Cu+":  (11, 4, 10, 0.8), "Cu2+": (11, 4, 9, 0.5),
    "Zn2+": (12, 4, 10, 0.2),
    "Pd2+": (10, 5, 8, 0.8),  "Pt2+": (10, 6, 8, 0.9),
    "Ag+":  (11, 5, 10, 0.9),
    "Cd2+": (12, 5, 10, 0.6),
    "Hg2+": (12, 6, 10, 1.0), "Hg+": (12, 6, 10, 1.0),
    "Pb2+": (14, 6, 0, 0.4),
    "La3+": (3,  6, 0, 0.0),  "Gd3+": (3, 6, 7, 0.0),
}


def _metal_identity_features(metal_ion: str) -> dict[str, float]:
    key = metal_ion.strip()
    row = _METAL_IDENTITY.get(key, (0, 0, 0, 0.0))
    return {
        'metal_group':       float(row[0]),
        'metal_period':      float(row[1]),
        'metal_d_electrons': float(row[2]),
        'metal_hsab_soft':   float(row[3]),
    }


def _parse_metal_charge(metal_ion: str) -> float:
    match = re.search(r'([+-])(\d*)$', metal_ion.strip())
    if match is None:
        return 0.0
    sign = 1.0 if match.group(1) == '+' else -1.0
    magnitude = float(match.group(2) or 1)
    return sign * magnitude


def _ligand_features(ligand: str) -> dict[str, float]:
    mol = Chem.MolFromSmiles(ligand)
    if mol is None:
        raise ValueError(f'Invalid ligand SMILES: {ligand}')
    heavy_atoms = float(mol.GetNumHeavyAtoms())
    tpsa = float(rdMolDescriptors.CalcTPSA(mol))
    hbd = float(Lipinski.NumHDonors(mol))
    hba = float(Lipinski.NumHAcceptors(mol))
    logp = float(Crippen.MolLogP(mol))
    rings = float(rdMolDescriptors.CalcNumRings(mol))
    rot_bonds = float(Lipinski.NumRotatableBonds(mol))
    mol_wt = float(Descriptors.MolWt(mol))
    formal_charge = float(sum(atom.GetFormalCharge() for atom in mol.GetAtoms()))
    aromatic_atoms = float(sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic()))
    return {
        'ligand_heavy_atoms': heavy_atoms,
        'ligand_tpsa': tpsa,
        'ligand_hbd': hbd,
        'ligand_hba': hba,
        'ligand_logp': logp,
        'ligand_rings': rings,
        'ligand_rot_bonds': rot_bonds,
        'ligand_mol_wt': mol_wt,
        'ligand_formal_charge': formal_charge,
        'ligand_aromatic_atoms': aromatic_atoms,
    }


def _pair_features(metal_ion: str, ligand: str) -> dict[str, float]:
    features = _ligand_features(ligand)
    features['metal_charge'] = _parse_metal_charge(metal_ion)
    features['metal_symbol_len'] = float(len(re.sub(r'[^A-Za-z]', '', metal_ion)))
    features['metal_token_len'] = float(len(metal_ion))
    features['abs_metal_charge'] = abs(features['metal_charge'])
    features['ligand_to_metal_size_ratio'] = features['ligand_heavy_atoms'] / max(1.0, features['metal_symbol_len'])
    return features


def _heuristic_pair_features(metal_ion: str, ligand: str) -> dict[str, float]:
    features = _pair_features(metal_ion, ligand)
    features.update(_metal_identity_features(metal_ion))
    # Pre-compute HSAB cross-term for use by both heuristic and linear artifact model
    features['metal_hsab_soft_x_aromatic'] = (
        features['metal_hsab_soft'] * features['ligand_aromatic_atoms']
    )
    return features


def _heuristic_log_k(features: dict[str, float]) -> float:
    # HSAB cross-term: soft metals (high hsab_soft) benefit from aromatic N-donors (high aromatic_atoms)
    # and S-donors (approximated via low tpsa/hba ratio with high logp)
    hsab_aromatic = features.get('metal_hsab_soft_x_aromatic',
                                  features.get('metal_hsab_soft', 0.0) * features['ligand_aromatic_atoms'])
    raw = (
        4.5
        + 0.18 * features['ligand_hbd']
        + 0.22 * features['ligand_hba']
        + 0.04 * features['ligand_tpsa']
        + 0.06 * features['ligand_rings']
        + 0.01 * features['ligand_heavy_atoms']
        + 0.05 * features['ligand_aromatic_atoms']
        + 0.35 * features['abs_metal_charge']
        + 0.03 * features['metal_token_len']
        - 0.03 * features['ligand_logp']
        + 0.05 * features['ligand_to_metal_size_ratio']
        - 0.02 * abs(features['ligand_formal_charge'])
        + 0.12 * hsab_aromatic
        + 0.08 * features.get('metal_d_electrons', 0.0)
        + 0.04 * features.get('metal_period', 0.0)
    )
    return clamp(raw, -5.0, 20.0)


def _predict_from_model(model: Any, features: dict[str, float]) -> float:
    ordered = [features[key] for key in sorted(features)]
    if isinstance(model, dict) and model.get('kind') == 'callable' and hasattr(model.get('model'), 'predict'):
        prediction = model['model'].predict([ordered])
        if isinstance(prediction, (list, tuple)):
            return float(prediction[0])
        return float(prediction)
    if hasattr(model, 'predict'):
        try:
            prediction = model.predict([ordered])
            if isinstance(prediction, (list, tuple)):
                return float(prediction[0])
            return float(prediction)
        except (ValueError, TypeError, AttributeError):
            pass
    if isinstance(model, dict):
        return linear_predict(model, features)
    raise TypeError(f'Unsupported stability model object: {type(model)!r}')


def predict_log_k(
    metal_ion: str,
    ligand: str,
    model_path: str | Path | None = None,
    *,
    allow_fallback: bool = False,
) -> StabilityConstantPrediction:
    features = _pair_features(metal_ion, ligand)
    warnings: list[str] = []
    artifact_path: Path | None = None
    model = None
    # Prefer the bundled artifact when available; fall back only if loading fails and fallback is allowed.
    try:
        artifact_path = resolve_artifact(model_path, 'metal_binding')
        payload = load_artifact_payload(artifact_path)
        model = parse_local_model(payload)
    except Exception as exc:
        warnings.append(f'Stability constant model unavailable: {exc}')
        if not allow_fallback:
            raise
        model = None
    if model is None:
        value = _heuristic_log_k(_heuristic_pair_features(metal_ion, ligand))
        source = 'heuristic-fallback'
    else:
        try:
            # For linear dict models, pass enriched features so metal-identity coefficients are populated.
            # For positional (sklearn/callable) models, keep the original base features to preserve
            # the feature vector order the model was trained on.
            if isinstance(model, dict) and model.get('kind') == 'linear':
                model_features = _heuristic_pair_features(metal_ion, ligand)
            else:
                model_features = features
            value = clamp(_predict_from_model(model, model_features), -5.0, 20.0)
            source = 'artifact'
        except Exception as exc:
            warnings.append(f'Stability constant prediction failed: {exc}')
            if not allow_fallback:
                raise
            value = _heuristic_log_k(_heuristic_pair_features(metal_ion, ligand))
            source = 'heuristic-fallback'
    return StabilityConstantPrediction(
        task='stability_constant',
        value=float(value),
        units='log K',
        model_name='stabilityconstant-ml-models',
        source=source,
        warnings=default_warning_tuple(warnings),
        metadata={
            'metal_ion': metal_ion,
            'ligand': ligand,
            'model_path': str(artifact_path) if artifact_path is not None else None,
        },
        metal_ion=metal_ion,
        ligand=ligand,
    )
