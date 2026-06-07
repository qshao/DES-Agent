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


def _heuristic_log_k(features: dict[str, float]) -> float:
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
        except Exception:
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
        value = _heuristic_log_k(features)
        source = 'heuristic-fallback'
    else:
        try:
            value = clamp(_predict_from_model(model, features), -5.0, 20.0)
            source = 'artifact'
        except Exception as exc:
            warnings.append(f'Stability constant prediction failed: {exc}')
            if not allow_fallback:
                raise
            value = _heuristic_log_k(features)
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
