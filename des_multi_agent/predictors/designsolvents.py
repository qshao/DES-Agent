from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

from .artifacts import resolve_artifact
from .base import clamp, default_warning_tuple, linear_predict, load_artifact_payload, parse_local_model


def _component_features(smiles: str, prefix: str) -> dict[str, float]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f'Invalid SMILES for viscosity predictor: {smiles}')
    heavy_atoms = float(mol.GetNumHeavyAtoms())
    tpsa = float(rdMolDescriptors.CalcTPSA(mol))
    hbd = float(Lipinski.NumHDonors(mol))
    hba = float(Lipinski.NumHAcceptors(mol))
    logp = float(Crippen.MolLogP(mol))
    rings = float(rdMolDescriptors.CalcNumRings(mol))
    rot_bonds = float(Lipinski.NumRotatableBonds(mol))
    mol_wt = float(Descriptors.MolWt(mol))
    formal_charge = float(sum(atom.GetFormalCharge() for atom in mol.GetAtoms()))
    hetero_atoms = float(sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() not in {1, 6}))
    return {
        f'{prefix}_heavy_atoms': heavy_atoms,
        f'{prefix}_tpsa': tpsa,
        f'{prefix}_hbd': hbd,
        f'{prefix}_hba': hba,
        f'{prefix}_logp': logp,
        f'{prefix}_rings': rings,
        f'{prefix}_rot_bonds': rot_bonds,
        f'{prefix}_mol_wt': mol_wt,
        f'{prefix}_formal_charge': formal_charge,
        f'{prefix}_hetero_atoms': hetero_atoms,
    }


def _pair_features(component_a: str, component_b: str) -> dict[str, float]:
    features = {}
    features.update(_component_features(component_a, 'a'))
    features.update(_component_features(component_b, 'b'))
    features['total_heavy_atoms'] = features['a_heavy_atoms'] + features['b_heavy_atoms']
    features['total_tpsa'] = features['a_tpsa'] + features['b_tpsa']
    features['total_hbd'] = features['a_hbd'] + features['b_hbd']
    features['total_hba'] = features['a_hba'] + features['b_hba']
    features['total_logp'] = features['a_logp'] + features['b_logp']
    features['total_rings'] = features['a_rings'] + features['b_rings']
    features['total_rot_bonds'] = features['a_rot_bonds'] + features['b_rot_bonds']
    features['total_mol_wt'] = features['a_mol_wt'] + features['b_mol_wt']
    features['total_formal_charge'] = features['a_formal_charge'] + features['b_formal_charge']
    features['total_hetero_atoms'] = features['a_hetero_atoms'] + features['b_hetero_atoms']
    features['abs_logp_diff'] = abs(features['a_logp'] - features['b_logp'])
    features['abs_tpsa_diff'] = abs(features['a_tpsa'] - features['b_tpsa'])
    return features


@dataclass(frozen=True)
class ViscosityPrediction:
    task: str
    value: float
    units: str
    model_name: str
    source: str
    warnings: tuple[str, ...]
    metadata: dict[str, object]
    component_a: str
    component_b: str


def _heuristic_viscosity(features: dict[str, float]) -> float:
    raw = (
        5.0
        + 0.08 * features['total_heavy_atoms']
        + 0.015 * features['total_tpsa']
        + 0.45 * features['total_hbd']
        + 0.22 * features['total_hba']
        + 0.28 * features['total_rings']
        + 0.12 * features['total_rot_bonds']
        + 0.004 * features['total_mol_wt']
        - 0.25 * features['total_logp']
        + 0.08 * abs(features['total_formal_charge'])
        + 0.05 * features['total_hetero_atoms']
        + 0.10 * features['abs_logp_diff']
        + 0.03 * features['abs_tpsa_diff']
    )
    return clamp(raw, 0.1, 5000.0)


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
    raise TypeError(f'Unsupported viscosity model object: {type(model)!r}')


def predict_viscosity(
    component_a: str,
    component_b: str,
    model_path: str | Path | None = None,
    *,
    allow_fallback: bool = False,
) -> ViscosityPrediction:
    features = _pair_features(component_a, component_b)
    warnings: list[str] = []
    artifact_path: Path | None = None
    model = None
    # Prefer the bundled artifact when available; fall back only if loading fails and fallback is allowed.
    try:
        artifact_path = resolve_artifact(model_path, 'des_viscosity')
        payload = load_artifact_payload(artifact_path)
        model = parse_local_model(payload)
    except Exception as exc:
        warnings.append(f'Viscosity model unavailable: {exc}')
        if not allow_fallback:
            raise
        model = None
    if model is None:
        value = _heuristic_viscosity(features)
        source = 'heuristic-fallback'
    else:
        try:
            value = clamp(_predict_from_model(model, features), 0.1, 5000.0)
            source = 'artifact'
        except Exception as exc:
            warnings.append(f'Viscosity prediction failed: {exc}')
            if not allow_fallback:
                raise
            value = _heuristic_viscosity(features)
            source = 'heuristic-fallback'
    return ViscosityPrediction(
        task='viscosity',
        value=float(value),
        units='mPa*s',
        model_name='DESignSolvents',
        source=source,
        warnings=default_warning_tuple(warnings),
        metadata={
            'component_a': component_a,
            'component_b': component_b,
            'model_path': str(artifact_path) if artifact_path is not None else None,
        },
        component_a=component_a,
        component_b=component_b,
    )
