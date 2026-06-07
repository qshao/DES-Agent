from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..predictors.stability_constants import StabilityConstantPrediction, predict_log_k


@dataclass(frozen=True)
class MetalBindingOutcome:
    metal_ion: str
    ligand_smiles: str
    prediction: StabilityConstantPrediction
    warnings: tuple[str, ...]


def run_metal_binding_workflow(
    metal_ion: str,
    ligand_smiles: str,
    model_path: str | Path | None = None,
    *,
    allow_fallback: bool = False,
) -> MetalBindingOutcome:
    prediction = predict_log_k(metal_ion, ligand_smiles, model_path=model_path, allow_fallback=allow_fallback)
    return MetalBindingOutcome(
        metal_ion=metal_ion,
        ligand_smiles=ligand_smiles,
        prediction=prediction,
        warnings=prediction.warnings,
    )
