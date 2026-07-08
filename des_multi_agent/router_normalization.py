from __future__ import annotations

from pathlib import Path

from .paths import resolve_existing_path

FIELD_ALIASES: dict[str, str] = {
    # component_a
    "molecule": "component_a",
    "target_molecule": "component_a",
    "target_compound": "component_a",
    "target_substance": "component_a",
    "chemical_formula": "component_a",
    # NOTE: "smiles" is DES-only — metal-binding ligand SMILES use "ligand_smiles"/"target_ligand"/"ligand" instead.
    "smiles": "component_a",
    # n
    "num_candidates": "n",
    "max_candidates": "n",
    "candidate_count": "n",
    "candidates": "n",
    # checkpoint_path
    "checkpoint": "checkpoint_path",
    "use_shipped_checkpoint": "checkpoint_path",
    # config_path
    "config": "config_path",
    "vllm_config": "config_path",
    "use_default_config": "config_path",
    # stability_constant_model_path
    "stability_model": "stability_constant_model_path",
    "stability_constant_model": "stability_constant_model_path",
    # metal_ion
    "target_metal": "metal_ion",
    "metal": "metal_ion",
    "ion": "metal_ion",
    # ligand_smiles
    "target_ligand": "ligand_smiles",
    "ligand": "ligand_smiles",
}


def apply_field_aliases(job_data: dict) -> dict:
    # If a payload has two aliases for the same canonical field, the one earlier in
    # this dict's insertion order wins (iteration order below is deterministic).
    out = dict(job_data)
    for alias, canonical in FIELD_ALIASES.items():
        if alias in out and canonical not in out:
            out[canonical] = out.pop(alias)
    return out


def resolve_path_or_default(value, default: Path) -> str:
    if value:
        try:
            resolve_existing_path(str(value))
            return str(value)
        except FileNotFoundError:
            pass
    return str(default)
