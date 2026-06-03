from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

from .schemas import MeltingPointEstimate


def resolve_melting_point(component: str, override_k: float | None = None) -> MeltingPointEstimate:
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
    tm_k = max(150.0, float(tm_k))

    return MeltingPointEstimate(
        component=component,
        tm_k=tm_k,
        source="heuristic",
        confidence=0.35,
    )
