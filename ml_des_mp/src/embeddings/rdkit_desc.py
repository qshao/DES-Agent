from __future__ import annotations
from typing import List, Sequence
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen, Lipinski
from .base import MoleculeEmbedder

# Map descriptor names to callables (keep small + relevant)
_DESCRIPTOR_FUNCS = {
    "MolWt": Descriptors.MolWt,
    "TPSA": rdMolDescriptors.CalcTPSA,
    "MolLogP": Crippen.MolLogP,
    "NumHDonors": Lipinski.NumHDonors,
    "NumHAcceptors": Lipinski.NumHAcceptors,
    "NumRotatableBonds": Lipinski.NumRotatableBonds,
    "RingCount": rdMolDescriptors.CalcNumRings,
    "NumAromaticRings": rdMolDescriptors.CalcNumAromaticRings,
    "FractionCSP3": rdMolDescriptors.CalcFractionCSP3,
    "HeavyAtomCount": Lipinski.HeavyAtomCount,
}

class RDKitDescriptorEmbedder(MoleculeEmbedder):
    def __init__(self, descriptor_names: Sequence[str]):
        self.descriptor_names = list(descriptor_names)
        unknown = [d for d in self.descriptor_names if d not in _DESCRIPTOR_FUNCS]
        if unknown:
            raise ValueError(f"Unknown RDKit descriptor(s): {unknown}")

    @property
    def dim(self) -> int:
        return len(self.descriptor_names)

    def embed(self, smiles: List[str]) -> np.ndarray:
        out = np.zeros((len(smiles), self.dim), dtype=np.float32)
        for i, s in enumerate(smiles):
            m = Chem.MolFromSmiles(s)
            if m is None:
                raise ValueError(f"RDKit failed to parse SMILES: {s}")
            vals = []
            for name in self.descriptor_names:
                vals.append(float(_DESCRIPTOR_FUNCS[name](m)))
            out[i] = np.array(vals, dtype=np.float32)
        # Simple standardization is handled later (projector learns scaling), but
        # we do a mild normalization here to avoid extreme scales:
        # log1p on strictly positive size-ish descriptors
        return out
