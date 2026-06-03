from __future__ import annotations
from typing import List
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from .base import MoleculeEmbedder

class MorganFPEmbedder(MoleculeEmbedder):
    def __init__(self, radius: int = 2, n_bits: int = 2048, use_chirality: bool = True):
        self.radius = radius
        self.n_bits = n_bits
        self.use_chirality = use_chirality

    @property
    def dim(self) -> int:
        return self.n_bits

    def embed(self, smiles: List[str]) -> np.ndarray:
        """Return Morgan fingerprint as a 0/1 float32 array (n, n_bits).

        Uses RDKit's ConvertToNumpyArray to avoid dtype/encoding pitfalls.
        """
        arr = np.zeros((len(smiles), self.n_bits), dtype=np.float32)
        for i, s in enumerate(smiles):
            m = Chem.MolFromSmiles(s)
            if m is None:
                raise ValueError(f"RDKit failed to parse SMILES: {s}")
            fp = AllChem.GetMorganFingerprintAsBitVect(
                m,
                self.radius,
                nBits=self.n_bits,
                useChirality=self.use_chirality,
            )
            tmp = np.zeros((self.n_bits,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(fp, tmp)
            arr[i] = tmp.astype(np.float32)
        return arr
