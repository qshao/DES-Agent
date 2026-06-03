from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List
import numpy as np

class MoleculeEmbedder(ABC):
    @abstractmethod
    def embed(self, smiles: List[str]) -> np.ndarray:
        """Return embeddings as float32 array of shape (n, d)."""
        raise NotImplementedError

    @property
    @abstractmethod
    def dim(self) -> int:
        raise NotImplementedError
