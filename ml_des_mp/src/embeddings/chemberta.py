from __future__ import annotations
from typing import List, Optional
import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from .base import MoleculeEmbedder

class ChemBERTaEmbedder(MoleculeEmbedder):
    def __init__(self, model_name: str, pooling: str = "mean", batch_size: int = 64,
                 max_length: int = 256, device: torch.device | str = "cpu",
                 cache_dir: Optional[str] = None):
        self.model_name = model_name
        self.pooling = pooling
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.cache_dir = cache_dir

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # Determine hidden size
        self._dim = int(self.model.config.hidden_size)

        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

    @property
    def dim(self) -> int:
        return self._dim

    def _cache_path(self, smiles: str) -> str:
        safe = smiles.replace("/", "_").replace("\\", "_")
        return os.path.join(self.cache_dir, f"{hash(safe)}.npy")

    def embed(self, smiles: List[str]) -> np.ndarray:
        # Optional per-SMILES caching (useful on HPC if you rerun many splits).
        if self.cache_dir:
            embs = []
            missing = []
            missing_idx = []
            for i, s in enumerate(smiles):
                p = self._cache_path(s)
                if os.path.exists(p):
                    embs.append(np.load(p))
                else:
                    embs.append(None)
                    missing.append(s)
                    missing_idx.append(i)
            if missing:
                new_embs = self._embed_nocache(missing)
                for s, e in zip(missing, new_embs):
                    np.save(self._cache_path(s), e)
                # fill
                it = iter(new_embs)
                for i in missing_idx:
                    embs[i] = next(it)
            arr = np.stack(embs).astype(np.float32)
            return arr

        return self._embed_nocache(smiles).astype(np.float32)

    def _embed_nocache(self, smiles: List[str]) -> np.ndarray:
        outs = []
        with torch.no_grad():
            for i in tqdm(range(0, len(smiles), self.batch_size), desc="ChemBERTa embedding", leave=False):
                batch = smiles[i:i+self.batch_size]
                inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                last = outputs.last_hidden_state  # (B, L, H)
                if self.pooling == "cls":
                    emb = last[:, 0, :]
                else:
                    emb = last.mean(dim=1)
                outs.append(emb.detach().cpu().numpy())
        return np.concatenate(outs, axis=0)
