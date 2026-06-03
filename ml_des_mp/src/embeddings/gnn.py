from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import torch
import torch.nn as nn

try:
    from torch_geometric.data import Data, Batch
    from torch_geometric.nn import GINConv, GCNConv, global_mean_pool, global_add_pool, global_max_pool
except Exception as e:
    Data = None  # type: ignore
    Batch = None  # type: ignore
    GINConv = None  # type: ignore
    GCNConv = None  # type: ignore
    global_mean_pool = None  # type: ignore
    global_add_pool = None  # type: ignore
    global_max_pool = None  # type: ignore

from rdkit import Chem

_SMILES_CACHE = {}

@dataclass
class GNNConfig:
    emb_dim: int = 256
    num_layers: int = 5
    hidden_dim: int = 128
    dropout: float = 0.2
    conv: str = "gin"   # gin or gcn
    readout: str = "mean"  # mean/sum/max

def _atom_features(atom: Chem.Atom) -> List[float]:
    return [
        atom.GetAtomicNum(),
        atom.GetTotalDegree(),
        atom.GetFormalCharge(),
        int(atom.GetIsAromatic()),
        atom.GetTotalNumHs(),
        int(atom.IsInRing()),
    ]

def smiles_to_pyg(smiles: str) -> "Data":
    if Data is None:
        raise ImportError("torch_geometric is required for GNN embedding. Install torch-geometric.")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit failed to parse SMILES: {smiles}")
    # Add explicit Hs not necessary for simple features.
    x = torch.tensor([_atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)
    edges = []
    for b in mol.GetBonds():
        i = b.GetBeginAtomIdx()
        j = b.GetEndAtomIdx()
        edges.append((i, j))
        edges.append((j, i))
    if len(edges) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return Data(x=x, edge_index=edge_index)

class MoleculeGNN(nn.Module):
    def __init__(self, cfg: GNNConfig, in_dim: int = 6):
        super().__init__()
        if GINConv is None:
            raise ImportError("torch_geometric is required for GNN embedding. Install torch-geometric.")
        self.cfg = cfg
        self.dropout = nn.Dropout(cfg.dropout)
        self.act = nn.ReLU()

        self.node_in = nn.Linear(in_dim, cfg.hidden_dim)

        self.convs = nn.ModuleList()
        for _ in range(cfg.num_layers):
            if cfg.conv == "gcn":
                self.convs.append(GCNConv(cfg.hidden_dim, cfg.hidden_dim))
            else:
                mlp = nn.Sequential(
                    nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
                    nn.ReLU(),
                    nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
                )
                self.convs.append(GINConv(mlp))

        self.node_out = nn.Linear(cfg.hidden_dim, cfg.emb_dim)

    def forward(self, batch: "Batch") -> torch.Tensor:
        x = self.act(self.node_in(batch.x))
        for conv in self.convs:
            x = conv(x, batch.edge_index)
            x = self.act(x)
            x = self.dropout(x)
        x = self.node_out(x)
        if self.cfg.readout == "sum":
            g = global_add_pool(x, batch.batch)
        elif self.cfg.readout == "max":
            g = global_max_pool(x, batch.batch)
        else:
            g = global_mean_pool(x, batch.batch)
        return g

def batch_smiles(smiles: List[str], device: torch.device) -> "Batch":
    if Batch is None:
        raise ImportError("torch_geometric is required for GNN embedding. Install torch-geometric.")
    data_list = []
    for s in smiles:
        if s in _SMILES_CACHE:
            data_list.append(_SMILES_CACHE[s])
        else:
            d = smiles_to_pyg(s)
            _SMILES_CACHE[s] = d
            data_list.append(d)
    batch = Batch.from_data_list(data_list)
    return batch.to(device)

class GNNEmbedderWrapper:
    """A thin wrapper so training code can treat GNN as part of the full model (end-to-end)."""
    def __init__(self, gnn: MoleculeGNN, device: torch.device):
        self.gnn = gnn
        self.device = device

    @property
    def dim(self) -> int:
        return self.gnn.cfg.emb_dim

    def forward(self, smiles: List[str]) -> torch.Tensor:
        b = batch_smiles(smiles, self.device)
        return self.gnn(b)
