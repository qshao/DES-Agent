from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict
import torch

from .base import MoleculeEmbedder
from .morgan import MorganFPEmbedder
from .rdkit_desc import RDKitDescriptorEmbedder
from .gnn import GNNConfig, MoleculeGNN, GNNEmbedderWrapper

@dataclass
class EmbedderBundle:
    kind: str
    embedder: MoleculeEmbedder | None
    gnn_wrapper: GNNEmbedderWrapper | None
    dim: int

def build_embedder(cfg: Dict[str, Any], device: torch.device) -> EmbedderBundle:
    method = cfg["method"].lower()
    if method == "chemberta":
        try:
            from .chemberta import ChemBERTaEmbedder
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "ChemBERTa embedding requires the optional 'transformers' dependency. "
                "Install the full ml_des_mp requirements or switch to a non-ChemBERTa profile."
            ) from exc
        c = cfg["chemberta"]
        emb = ChemBERTaEmbedder(
            model_name=c["model_name"],
            pooling=c.get("pooling","mean"),
            batch_size=int(c.get("batch_size",64)),
            max_length=int(c.get("max_length",256)),
            device=device,
            cache_dir=c.get("cache_dir", None),
        )
        return EmbedderBundle(kind=method, embedder=emb, gnn_wrapper=None, dim=emb.dim)

    if method == "morgan":
        c = cfg["morgan"]
        emb = MorganFPEmbedder(radius=int(c["radius"]), n_bits=int(c["n_bits"]), use_chirality=bool(c.get("use_chirality", True)))
        return EmbedderBundle(kind=method, embedder=emb, gnn_wrapper=None, dim=emb.dim)

    if method == "rdkit":
        c = cfg["rdkit"]
        emb = RDKitDescriptorEmbedder(descriptor_names=c["descriptor_names"])
        return EmbedderBundle(kind=method, embedder=emb, gnn_wrapper=None, dim=emb.dim)

    if method == "gnn":
        c = cfg["gnn"]
        gcfg = GNNConfig(
            emb_dim=int(c.get("emb_dim",256)),
            num_layers=int(c.get("num_layers",5)),
            hidden_dim=int(c.get("hidden_dim",128)),
            dropout=float(c.get("dropout",0.2)),
            conv=str(c.get("conv","gin")),
            readout=str(c.get("readout","mean")),
        )
        gnn = MoleculeGNN(gcfg)
        gnn.to(device)
        wrapper = GNNEmbedderWrapper(gnn, device=device)
        return EmbedderBundle(kind=method, embedder=None, gnn_wrapper=wrapper, dim=wrapper.dim)

    raise ValueError(f"Unknown embedding method: {method}")
