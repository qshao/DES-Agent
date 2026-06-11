from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Dict

import numpy as np
import torch
import yaml

from .config import DEFAULT_RATIO_GRID
from .paths import ensure_ml_des_path, resolve_existing_path


ensure_ml_des_path()

from src.embeddings.morgan import MorganFPEmbedder  # noqa: E402
from src.embeddings.rdkit_desc import RDKitDescriptorEmbedder  # noqa: E402
from src.embeddings.gnn import GNNConfig, GNNEmbedderWrapper  # noqa: E402
from src.utils import get_device  # noqa: E402


@dataclass(frozen=True)
class CurvePrediction:
    smiles_a: str
    smiles_b: str
    ratios: list[float]
    tm_pred_k: list[float]
    t1_k: float
    t2_k: float
    checkpoint_path: str
    ensemble_std_k: list[float] | None = None
    ensemble_checkpoint_count: int | None = None


@dataclass(frozen=True)
class EmbedderBundle:
    kind: str
    embedder: object | None
    gnn_wrapper: object | None
    dim: int


def build_ratio_grid() -> list[float]:
    return list(DEFAULT_RATIO_GRID)


def load_model(ckpt_path: str | Path, device: torch.device):
    from src.models.model import DESPhysicsModel, DESPhysicsModelWithProjector, ProjectorConfig

    ckpt_path = resolve_existing_path(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device)
    if "projector_cfg" in ckpt:
        proj_cfg = ProjectorConfig(**ckpt["projector_cfg"])
        model = DESPhysicsModel(emb_dim_in=int(ckpt["emb_dim_in"]), projector=proj_cfg).to(device)
    else:
        model = DESPhysicsModel(emb_dim_in=int(ckpt["emb_dim_in"])).to(device)

    try:
        model.load_state_dict(ckpt["model_state"])
    except RuntimeError:
        if "projector_cfg" in ckpt:
            proj_cfg = ProjectorConfig(**ckpt["projector_cfg"])
            legacy = DESPhysicsModelWithProjector(emb_dim_in=int(ckpt["emb_dim_in"]), projector=proj_cfg).to(device)
            legacy.load_state_dict(ckpt["model_state"])
            model = legacy
        else:
            raise

    model.eval()
    return model


def _load_embedder(cfg: Dict[str, Any], device: torch.device) -> EmbedderBundle:
    method = str(cfg["method"]).lower()
    if method == "morgan":
        c = cfg["morgan"]
        emb = MorganFPEmbedder(
            radius=int(c["radius"]),
            n_bits=int(c["n_bits"]),
            use_chirality=bool(c.get("use_chirality", True)),
        )
        return EmbedderBundle(kind=method, embedder=emb, gnn_wrapper=None, dim=emb.dim)

    if method == "rdkit":
        c = cfg["rdkit"]
        emb = RDKitDescriptorEmbedder(descriptor_names=c["descriptor_names"])
        return EmbedderBundle(kind=method, embedder=emb, gnn_wrapper=None, dim=emb.dim)

    if method == "gnn":
        from src.embeddings.gnn import MoleculeGNN  # imported lazily

        c = cfg["gnn"]
        gcfg = GNNConfig(
            emb_dim=int(c.get("emb_dim", 256)),
            num_layers=int(c.get("num_layers", 5)),
            hidden_dim=int(c.get("hidden_dim", 128)),
            dropout=float(c.get("dropout", 0.2)),
            conv=str(c.get("conv", "gin")),
            readout=str(c.get("readout", "mean")),
        )
        gnn = MoleculeGNN(gcfg)
        gnn.to(device)
        wrapper = GNNEmbedderWrapper(gnn, device=device)
        return EmbedderBundle(kind=method, embedder=None, gnn_wrapper=wrapper, dim=wrapper.dim)

    if method == "chemberta":
        try:
            from src.embeddings.chemberta import ChemBERTaEmbedder
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "ChemBERTa embedding requires the optional 'transformers' dependency. "
                "Install the full ml_des_mp requirements or switch to a non-ChemBERTa profile."
            ) from exc

        c = cfg["chemberta"]
        emb = ChemBERTaEmbedder(
            model_name=c["model_name"],
            pooling=c.get("pooling", "mean"),
            batch_size=int(c.get("batch_size", 64)),
            max_length=int(c.get("max_length", 256)),
            device=device,
            cache_dir=c.get("cache_dir", None),
        )
        return EmbedderBundle(kind=method, embedder=emb, gnn_wrapper=None, dim=emb.dim)

    raise ValueError(f"Unknown embedding method: {method}")


def _predict_Tm_from_params(d1, d2, W, T1, T2, r, R=8.314):
    eps = 1e-8
    r_c = torch.clamp(r, min=eps, max=1.0 - eps)
    T_ref = (T1 + T2) / 2.0
    ln_gamma1 = (W / (R * T_ref)) * (1 - r_c) ** 2
    ln_gamma2 = (W / (R * T_ref)) * (r_c) ** 2
    ln_a1 = torch.log(r_c) + ln_gamma1
    ln_a2 = torch.log(1 - r_c) + ln_gamma2
    denom1 = torch.clamp(1.0 - (R / d1) * ln_a1, min=0.1)
    denom2 = torch.clamp(1.0 - (R / d2) * ln_a2, min=0.1)
    return torch.max(T1 / denom1, T2 / denom2)


def check_checkpoint_config_compat(
    checkpoint_path: str | Path,
    config: dict,
) -> list[str]:
    """Return a list of warning strings if the checkpoint's training config
    disagrees with *config* on embedding settings.  Returns [] when compatible
    or when the checkpoint carries no metadata to compare against.
    """
    warnings: list[str] = []
    try:
        ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    except Exception:
        return warnings
    ckpt_cfg = ckpt.get("cfg")
    if not ckpt_cfg or not isinstance(ckpt_cfg, dict):
        return warnings
    ckpt_emb = ckpt_cfg.get("embedding", {})
    cfg_emb = config.get("embedding", {})
    ckpt_method = str(ckpt_emb.get("method", "")).lower()
    cfg_method = str(cfg_emb.get("method", "")).lower()
    if ckpt_method and cfg_method and ckpt_method != cfg_method:
        warnings.append(
            f"Checkpoint–config mismatch: embedding method is '{ckpt_method}' in checkpoint "
            f"but '{cfg_method}' in config — predictions may be incorrect"
        )
        return warnings  # n_bits comparison is irrelevant if methods differ
    if ckpt_method == "morgan":
        ckpt_n_bits = ckpt_emb.get("morgan", {}).get("n_bits")
        cfg_n_bits = cfg_emb.get("morgan", {}).get("n_bits")
        if ckpt_n_bits is not None and cfg_n_bits is not None and int(ckpt_n_bits) != int(cfg_n_bits):
            warnings.append(
                f"Checkpoint–config mismatch: morgan n_bits is {ckpt_n_bits} in checkpoint "
                f"but {cfg_n_bits} in config — predictions may be incorrect"
            )
    return warnings


def predict_curve(
    component_a: str,
    component_b: str,
    t1_k: float,
    t2_k: float,
    checkpoint_path: str | Path,
    config_path: str | Path = Path("ml_des_mp") / "config.yaml",
) -> CurvePrediction:
    config_path = resolve_existing_path(config_path)
    checkpoint_path = resolve_existing_path(checkpoint_path, base_dir=config_path.parent)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for warn in check_checkpoint_config_compat(checkpoint_path, cfg):
        print(f"[WARNING] {warn}", file=sys.stderr, flush=True)
    device = get_device(cfg.get("device", "cuda"))
    model = load_model(checkpoint_path, device)
    emb_bundle = _load_embedder(cfg["embedding"], device=device)

    if emb_bundle.kind == "gnn":
        raise ValueError("GNN checkpoints require an end-to-end inference path that is not enabled in v1.")

    x1 = torch.tensor(emb_bundle.embedder.embed([component_a]), device=device)
    x2 = torch.tensor(emb_bundle.embedder.embed([component_b]), device=device)
    t1 = torch.tensor([float(t1_k)], device=device)
    t2 = torch.tensor([float(t2_k)], device=device)

    ratios = build_ratio_grid()
    tm_pred_k: list[float] = []
    with torch.no_grad():
        d1, d2, w = model.forward_params(x1, x2)
        for ratio in ratios:
            r = torch.tensor([ratio], device=device)
            tm_pred_k.append(float(_predict_Tm_from_params(d1, d2, w, t1, t2, r).item()))

    return CurvePrediction(
        smiles_a=component_a,
        smiles_b=component_b,
        ratios=ratios,
        tm_pred_k=tm_pred_k,
        t1_k=float(t1_k),
        t2_k=float(t2_k),
        checkpoint_path=str(checkpoint_path),
    )


def discover_ensemble_checkpoints(runs_dir: str | Path | None = None) -> list[Path]:
    """Return all ``*_best.pt`` checkpoints found in *runs_dir*.

    Defaults to ``ml_des_mp/runs/`` relative to this file's package root.
    """
    if runs_dir is None:
        runs_dir = Path(__file__).parent.parent / "ml_des_mp" / "runs"
    runs_dir = Path(runs_dir)
    if not runs_dir.is_dir():
        return []
    return sorted(runs_dir.glob("*_best.pt"))


def predict_curve_ensemble(
    component_a: str,
    component_b: str,
    t1_k: float,
    t2_k: float,
    checkpoint_paths: list[str | Path],
    config_path: str | Path = Path("ml_des_mp") / "config.yaml",
) -> CurvePrediction:
    """Run *predict_curve* over every checkpoint and return the mean curve.

    ``CurvePrediction.tm_pred_k`` is the per-ratio mean across all folds.
    ``CurvePrediction.ensemble_std_k`` is the per-ratio standard deviation.
    ``CurvePrediction.ensemble_checkpoint_count`` is the number of folds used.
    ``CurvePrediction.checkpoint_path`` is the first checkpoint path (for
    provenance).

    Raises ``ValueError`` when fewer than two checkpoints are supplied.
    """
    if len(checkpoint_paths) < 2:
        raise ValueError(
            f"predict_curve_ensemble requires at least 2 checkpoints; got {len(checkpoint_paths)}"
        )

    all_tm: list[list[float]] = []
    first_ratios: list[float] = []
    first_ckpt = str(checkpoint_paths[0])

    for ckpt in checkpoint_paths:
        curve = predict_curve(
            component_a=component_a,
            component_b=component_b,
            t1_k=t1_k,
            t2_k=t2_k,
            checkpoint_path=ckpt,
            config_path=config_path,
        )
        all_tm.append(curve.tm_pred_k)
        if not first_ratios:
            first_ratios = curve.ratios

    arr = np.array(all_tm)  # shape: (n_checkpoints, n_ratios)
    mean_tm = arr.mean(axis=0).tolist()
    std_tm = arr.std(axis=0, ddof=1).tolist()

    return CurvePrediction(
        smiles_a=component_a,
        smiles_b=component_b,
        ratios=first_ratios,
        tm_pred_k=mean_tm,
        t1_k=float(t1_k),
        t2_k=float(t2_k),
        checkpoint_path=first_ckpt,
        ensemble_std_k=std_tm,
        ensemble_checkpoint_count=len(checkpoint_paths),
    )
