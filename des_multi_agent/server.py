"""Thin FastAPI wrapper around the DES-Agent orchestrators.

Start with:
    python -m des_multi_agent.server [--host HOST] [--port PORT]

Or from another process:
    from des_multi_agent.server import app
    import uvicorn; uvicorn.run(app)

Endpoints
---------
GET  /health
POST /search          — DES screening (wraps run_search_report)
POST /metal-binding   — metal-ion binding (wraps run_metal_binding_workflow)
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import DEFAULT_CONFIG_PATH
from .orchestrator import run_search_report
from .prediction import discover_ensemble_checkpoints
from .reporting import format_report, format_metal_binding_report
from .schemas import DesThresholds
from .uncertainty import UncertaintyPolicy
from .workflows.metal_binding import run_metal_binding_workflow


app = FastAPI(
    title="DES-Agent API",
    description="REST interface for DES screening and metal-ion binding prediction.",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class DESSearchRequest(BaseModel):
    component_a: str = Field(..., description="SMILES of the fixed component (HBA)")
    n: int = Field(20, ge=1, description="Number of candidate HBDs to screen")
    checkpoint_path: str = Field(..., description="Path to a *_best.pt model checkpoint")
    config_path: str = Field(str(DEFAULT_CONFIG_PATH), description="Path to ml_des_mp/config.yaml")
    discovery_path: str | None = Field(None, description="Optional local discovery directory")
    viscosity_model_path: str | None = Field(None)
    save_run_memory_path: str | None = Field(None)
    reuse_run_path: str | None = Field(None)
    output_dir: str | None = Field(None, description="If set, write run artifacts to this directory")
    ensemble: bool = Field(False, description="Use all fold checkpoints for ensemble prediction")
    uncertainty_mode: str = Field("penalize", pattern="^(filter|penalize|report_only)$")
    min_trust_score: float = Field(0.55, ge=0.0, le=1.0)
    soft_penalty_weight: float = Field(0.35, ge=0.0, le=1.0)
    std_high_threshold_k: float = Field(15.0, gt=0)
    std_medium_threshold_k: float = Field(5.0, gt=0)
    absolute_tm_max_k: float = Field(260.0, gt=0)
    relative_drop_min: float = Field(0.1, ge=0.0, le=1.0)


class DESSearchResponse(BaseModel):
    report: str
    results: list[dict[str, Any]]
    llm_warnings: list[str]
    memory_notes: list[str]
    ensemble_folds: int | None = None


class MetalBindingRequest(BaseModel):
    metal_ion: str = Field(..., description="Metal ion symbol, e.g. 'Cu2+' or 'Fe3+'")
    ligand_smiles: str = Field(..., description="SMILES of the ligand")
    stability_constant_model_path: str | None = Field(None)


class MetalBindingResponse(BaseModel):
    report: str
    metal_ion: str
    ligand_smiles: str
    value: float | None = None
    units: str | None = None
    warnings: list[str]


class HealthResponse(BaseModel):
    status: str
    version: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", version=app.version)


@app.post("/search", response_model=DESSearchResponse)
def search(req: DESSearchRequest) -> DESSearchResponse:
    """Screen component B candidates against a fixed component A."""
    try:
        policy = UncertaintyPolicy(
            mode=req.uncertainty_mode,
            min_trust_score=req.min_trust_score,
            soft_penalty_weight=req.soft_penalty_weight,
            std_high_threshold_k=req.std_high_threshold_k,
            std_medium_threshold_k=req.std_medium_threshold_k,
        )
        thresholds = DesThresholds(
            absolute_tm_max_k=req.absolute_tm_max_k,
            relative_drop_min=req.relative_drop_min,
        )
        ensemble_ckpts: list[str] | None = None
        if req.ensemble:
            found = discover_ensemble_checkpoints()
            if len(found) < 2:
                raise HTTPException(
                    status_code=422,
                    detail=f"ensemble=true requires at least 2 checkpoints in ml_des_mp/runs/; found {len(found)}",
                )
            ensemble_ckpts = [str(p) for p in found]

        outcome = run_search_report(
            component_a=req.component_a,
            n=req.n,
            checkpoint_path=req.checkpoint_path,
            config_path=req.config_path,
            thresholds=thresholds,
            uncertainty_policy=policy,
            discovery_path=req.discovery_path,
            viscosity_model_path=req.viscosity_model_path,
            save_run_memory_path=req.save_run_memory_path,
            reuse_run_path=req.reuse_run_path,
            output_dir=req.output_dir,
            ensemble_checkpoints=ensemble_ckpts,
        )
        report = format_report(
            outcome.results,
            annotated_results=outcome.annotated_results,
            candidate_proposals=outcome.candidate_proposals,
            candidate_reviews=outcome.candidate_reviews,
            explanation_notes=outcome.explanation_notes,
            critique_notes=outcome.critique_notes,
            brainstorm_candidates=outcome.brainstorm_candidates,
            llm_warnings=outcome.llm_warnings,
            memory_notes=outcome.memory_notes,
        )
        results_payload = [
            {
                "smiles_b": r.curve.smiles_b,
                "is_des": r.is_des,
                "min_tm_k": r.min_tm_k,
                "rationale": r.rationale,
                "ensemble_folds": r.curve.ensemble_checkpoint_count,
            }
            for r in outcome.results
        ]
        return DESSearchResponse(
            report=report,
            results=results_payload,
            llm_warnings=outcome.llm_warnings,
            memory_notes=outcome.memory_notes,
            ensemble_folds=ensemble_ckpts and len(ensemble_ckpts),
        )
    except HTTPException:
        raise
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/metal-binding", response_model=MetalBindingResponse)
def metal_binding(req: MetalBindingRequest) -> MetalBindingResponse:
    """Predict metal-ion binding stability constant for a ligand."""
    try:
        outcome = run_metal_binding_workflow(
            metal_ion=req.metal_ion,
            ligand_smiles=req.ligand_smiles,
            model_path=req.stability_constant_model_path,
            allow_fallback=False,
        )
        report = format_metal_binding_report(outcome)
        pred = outcome.prediction
        return MetalBindingResponse(
            report=report,
            metal_ion=outcome.metal_ion,
            ligand_smiles=outcome.ligand_smiles,
            value=getattr(pred, "value", None),
            units=getattr(pred, "units", None),
            warnings=list(getattr(pred, "warnings", None) or getattr(outcome, "warnings", None) or []),
        )
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Start the DES-Agent REST API server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="Enable uvicorn auto-reload (dev mode)")
    return parser


def main(argv=None) -> None:
    import uvicorn

    args = _build_arg_parser().parse_args(argv)
    uvicorn.run(
        "des_multi_agent.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
