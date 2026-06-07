from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Optional


REQUIRED_FIELDS_BY_WORKFLOW = {
    "des": ("component_a", "n", "checkpoint_path", "config_path"),
    "metal-binding": ("metal_ion", "ligand_smiles", "stability_constant_model_path"),
}


@dataclass(frozen=True)
class RouterJob:
    component_a: Optional[str] = None
    n: Optional[int] = None
    checkpoint_path: Optional[str] = None
    config_path: Optional[str] = None
    llm_config: Optional[str] = None
    discovery_path: Optional[str] = None
    viscosity_model_path: Optional[str] = None
    metal_ion: Optional[str] = None
    ligand_smiles: Optional[str] = None
    stability_constant_model_path: Optional[str] = None

    def missing_required_fields(self, workflow: str) -> list[str]:
        required_fields = REQUIRED_FIELDS_BY_WORKFLOW.get(workflow, ())
        missing = []
        for field_name in required_fields:
            value = getattr(self, field_name, None)
            if value is None or value == "":
                missing.append(field_name)
        return missing


@dataclass(frozen=True)
class RouterResponse:
    workflow: str
    needs_clarification: bool
    clarifying_questions: list[str]
    job: RouterJob | None

    def validate(self) -> None:
        workflow = self.workflow.strip().lower()
        if self.needs_clarification:
            if not self.clarifying_questions:
                raise ValueError("router response requested clarification but provided no questions")
            if self.job is not None:
                raise ValueError("router response with clarification must not include a job")
            return
        if workflow not in REQUIRED_FIELDS_BY_WORKFLOW:
            raise ValueError(f"Unsupported workflow: {self.workflow}")
        if self.job is None:
            raise ValueError("router response must include a job when clarification is not needed")
        missing = self.job.missing_required_fields(workflow)
        if missing:
            raise ValueError(
                "router response job is missing required fields for "
                + workflow
                + ": "
                + ", ".join(missing)
            )
        if self.clarifying_questions:
            raise ValueError("router response included clarifying questions without requesting clarification")

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)
