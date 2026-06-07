from pathlib import Path

from des_multi_agent.reporting import format_metal_binding_report
from des_multi_agent.workflows.metal_binding import run_metal_binding_workflow


def test_metal_binding_workflow_renders_report(tmp_path: Path):
    model_path = tmp_path / "model.json"
    model_path.write_text(
        """
{
  "model_name": "stabilityconstant-ml-models",
  "units": "log K",
  "bias": 5.0,
  "coefficients": {
    "ligand_hbd": 0.2,
    "ligand_hba": 0.3,
    "abs_metal_charge": 0.5
  }
}
""".strip(),
        encoding="utf-8",
    )
    outcome = run_metal_binding_workflow("Cu2+", "NCCN", model_path=model_path, allow_fallback=False)
    report = format_metal_binding_report(outcome)
    assert "metal_ion | ligand_smiles | value | units | model | source" in report
    assert "Cu2+ | NCCN" in report
    assert "stabilityconstant-ml-models" in report
