from types import SimpleNamespace

from des_multi_agent.reporting import format_metal_binding_report, format_report
from des_multi_agent.workflows.metal_binding import MetalBindingOutcome


def test_format_report_includes_viscosity_section():
    result = SimpleNamespace(curve=SimpleNamespace(smiles_b="O"), is_des=True, min_tm_k=200.0, rationale="demo")
    output = format_report(
        [result],
        viscosity_predictions=[SimpleNamespace(value=12.3, units="mPa*s", model_name="DESignSolvents", source="artifact", metadata={"component_a": "CCO", "component_b": "O"})],
    )
    assert "Viscosity predictions:" in output
    assert "smiles_a | smiles_b | viscosity | units | model | source" in output


def test_format_metal_binding_report():
    prediction = SimpleNamespace(value=6.78, units="log K", model_name="stabilityconstant-ml-models", source="artifact", warnings=())
    outcome = MetalBindingOutcome(metal_ion="Cu2+", ligand_smiles="NCCN", prediction=prediction, warnings=())
    report = format_metal_binding_report(outcome)
    assert "metal_ion | ligand_smiles | value | units | model | source" in report
    assert "Cu2+ | NCCN | 6.78" in report
