from types import SimpleNamespace

from des_multi_agent.chemical_lesson_summary import ChemistryLessonSummary
from des_multi_agent.llm.schemas import ChemistryAssessment, ChemistryNextStep
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


def test_format_report_includes_chemistry_advisor_sections():
    output = format_report(
        [],
        advisor_assessments=[
            ChemistryAssessment(
                smiles="OCCO",
                decision="keep",
                confidence=0.91,
                rationale="Strong H-bonding motif",
                warnings=["phase separation risk"],
            )
        ],
        advisor_next_steps=[
            ChemistryNextStep(
                mode="conservative",
                summary="Tighten family set",
                rationale="Keep search narrow",
            ),
            ChemistryNextStep(
                mode="exploratory",
                summary="Shift donor families",
                rationale="Probe nearby chemistry",
            ),
        ],
    )
    assert "LLM chemistry advisor:" in output
    assert "LLM next steps:" in output
    assert "phase separation risk" in output


def test_format_report_includes_chemistry_lesson_summary():
    output = format_report(
        [],
        chemistry_lesson_summary=ChemistryLessonSummary(
            productive_patterns={"diol": 2},
            avoid_patterns={"aromatic acid": 1},
            cycle_summary=["Productive patterns: diol (2)."],
            run_summary=["Run summary: keep diols near the top."],
            next_steps=["Stay near the productive families."],
            warnings=["Evidence is tentative."],
        ),
    )
    assert "Chemistry lessons:" in output
    assert "productive patterns: diol (2)" in output
    assert "avoid patterns: aromatic acid (1)" in output
    assert "next steps:" in output
    assert "warnings:" in output
