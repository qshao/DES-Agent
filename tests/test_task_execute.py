from types import SimpleNamespace

from des_multi_agent import task_executor
from des_multi_agent.task_executor import execute_task_request_detailed


def test_execute_task_request_returns_clarification_json(monkeypatch):
    class _FakeResponse:
        needs_clarification = True
        job = None

        def to_json(self):
            return '{"workflow":"clarify","needs_clarification":true,"clarifying_questions":["Which workflow?"],"job":null}'

    monkeypatch.setattr(task_executor, "route_task", lambda request, provider=None: _FakeResponse())
    out = task_executor.execute_task_request("find DES partners for lidocaine")
    assert out.startswith("{")
    assert "clarifying_questions" in out


def test_execute_task_request_detailed_distinguishes_clarification_and_execution(monkeypatch):
    class _FakeResponse:
        needs_clarification = True
        job = None

        def to_json(self):
            return '{"workflow":"clarify","needs_clarification":true,"clarifying_questions":["Which workflow?"],"job":null}'

    monkeypatch.setattr("des_multi_agent.task_executor.route_task", lambda request, provider=None: _FakeResponse())
    clarified = execute_task_request_detailed("find DES partners")
    assert clarified.needs_clarification is True
    assert clarified.summary_status == "clarified"
    assert clarified.output.startswith("{")


def test_execute_task_request_detailed_runs_des_workflow(monkeypatch):
    class _FakeJob:
        component_a = "CCO"
        n = 5
        checkpoint_path = "ckpt.pt"
        config_path = "ml_des_mp/config.yaml"
        discovery_path = None
        viscosity_model_path = None

    class _FakeResponse:
        workflow = "des"
        needs_clarification = False
        job = _FakeJob()

        def to_json(self):
            return "{}"

    fake_outcome = SimpleNamespace(
        results=[],
        annotated_results=[],
        candidate_proposals=[],
        candidate_reviews=[],
        explanation_notes=[],
        critique_notes=[],
        brainstorm_candidates=[],
        llm_warnings=[],
        viscosity_predictions=[],
    )

    monkeypatch.setattr(task_executor, "route_task", lambda request, provider=None: _FakeResponse())
    monkeypatch.setattr(task_executor, "run_search_report", lambda **kwargs: fake_outcome)
    monkeypatch.setattr(task_executor, "format_report", lambda *args, **kwargs: "DES REPORT")
    detailed = execute_task_request_detailed("find DES partners for lidocaine")
    assert detailed.needs_clarification is False
    assert detailed.summary_status == "executed"
    assert detailed.output == "DES REPORT"
    out = task_executor.execute_task_request("find DES partners for lidocaine")
    assert out == "DES REPORT"


def test_execute_task_request_runs_metal_binding_workflow(monkeypatch):
    class _FakeJob:
        metal_ion = "Cu2+"
        ligand_smiles = "NCCN"
        stability_constant_model_path = "artifact.json"

    class _FakeResponse:
        workflow = "metal-binding"
        needs_clarification = False
        job = _FakeJob()

        def to_json(self):
            return "{}"

    fake_outcome = SimpleNamespace(
        metal_ion="Cu2+",
        ligand_smiles="NCCN",
        prediction=SimpleNamespace(value=9.89, units="log K", model_name="m", source="artifact", warnings=()),
        warnings=(),
    )

    monkeypatch.setattr(task_executor, "route_task", lambda request, provider=None: _FakeResponse())
    monkeypatch.setattr(task_executor, "run_metal_binding_workflow", lambda **kwargs: fake_outcome)
    monkeypatch.setattr(task_executor, "format_metal_binding_report", lambda outcome: "METAL REPORT")
    detailed = execute_task_request_detailed("predict stability constant")
    assert detailed.needs_clarification is False
    assert detailed.summary_status == "executed"
    assert detailed.output == "METAL REPORT"
    out = task_executor.execute_task_request("predict stability constant")
    assert out == "METAL REPORT"
