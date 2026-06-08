
from des_multi_agent.request_normalization import normalize_request_text
from des_multi_agent.task_router import route_task
from des_multi_agent.task_router_prompts import task_router_prompt


def test_normalize_request_text_extracts_workflow_hint():
    result = normalize_request_text("find DES partners for lidocaine")
    assert result.workflow_hint == "des"
    assert result.compound_hint == "lidocaine"
    assert result.needs_clarification is False


def test_normalize_request_text_flags_salt_free_base_ambiguity():
    result = normalize_request_text("find DES partners for lidocaine hydrochloride")
    assert result.needs_clarification is True
    assert "free base" in " ".join(result.clarifying_questions).lower()


def test_normalize_request_text_handles_metal_binding_intent():
    result = normalize_request_text("predict stability constant for Cu2+ with NCCN")
    assert result.workflow_hint == "metal-binding"
    assert result.metal_ion_hint == "Cu2+"
    assert result.ligand_hint == "NCCN"
    assert result.needs_clarification is False


def test_task_router_prompt_includes_normalized_hints():
    normalized = normalize_request_text("find DES partners for lidocaine")
    prompt = task_router_prompt("find DES partners for lidocaine", normalized=normalized)
    assert "Normalized request hints" in prompt
    assert "workflow_hint: des" in prompt
    assert "compound_hint: lidocaine" in prompt


def test_route_task_passes_normalized_request_to_provider():
    captured = {}

    class _FakeProvider:
        def route_request(self, request, normalized=None):
            captured["request"] = request
            captured["normalized"] = normalized
            return '{"workflow":"clarify","needs_clarification":true,"clarifying_questions":["Which workflow?"],"job":null}'

    response = route_task("find DES partners for lidocaine hydrochloride", provider=_FakeProvider())
    assert response.workflow == "clarify"
    assert captured["request"] == "find DES partners for lidocaine hydrochloride"
    assert captured["normalized"].needs_clarification is True
    assert "free base" in " ".join(captured["normalized"].clarifying_questions).lower()
