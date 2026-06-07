from des_multi_agent.llm.parser import (
    extract_json_object,
    parse_candidate_brainstorms,
    parse_candidate_review,
    parse_critique_notes,
    parse_explanation_notes,
)
from des_multi_agent.task_router import parse_router_response
from des_multi_agent.task_router_prompts import task_router_prompt

from des_multi_agent.llm.custom_http_provider import CustomHTTPProvider
from des_multi_agent.llm.gemini_provider import GeminiProvider
from des_multi_agent.llm.hosted_provider import OpenAIProvider
from des_multi_agent.llm.local_provider import OllamaProvider
from des_multi_agent.llm.prompts import candidate_review_prompt
import pytest




def test_parser_accepts_fenced_candidate_json():
    raw = """```json
    [{"smiles":"OCCO","rationale":"polyol","family":"polyol"}]
    ```"""
    items = parse_candidate_brainstorms(raw)
    assert len(items) == 1
    assert items[0].smiles == "OCCO"




def test_parser_strips_thinking_trace_and_fenced_json():
    raw = """Thinking...
```json
[{"smiles":"OCCO","rationale":"polyol","family":"polyol"}]
```"""
    items = parse_candidate_brainstorms(raw)
    assert len(items) == 1
    assert items[0].smiles == "OCCO"

def test_parser_discards_invalid_candidate_entries():
    raw = '[{"smiles":"OCCO","rationale":"polyol","family":"polyol"},{"smiles":"","rationale":"bad","family":"polyol"}]'
    items = parse_candidate_brainstorms(raw)
    assert len(items) == 1
    assert items[0].smiles == "OCCO"


def test_parser_discards_invalid_explanation_entries():
    raw = '[{"smiles":"OCCO","summary":"ranked highly","evidence":["low min Tm"]},{"smiles":"","summary":"bad","evidence":[]}]'
    items = parse_explanation_notes(raw)
    assert len(items) == 1
    assert items[0].summary == "ranked highly"


def test_parser_discards_invalid_critique_entries():
    raw = '[{"smiles":"OCCO","assessment":"advisory only","concerns":["possible outlier"]},{"smiles":"","assessment":"bad","concerns":[]}]'
    items = parse_critique_notes(raw)
    assert len(items) == 1
    assert items[0].assessment == "advisory only"


def test_ollama_provider_rejects_malformed_response():
    raw = '{"unexpected": true}'
    provider = OllamaProvider(
        model_name="llama3.1",
        api_base_url="http://localhost:11434",
        request_fn=lambda *args, **kwargs: raw,
    )
    with pytest.raises(ValueError) as exc:
        provider.brainstorm_candidates("CCO", None, "context")
    message = str(exc.value)
    assert "Ollama" in message
    assert "unexpected" in message


def test_openai_provider_rejects_malformed_response():
    raw = '{"choices": []}'
    provider = OpenAIProvider(
        model_name="gpt-4.1-mini",
        api_base_url="https://api.openai.com/v1",
        request_fn=lambda *args, **kwargs: raw,
    )
    with pytest.raises(ValueError) as exc:
        provider.brainstorm_candidates("CCO", None, "context")
    message = str(exc.value)
    assert "OpenAI" in message
    assert "choices" in message


def test_gemini_provider_rejects_malformed_response():
    raw = '{"candidates": []}'
    provider = GeminiProvider(
        model_name="gemini-2.0-flash",
        api_base_url="https://generativelanguage.googleapis.com",
        request_fn=lambda *args, **kwargs: raw,
    )
    with pytest.raises(ValueError) as exc:
        provider.brainstorm_candidates("CCO", None, "context")
    message = str(exc.value)
    assert "Gemini" in message
    assert "candidates" in message


def test_custom_http_provider_rejects_malformed_response():
    raw = '{"choices": []}'
    provider = CustomHTTPProvider(
        model_name="custom-model",
        api_base_url="https://api.example.com/v1/chat/completions",
        request_fn=lambda *args, **kwargs: raw,
    )
    with pytest.raises(ValueError) as exc:
        provider.brainstorm_candidates("CCO", None, "context")
    message = str(exc.value)
    assert "Custom HTTP" in message
    assert "choices" in message



def test_parse_candidate_review_accepts_strict_json():
    raw = '{"smiles":"OCCO","decision":"keep","confidence":0.87,"rationale":"Good hydrogen bonding candidate.","notes":["Stable","Small polyol"]}'
    review = parse_candidate_review(raw)
    assert review.smiles == "OCCO"
    assert review.decision == "keep"
    assert review.confidence == 0.87
    assert review.rationale == "Good hydrogen bonding candidate."
    assert review.notes == ["Stable", "Small polyol"]


def test_parse_candidate_review_rejects_invalid_decision():
    raw = '{"smiles":"OCCO","decision":"maybe","confidence":0.87,"rationale":"Good hydrogen bonding candidate.","notes":[]}'
    with pytest.raises(ValueError):
        parse_candidate_review(raw)


def test_candidate_review_prompt_requests_one_json_object():
    prompt = candidate_review_prompt(
        component_a="CCN(CC)CC(=O)Nc1c(C)cccc1C",
        candidate_smiles="OCCO",
        context="demo context",
    )
    assert "Return raw JSON only" in prompt
    assert '"smiles": "OCCO"' in prompt
    assert "decision" in prompt
    assert "confidence" in prompt



def test_extract_json_object_returns_json_payload():
    raw = """thinking...
```json
{"workflow":"des"}
```"""
    payload = extract_json_object(raw)
    assert payload == "{\"workflow\":\"des\"}"


def test_parse_router_response_accepts_complete_job():
    payload = (
        "{\"workflow\":\"des\",\"needs_clarification\":false,\"clarifying_questions\":[],"
        "\"job\":{\"component_a\":\"CCO\",\"n\":5,\"checkpoint_path\":\"ckpt.pt\",\"config_path\":\"ml_des_mp/config.yaml\"}}"
    )
    response = parse_router_response(payload)
    assert response.workflow == "des"
    assert response.needs_clarification is False
    assert response.clarifying_questions == []
    assert response.job is not None
    assert response.job.component_a == "CCO"
    assert response.job.n == 5


def test_parse_router_response_accepts_clarification_state():
    payload = (
        "{\"workflow\":\"clarify\",\"needs_clarification\":true,\"clarifying_questions\":[\"Which workflow?\"],"
        "\"job\":null}"
    )
    response = parse_router_response(payload)
    assert response.workflow == "clarify"
    assert response.needs_clarification is True
    assert response.clarifying_questions == ["Which workflow?"]
    assert response.job is None


def test_parse_router_response_ignores_extra_job_fields():
    payload = '{"workflow":"des","needs_clarification":false,"clarifying_questions":[],"job":{"component_a":"CCO","n":5,"checkpoint_path":"ckpt.pt","config_path":"ml_des_mp/config.yaml","model":"gemma4:12b"}}'
    response = parse_router_response(payload)
    assert response.job is not None
    assert response.job.component_a == "CCO"
    assert response.job.n == 5
    assert not hasattr(response.job, "model")


def test_task_router_prompt_mentions_clarify_state():
    prompt = task_router_prompt("find DES partners for lidocaine")
    assert "workflow=\"clarify\"" in prompt
    assert "clarification questions" in prompt.lower() or "clarification" in prompt.lower()



def test_parse_router_response_rejects_missing_des_required_field():
    payload = (
        "{\"workflow\":\"des\",\"needs_clarification\":false,\"clarifying_questions\":[],"
        "\"job\":{\"component_a\":\"CCO\",\"n\":5,\"checkpoint_path\":\"ckpt.pt\"}}"
    )
    with pytest.raises(ValueError, match="missing required fields for des"):
        parse_router_response(payload)


def test_parse_router_response_rejects_job_when_clarifying():
    payload = (
        "{\"workflow\":\"clarify\",\"needs_clarification\":true,\"clarifying_questions\":[\"Which workflow?\"],"
        "\"job\":{\"component_a\":\"CCO\"}}"
    )
    with pytest.raises(ValueError, match="must not include a job"):
        parse_router_response(payload)



def test_parse_router_response_rejects_missing_metal_binding_required_field():
    payload = (
        "{\"workflow\":\"metal-binding\",\"needs_clarification\":false,\"clarifying_questions\":[],"
        "\"job\":{\"metal_ion\":\"Cu2+\",\"ligand_smiles\":\"NCCN\"}}"
    )
    with pytest.raises(ValueError, match="missing required fields for metal-binding"):
        parse_router_response(payload)
