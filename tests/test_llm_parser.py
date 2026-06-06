from des_multi_agent.llm.parser import (
    parse_candidate_brainstorms,
    parse_critique_notes,
    parse_explanation_notes,
)
from des_multi_agent.llm.local_provider import OllamaProvider
from des_multi_agent.llm.hosted_provider import OpenAIProvider
from des_multi_agent.llm.gemini_provider import GeminiProvider
from des_multi_agent.llm.custom_http_provider import CustomHTTPProvider
import pytest




def test_parser_accepts_fenced_candidate_json():
    raw = """```json
    [{"smiles":"OCCO","rationale":"polyol","family":"polyol"}]
    ```"""
    items = parse_candidate_brainstorms(raw)
    assert len(items) == 1
    assert items[0].smiles == "OCCO"




def test_parser_strips_thinking_trace_and_fenced_json():
    raw = "Thinking...\n```json\n[{\"smiles\":\"OCCO\",\"rationale\":\"polyol\",\"family\":\"polyol\"}]\n```"
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
