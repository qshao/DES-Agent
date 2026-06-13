from des_multi_agent.llm.factory import build_llm_provider
import json
import pytest


def test_provider_disabled_returns_none():
    provider = build_llm_provider({"enabled": False})
    assert provider is None


def test_provider_ollama_returns_ollama_provider():
    provider = build_llm_provider(
        {
            "enabled": True,
            "provider": "ollama",
            "model_name": "gemma4:12b",
            "api_base_url": "http://localhost:11434",
        },
        request_fn=lambda *args, **kwargs: '{"message":{"content":"[]"}}',
    )
    assert provider.__class__.__name__ == "OllamaProvider"


def test_provider_ollama_returns_qwen_provider():
    provider = build_llm_provider(
        {
            "enabled": True,
            "provider": "ollama",
            "model_name": "qwen3.6",
            "api_base_url": "http://localhost:11434",
        },
        request_fn=lambda *args, **kwargs: '{"message":{"content":"[]"}}',
    )
    assert provider.__class__.__name__ == "QwenProvider"


def test_provider_ollama_returns_nemotron_provider():
    provider = build_llm_provider(
        {
            "enabled": True,
            "provider": "ollama",
            "model_name": "nemotron-3-nano:latest",
            "api_base_url": "http://localhost:11434",
        },
        request_fn=lambda *args, **kwargs: '{"message":{"content":"[]"}}',
    )
    assert provider.__class__.__name__ == "NemotronProvider"



def test_provider_ollama_reviews_one_candidate():
    review_payload = {
        "smiles": "OCCO",
        "decision": "keep",
        "confidence": 0.87,
        "rationale": "Good candidate.",
        "notes": ["stable"],
    }
    provider = build_llm_provider(
        {
            "enabled": True,
            "provider": "ollama",
            "model_name": "gemma4:12b",
            "api_base_url": "http://localhost:11434",
        },
        request_fn=lambda *args, **kwargs: json.dumps({"message": {"content": json.dumps(review_payload)}}),
    )
    review = provider.review_candidate("CCO", "OCCO", "context")
    assert review.smiles == "OCCO"
    assert review.decision == "keep"
    assert review.confidence == 0.87
    assert review.rationale == "Good candidate."
    assert review.notes == ["stable"]

def test_provider_openai_returns_openai_provider():
    provider = build_llm_provider(
        {
            "enabled": True,
            "provider": "openai",
            "model_name": "gpt-4.1-mini",
            "api_key_env": "OPENAI_API_KEY",
        },
        request_fn=lambda *args, **kwargs: '{"choices":[{"message":{"content":"[]"}}]}',
    )
    assert provider.__class__.__name__ == "OpenAIProvider"


def test_provider_gemini_returns_gemini_provider():
    provider = build_llm_provider(
        {
            "enabled": True,
            "provider": "gemini",
            "model_name": "gemini-2.0-flash",
            "api_key_env": "GEMINI_API_KEY",
        },
        request_fn=lambda *args, **kwargs: '{"candidates":[{"content":{"parts":[{"text":"[]"}]}}]}',
    )
    assert provider.__class__.__name__ == "GeminiProvider"


def test_provider_custom_http_returns_custom_http_provider():
    provider = build_llm_provider(
        {
            "enabled": True,
            "provider": "custom_http",
            "model_name": "custom-model",
            "api_base_url": "https://api.example.com/v1/chat/completions",
            "api_key_env": "CUSTOM_API_KEY",
        },
        request_fn=lambda *args, **kwargs: '{"choices":[{"message":{"content":"[]"}}]}',
    )
    assert provider.__class__.__name__ == "CustomHTTPProvider"


def test_disabled_provider_can_ignore_unknown_provider_name():
    provider = build_llm_provider({"enabled": False, "provider": "unknown"})
    assert provider is None


def test_provider_rejects_unknown_provider():
    with pytest.raises(ValueError, match="Unsupported"):
        build_llm_provider({"enabled": True, "provider": "unknown"})


def test_provider_receives_diversity_controls():
    provider = build_llm_provider(
        {
            "enabled": True,
            "provider": "ollama",
            "model_name": "gemma4:12b",
            "api_base_url": "http://localhost:11434",
            "diversity_mode": "exploit",
            "max_families": 4,
            "family_bias_strength": 0.75,
        },
        request_fn=lambda *args, **kwargs: '{"message":{"content":"[]"}}',
    )
    assert provider.diversity_mode == "exploit"
    assert provider.max_families == 4
    assert provider.family_bias_strength == 0.75
