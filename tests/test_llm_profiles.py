from des_multi_agent.llm.local_provider import OllamaProvider
from des_multi_agent.llm.hosted_provider import OpenAIProvider
from des_multi_agent.llm.gemini_provider import GeminiProvider
from des_multi_agent.llm.custom_http_provider import CustomHTTPProvider


def test_ollama_provider_exposes_request_profile():
    assert OllamaProvider.request_profile.name == "Ollama"
    assert OllamaProvider.request_profile.path_template == "/api/chat"
    assert OllamaProvider.request_profile.payload_style == "ollama"
    assert OllamaProvider.request_profile.api_key_in_header is True
    assert OllamaProvider.request_profile.api_key_in_query is False


def test_openai_provider_exposes_request_profile():
    assert OpenAIProvider.request_profile.name == "OpenAI"
    assert OpenAIProvider.request_profile.path_template == "/chat/completions"
    assert OpenAIProvider.request_profile.payload_style == "openai"
    assert OpenAIProvider.request_profile.api_key_in_header is True
    assert OpenAIProvider.request_profile.api_key_in_query is False


def test_gemini_provider_exposes_request_profile():
    assert GeminiProvider.request_profile.name == "Gemini"
    assert GeminiProvider.request_profile.path_template == "/v1beta/models/{model_name}:generateContent"
    assert GeminiProvider.request_profile.payload_style == "gemini"
    assert GeminiProvider.request_profile.api_key_in_header is False
    assert GeminiProvider.request_profile.api_key_in_query is True


def test_custom_http_provider_exposes_request_profile():
    assert CustomHTTPProvider.request_profile.name == "Custom HTTP"
    assert CustomHTTPProvider.request_profile.path_template == ""
    assert CustomHTTPProvider.request_profile.payload_style == "openai"
    assert CustomHTTPProvider.request_profile.api_key_in_header is True
    assert CustomHTTPProvider.request_profile.api_key_in_query is False
