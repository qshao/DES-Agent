from des_multi_agent.llm.local_provider import OllamaProvider
from des_multi_agent.llm.hosted_provider import OpenAIProvider
from des_multi_agent.llm.gemini_provider import GeminiProvider
from des_multi_agent.llm.custom_http_provider import CustomHTTPProvider
from des_multi_agent.llm.nemotron_provider import NemotronProvider
from des_multi_agent.llm.qwen_provider import QwenProvider
from des_multi_agent.llm.vllm_provider import VLLMProvider


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


def test_nemotron_provider_exposes_request_profile():
    assert NemotronProvider.request_profile.name == "Nemotron"
    assert NemotronProvider.request_profile.path_template == "/api/chat"
    assert NemotronProvider.request_profile.payload_style == "ollama"
    assert NemotronProvider.request_profile.api_key_in_header is True
    assert NemotronProvider.request_profile.api_key_in_query is False


def test_qwen_provider_exposes_request_profile():
    assert QwenProvider.request_profile.name == "Qwen"
    assert QwenProvider.request_profile.path_template == "/api/chat"
    assert QwenProvider.request_profile.payload_style == "ollama"
    assert QwenProvider.request_profile.api_key_in_header is True
    assert QwenProvider.request_profile.api_key_in_query is False


def test_ollama_provider_payload_disables_thinking():
    provider = OllamaProvider(
        model_name="gemma4:12b",
        api_base_url="http://localhost:11434",
    )
    payload = provider.build_payload("Reply with just OK")
    assert payload["think"] is False
    assert payload["stream"] is False


def test_nemotron_provider_payload_disables_thinking():
    provider = NemotronProvider(
        model_name="nemotron-3-nano:latest",
        api_base_url="http://localhost:11434",
    )
    payload = provider.build_payload("Reply with just OK")
    assert payload["think"] is False
    assert payload["stream"] is False


def test_qwen_provider_payload_disables_thinking():
    provider = QwenProvider(
        model_name="qwen3.6",
        api_base_url="http://localhost:11434",
    )
    payload = provider.build_payload("Reply with just OK")
    assert payload["think"] is False
    assert payload["stream"] is False


def test_vllm_provider_exposes_request_profile():
    assert VLLMProvider.request_profile.name == "vLLM"
    assert VLLMProvider.request_profile.path_template == "/chat/completions"
    assert VLLMProvider.request_profile.payload_style == "openai"
    assert VLLMProvider.request_profile.api_key_in_header is True
    assert VLLMProvider.request_profile.api_key_in_query is False
