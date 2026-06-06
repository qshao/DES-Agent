from __future__ import annotations

from collections.abc import Mapping

from .client import post_json_chat
from .config import LLMConfig
from .custom_http_provider import CustomHTTPProvider
from .gemini_provider import GeminiProvider
from .hosted_provider import OpenAIProvider
from .local_provider import OllamaProvider
from .nemotron_provider import NemotronProvider
from .provider import LLMProvider
from .qwen_provider import QwenProvider


_PROVIDER_ALIASES = {
    "local": "ollama",
    "hosted": "openai",
    "openai_chat": "openai",
    "nemotron-3-nano": "ollama",
    "nemotron3nano": "ollama",
    "qwen3.6": "ollama",
    "qwen": "ollama",
}


def _normalize_provider_name(name: str) -> str:
    provider = name.strip().lower()
    return _PROVIDER_ALIASES.get(provider, provider)


def _ollama_provider_class(model_name: str):
    normalized = model_name.strip().lower()
    if normalized.startswith("nemotron-3-nano"):
        return NemotronProvider, "nemotron-3-nano:latest"
    if normalized.startswith("qwen3.6"):
        return QwenProvider, "qwen3.6"
    return OllamaProvider, model_name


def build_llm_provider(cfg: Mapping[str, object] | LLMConfig | None, request_fn=None) -> LLMProvider | None:
    llm_cfg = cfg if isinstance(cfg, LLMConfig) else LLMConfig.from_mapping(cfg)
    llm_cfg.validate()
    if not llm_cfg.enabled or _normalize_provider_name(llm_cfg.provider) in {"disabled", "none", "off"}:
        return None

    request_impl = request_fn or post_json_chat
    provider = _normalize_provider_name(llm_cfg.provider)
    if provider == "ollama":
        provider_cls, default_model_name = _ollama_provider_class(str(llm_cfg.model_name or ""))
        return provider_cls(
            model_name=str(llm_cfg.model_name or default_model_name),
            api_base_url=str(llm_cfg.api_base_url or "http://localhost:11434"),
            api_key_env=llm_cfg.api_key_env,
            max_candidates=llm_cfg.max_candidates,
            max_tokens=llm_cfg.max_tokens,
            temperature=llm_cfg.temperature,
            timeout_seconds=llm_cfg.timeout_seconds,
            request_fn=request_impl,
        )
    if provider == "nemotron":
        return NemotronProvider(
            model_name=str(llm_cfg.model_name or "nemotron-3-nano:latest"),
            api_base_url=str(llm_cfg.api_base_url or "http://localhost:11434"),
            api_key_env=llm_cfg.api_key_env,
            max_candidates=llm_cfg.max_candidates,
            max_tokens=llm_cfg.max_tokens,
            temperature=llm_cfg.temperature,
            timeout_seconds=llm_cfg.timeout_seconds,
            request_fn=request_impl,
        )
    if provider == "openai":
        return OpenAIProvider(
            model_name=str(llm_cfg.model_name or ""),
            api_base_url=str(llm_cfg.api_base_url or "https://api.openai.com/v1"),
            api_key_env=llm_cfg.api_key_env,
            max_candidates=llm_cfg.max_candidates,
            max_tokens=llm_cfg.max_tokens,
            temperature=llm_cfg.temperature,
            timeout_seconds=llm_cfg.timeout_seconds,
            request_fn=request_impl,
        )
    if provider == "gemini":
        return GeminiProvider(
            model_name=str(llm_cfg.model_name or ""),
            api_base_url=str(llm_cfg.api_base_url or "https://generativelanguage.googleapis.com"),
            api_key_env=llm_cfg.api_key_env,
            max_candidates=llm_cfg.max_candidates,
            max_tokens=llm_cfg.max_tokens,
            temperature=llm_cfg.temperature,
            timeout_seconds=llm_cfg.timeout_seconds,
            request_fn=request_impl,
        )
    if provider == "custom_http":
        return CustomHTTPProvider(
            model_name=str(llm_cfg.model_name or ""),
            api_base_url=str(llm_cfg.api_base_url or ""),
            api_key_env=llm_cfg.api_key_env,
            max_candidates=llm_cfg.max_candidates,
            max_tokens=llm_cfg.max_tokens,
            temperature=llm_cfg.temperature,
            timeout_seconds=llm_cfg.timeout_seconds,
            request_fn=request_impl,
        )
    raise ValueError(f"Unknown llm.provider: {llm_cfg.provider}")
