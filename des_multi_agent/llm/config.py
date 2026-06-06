from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


_ALLOWED_PROVIDERS = {"disabled", "none", "off", "ollama", "openai", "gemini", "custom_http", "local", "hosted", "openai_chat"}
_SUPPORTED_OLLAMA_MODEL_PREFIXES = ("gemma4:12b", "nemotron-3-nano", "qwen3.6")


def _is_supported_ollama_model(model_name: str | None) -> bool:
    if not model_name:
        return False
    normalized = model_name.strip().lower()
    return any(normalized.startswith(prefix) for prefix in _SUPPORTED_OLLAMA_MODEL_PREFIXES)


@dataclass(frozen=True)
class LLMConfig:
    enabled: bool = False
    provider: str = "disabled"
    model_name: str | None = None
    api_base_url: str | None = None
    api_key_env: str | None = None
    max_candidates: int = 20
    max_tokens: int = 512
    temperature: float = 0.2
    timeout_seconds: float = 30.0

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object] | None) -> "LLMConfig":
        if mapping is None:
            return cls()
        return cls(
            enabled=bool(mapping.get("enabled", False)),
            provider=str(mapping.get("provider", "disabled")),
            model_name=mapping.get("model_name") or None,
            api_base_url=mapping.get("api_base_url") or None,
            api_key_env=mapping.get("api_key_env") or None,
            max_candidates=int(mapping.get("max_candidates", 20)),
            max_tokens=int(mapping.get("max_tokens", 512)),
            temperature=float(mapping.get("temperature", 0.2)),
            timeout_seconds=float(mapping.get("timeout_seconds", 30.0)),
        )

    def validate(self) -> None:
        provider = self.provider.strip().lower()
        if not self.enabled or provider in {"disabled", "none", "off"}:
            return
        if provider not in _ALLOWED_PROVIDERS:
            raise ValueError(f"Unsupported llm.provider: {self.provider}")
        if provider in {"local", "ollama"}:
            missing = []
            if not self.model_name:
                missing.append("model_name")
            if not self.api_base_url:
                missing.append("api_base_url")
            if missing:
                raise ValueError("Ollama LLM config requires " + ", ".join(missing))
            if not _is_supported_ollama_model(self.model_name):
                raise ValueError(
                    "Unsupported Ollama model_name: "
                    f"{self.model_name}; supported models are gemma4:12b, nemotron-3-nano:latest, and qwen3.6"
                )
            return
        if provider in {"openai", "hosted", "openai_chat"}:
            missing = []
            if not self.model_name:
                missing.append("model_name")
            if not self.api_key_env:
                missing.append("api_key_env")
            if missing:
                raise ValueError("OpenAI LLM config requires " + ", ".join(missing))
            return
        if provider == "gemini":
            missing = []
            if not self.model_name:
                missing.append("model_name")
            if not self.api_key_env:
                missing.append("api_key_env")
            if missing:
                raise ValueError("Gemini LLM config requires " + ", ".join(missing))
            return
        if provider == "custom_http":
            missing = []
            if not self.model_name:
                missing.append("model_name")
            if not self.api_base_url:
                missing.append("api_base_url")
            if missing:
                raise ValueError("Custom HTTP LLM config requires " + ", ".join(missing))
            return
        raise ValueError(f"Unsupported llm.provider: {self.provider}")
