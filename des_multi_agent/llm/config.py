from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


_ALLOWED_PROVIDERS = {"disabled", "none", "off", "ollama", "openai", "gemini", "custom_http", "vllm"}
_ALLOWED_DIVERSITY_MODES = {"explore", "balanced", "exploit"}
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
    diversity_mode: str = "balanced"
    max_families: int = 6
    family_bias_strength: float = 0.5

    def __post_init__(self) -> None:
        object.__setattr__(self, "provider", self.provider.strip().lower())
        object.__setattr__(self, "diversity_mode", self.diversity_mode.strip().lower())

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
            diversity_mode=str(mapping.get("diversity_mode", "balanced")),
            max_families=int(mapping.get("max_families", 6)),
            family_bias_strength=float(mapping.get("family_bias_strength", 0.5)),
        )

    def validate(self) -> None:
        diversity_mode = self.diversity_mode
        if diversity_mode not in _ALLOWED_DIVERSITY_MODES:
            raise ValueError(
                "Unsupported llm.diversity_mode: "
                f"{self.diversity_mode}; expected one of explore, balanced, exploit"
            )
        if self.max_families <= 0:
            raise ValueError(f"llm.max_families must be positive, got {self.max_families}")
        if not 0.0 <= self.family_bias_strength <= 1.0:
            raise ValueError(
                "llm.family_bias_strength must be in [0.0, 1.0], "
                f"got {self.family_bias_strength}"
            )

        provider = self.provider
        if not self.enabled or provider in {"disabled", "none", "off"}:
            return
        if provider not in _ALLOWED_PROVIDERS:
            raise ValueError(f"Unsupported llm.provider: {self.provider}")
        if provider == "ollama":
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
        if provider == "openai":
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
        if provider == "vllm":
            missing = []
            if not self.model_name:
                missing.append("model_name")
            if not self.api_base_url:
                missing.append("api_base_url")
            if missing:
                raise ValueError("vLLM LLM config requires " + ", ".join(missing))
            return
        raise ValueError(f"Unsupported llm.provider: {self.provider}")
