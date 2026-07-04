import pytest

from des_multi_agent.llm.config import LLMConfig
from des_multi_agent.cli import load_llm_config


def test_ollama_config_requires_model_and_base_url():
    cfg = LLMConfig(enabled=True, provider="ollama")
    with pytest.raises(ValueError, match="model_name|api_base_url"):
        cfg.validate()


def test_ollama_config_rejects_unsupported_model_name():
    cfg = LLMConfig(enabled=True, provider="ollama", model_name="mistral", api_base_url="http://localhost:11434")
    with pytest.raises(ValueError, match="Unsupported Ollama model_name"):
        cfg.validate()


def test_provider_nemotron_is_rejected():
    cfg = LLMConfig(enabled=True, provider="nemotron", model_name="nemotron-3-nano:latest", api_base_url="http://localhost:11434")
    with pytest.raises(ValueError, match="Unsupported llm.provider"):
        cfg.validate()


def test_openai_config_requires_model_and_api_key_env():
    cfg = LLMConfig(enabled=True, provider="openai", api_base_url="https://api.openai.com/v1")
    with pytest.raises(ValueError, match="model_name|api_key_env"):
        cfg.validate()


def test_gemini_config_requires_model_and_api_key_env():
    cfg = LLMConfig(enabled=True, provider="gemini")
    with pytest.raises(ValueError, match="model_name|api_key_env"):
        cfg.validate()


def test_custom_http_config_requires_model_and_base_url():
    cfg = LLMConfig(enabled=True, provider="custom_http")
    with pytest.raises(ValueError, match="model_name|api_base_url"):
        cfg.validate()


def test_llm_config_rejects_invalid_diversity_mode():
    cfg = LLMConfig(
        enabled=True,
        provider="ollama",
        model_name="gemma4:12b",
        api_base_url="http://localhost:11434",
        diversity_mode="novelty",
    )
    with pytest.raises(ValueError, match="Unsupported llm.diversity_mode"):
        cfg.validate()


def test_llm_config_rejects_non_positive_max_families():
    cfg = LLMConfig(
        enabled=True,
        provider="ollama",
        model_name="gemma4:12b",
        api_base_url="http://localhost:11434",
        max_families=0,
    )
    with pytest.raises(ValueError, match="llm.max_families must be positive"):
        cfg.validate()


def test_llm_config_rejects_bias_strength_out_of_range():
    cfg = LLMConfig(
        enabled=True,
        provider="ollama",
        model_name="gemma4:12b",
        api_base_url="http://localhost:11434",
        family_bias_strength=1.5,
    )
    with pytest.raises(ValueError, match="llm.family_bias_strength must be in"):
        cfg.validate()


def test_llm_config_accepts_diversity_control_boundaries():
    low = LLMConfig(
        enabled=True,
        provider="ollama",
        model_name="gemma4:12b",
        api_base_url="http://localhost:11434",
        diversity_mode="Balanced",
        max_families=1,
        family_bias_strength=0.0,
    )
    high = LLMConfig(
        enabled=True,
        provider="ollama",
        model_name="gemma4:12b",
        api_base_url="http://localhost:11434",
        diversity_mode="exploit",
        max_families=3,
        family_bias_strength=1.0,
    )
    low.validate()
    high.validate()
    assert low.diversity_mode == "balanced"
    assert high.diversity_mode == "exploit"


def test_cli_loads_nested_llm_config(tmp_path):
    cfg_path = tmp_path / "llm.yaml"
    cfg_path.write_text(
        """
llm:
  enabled: true
  provider: ollama
  model_name: qwen3.6
  api_base_url: http://localhost:11434
""",
        encoding="utf-8",
    )
    cfg = load_llm_config(cfg_path)
    assert cfg.provider == "ollama"
    assert cfg.model_name == "qwen3.6"
    assert cfg.api_base_url == "http://localhost:11434"


def test_vllm_config_requires_model_and_base_url():
    cfg = LLMConfig(enabled=True, provider="vllm")
    with pytest.raises(ValueError, match="model_name|api_base_url"):
        cfg.validate()


def test_vllm_config_accepts_any_model_name():
    cfg = LLMConfig(
        enabled=True,
        provider="vllm",
        model_name="mistral-7b-instruct",
        api_base_url="http://localhost:8000/v1",
    )
    cfg.validate()  # must not raise — no allow-list for vllm model names
