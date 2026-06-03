import pytest

from des_multi_agent.llm.config import LLMConfig
from des_multi_agent.cli import load_llm_config


def test_ollama_config_requires_model_and_base_url():
    cfg = LLMConfig(enabled=True, provider="ollama")
    with pytest.raises(ValueError, match="model_name|api_base_url"):
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


def test_cli_loads_nested_llm_config(tmp_path):
    cfg_path = tmp_path / "llm.yaml"
    cfg_path.write_text(
        """
llm:
  enabled: true
  provider: ollama
  model_name: llama3.1
  api_base_url: http://localhost:11434
""",
        encoding="utf-8",
    )
    cfg = load_llm_config(cfg_path)
    assert cfg.provider == "ollama"
    assert cfg.model_name == "llama3.1"
    assert cfg.api_base_url == "http://localhost:11434"
