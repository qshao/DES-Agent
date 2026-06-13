from des_multi_agent.llm.config import LLMConfig


def test_llm_config_defaults():
    cfg = LLMConfig()
    assert cfg.enabled is False
    assert cfg.provider == "disabled"
    assert cfg.max_candidates == 20
    assert cfg.diversity_mode == "balanced"
    assert cfg.max_families == 6
    assert cfg.family_bias_strength == 0.5


def test_llm_config_from_mapping_reads_diversity_controls():
    cfg = LLMConfig.from_mapping(
        {
            "enabled": True,
            "provider": "ollama",
            "diversity_mode": "explore",
            "max_families": 4,
            "family_bias_strength": 0.75,
        }
    )
    assert cfg.diversity_mode == "explore"
    assert cfg.max_families == 4
    assert cfg.family_bias_strength == 0.75


def test_llm_config_normalizes_diversity_mode_case():
    cfg = LLMConfig(diversity_mode=" Balanced ")
    assert cfg.diversity_mode == "balanced"
