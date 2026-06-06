from des_multi_agent.llm.config import LLMConfig


def test_llm_config_defaults():
    cfg = LLMConfig()
    assert cfg.enabled is False
    assert cfg.provider == "disabled"
    assert cfg.max_candidates == 20
