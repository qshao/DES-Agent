from des_multi_agent.llm.base import _complementary_role


def test_complementary_role_mapping():
    assert _complementary_role("HBA") == "HBD"
    assert _complementary_role("HBD") == "HBA"
    assert _complementary_role("amphoteric") == "amphoteric"
    assert _complementary_role("none") == "amphoteric"
    assert _complementary_role("anything else") == "amphoteric"
