from des_multi_agent.property_resolution import resolve_melting_point


def test_resolve_melting_point_accepts_override():
    estimate = resolve_melting_point("CCO", override_k=310.0)
    assert estimate.tm_k == 310.0
    assert estimate.source == "override"


def test_resolve_melting_point_uses_heuristic_for_valid_smiles():
    estimate = resolve_melting_point("CCO")
    assert estimate.source == "heuristic"
    assert estimate.tm_k > 150.0
