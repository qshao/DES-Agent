from des_multi_agent.chemistry.partner_registry import is_known, known_inchikeys


def test_known_inchikeys_is_nonempty_frozenset():
    keys = known_inchikeys()
    assert isinstance(keys, frozenset)
    assert len(keys) > 50  # 57 curated + hundreds experimental


def test_is_known_true_for_registry_salt():
    # choline chloride salt is in common_names.json
    assert is_known("C[N+](C)(C)CCO.[Cl-]") is True


def test_is_known_true_for_experimental_compound():
    # glycolic acid O=C(O)CO is in melting_points/experimental.json
    assert is_known("O=C(O)CO") is True


def test_is_known_false_for_invented_molecule():
    # a large fused-ring system not in either dataset
    assert is_known("c1ccc2c(c1)c1ccc3ccccc3c1c1ccccc21") is False


def test_is_known_false_on_unparseable_without_raising():
    assert is_known("not_a_smiles((((") is False
