from des_multi_agent.chemistry.partner_registry import is_known, known_inchikeys, structural_sanity


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


def test_structural_sanity_passes_common_des_components():
    for smi in ("CCO", "NC(N)=O", "OCC(O)CO"):  # ethanol, urea, glycerol
        ok, reason = structural_sanity(smi)
        assert ok is True, (smi, reason)
        assert reason == ""


def test_structural_sanity_rejects_disallowed_element():
    ok, reason = structural_sanity("OB(O)O")  # boric acid — boron not allowed
    assert ok is False
    assert "element" in reason.lower()


def test_structural_sanity_rejects_oversized_molecule():
    # long alkane, MW well above 400
    ok, reason = structural_sanity("C" * 40)
    assert ok is False
    assert "weight" in reason.lower()


def test_structural_sanity_rejects_tiny_molecule():
    ok, reason = structural_sanity("C")  # methane, MW ~16 < 40
    assert ok is False
    assert "weight" in reason.lower()


def test_structural_sanity_rejects_radical():
    ok, reason = structural_sanity("[CH3]")  # methyl radical
    assert ok is False
    assert "radical" in reason.lower()


def test_structural_sanity_rejects_invalid_smiles():
    ok, reason = structural_sanity("xyz(((")
    assert ok is False
    assert "invalid" in reason.lower()
