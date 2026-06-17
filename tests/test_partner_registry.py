from des_multi_agent.chemistry.partner_registry import (
    MenuEntry,
    is_known,
    known_inchikeys,
    known_partner_menu,
    structural_sanity,
    _inchikey,
)


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


def test_menu_for_hbd_is_nonempty_and_role_serving():
    menu = known_partner_menu("HBD", limit=30)
    assert len(menu) > 0
    assert all(isinstance(e, MenuEntry) for e in menu)
    # every entry must be able to serve an HBD request
    assert all(e.role in ("HBD", "amphoteric") for e in menu)


def test_menu_respects_limit():
    menu = known_partner_menu("HBA", limit=5)
    assert len(menu) <= 5


def test_menu_has_no_duplicate_inchikeys():
    menu = known_partner_menu("amphoteric", limit=100)
    keys = [_inchikey(e.smiles) for e in menu]
    assert len(keys) == len(set(keys))


def test_menu_curated_entries_come_first():
    # curated entries carry human names; auto-tagged ones use the SMILES as name
    menu = known_partner_menu("HBA", limit=100)
    named = [i for i, e in enumerate(menu) if e.display_name != e.smiles]
    auto = [i for i, e in enumerate(menu) if e.display_name == e.smiles]
    if named and auto:
        assert max(named) < min(auto)


def test_menu_amphoteric_request_returns_any_role():
    menu = known_partner_menu("amphoteric", limit=100)
    roles = {e.role for e in menu}
    assert roles  # non-empty; may include HBD, HBA, amphoteric
