from pathlib import Path

from des_multi_agent.router_normalization import apply_field_aliases, resolve_path_or_default


def test_apply_field_aliases_renames_known_alias():
    result = apply_field_aliases({"target_molecule": "ethanol"})
    assert result == {"component_a": "ethanol"}


def test_apply_field_aliases_keeps_canonical_when_both_present():
    result = apply_field_aliases({"target_molecule": "ethanol", "component_a": "CCO"})
    assert result == {"component_a": "CCO", "target_molecule": "ethanol"}


def test_apply_field_aliases_leaves_unrelated_keys_untouched():
    result = apply_field_aliases({"component_a": "CCO", "model": "gemma4:12b"})
    assert result == {"component_a": "CCO", "model": "gemma4:12b"}


def test_apply_field_aliases_maps_all_known_aliases():
    result = apply_field_aliases({
        "num_candidates": 20,
        "checkpoint": "ckpt.pt",
        "config": "config.yaml",
        "stability_model": "model.json",
    })
    assert result == {
        "n": 20,
        "checkpoint_path": "ckpt.pt",
        "config_path": "config.yaml",
        "stability_constant_model_path": "model.json",
    }


def test_apply_field_aliases_maps_metal_binding_aliases():
    result = apply_field_aliases({
        "target_metal": "Cu2+",
        "target_ligand": "NCCN",
    })
    assert result == {
        "metal_ion": "Cu2+",
        "ligand_smiles": "NCCN",
    }


def test_resolve_path_or_default_returns_existing_path(tmp_path):
    real_file = tmp_path / "config.yaml"
    real_file.write_text("llm: {}")
    default = tmp_path / "default_config.yaml"
    result = resolve_path_or_default(str(real_file), default)
    assert result == str(real_file)


def test_resolve_path_or_default_falls_back_for_missing_path(tmp_path):
    default = tmp_path / "default_config.yaml"
    default.write_text("llm: {}")
    result = resolve_path_or_default("shipped_default", default)
    assert result == str(default)


def test_resolve_path_or_default_falls_back_for_falsy_value(tmp_path):
    default = tmp_path / "default_config.yaml"
    default.write_text("llm: {}")
    assert resolve_path_or_default(None, default) == str(default)
    assert resolve_path_or_default("", default) == str(default)


def test_resolve_path_or_default_falls_back_for_non_string_value(tmp_path):
    default = tmp_path / "default_config.yaml"
    default.write_text("llm: {}")
    assert resolve_path_or_default(True, default) == str(default)
