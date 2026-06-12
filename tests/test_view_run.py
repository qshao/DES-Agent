from pathlib import Path

import pytest

from des_multi_agent.view_run import build_run_view, format_run_view


def _write_run_dir(run_dir: Path) -> None:
    run_dir.mkdir()
    (run_dir / "run.manifest.json").write_text(
        '{"workflow":"des","component_a":"CCO","n":2,"report_filename":"report.txt","json_filename":"run.json","csv_filename":"run.csv"}',
        encoding="utf-8",
    )
    (run_dir / "run.json").write_text(
        '{"workflow":"des","component_a":"CCO","n":2,"results":[{"rank":1,"smiles_b":"O","is_des":true,"min_tm_k":208.69},{"rank":2,"smiles_b":"CC(=O)O","is_des":true,"min_tm_k":236.03}]}',
        encoding="utf-8",
    )


def test_build_run_view_reads_standardized_run_directory(tmp_path: Path):
    run_dir = tmp_path / "run_001"
    _write_run_dir(run_dir)
    (run_dir / "run.memory.json").write_text('{"labels":[{"smiles_b":"O","label":"good"}]}', encoding="utf-8")

    view = build_run_view(run_dir, top_n=1)

    assert view["workflow"] == "des"
    assert view["component_a"] == "CCO"
    assert view["candidate_count"] == 2
    assert view["label_count"] == 1
    assert view["top_candidates"][0]["smiles_b"] == "O"


def test_format_run_view_prints_artifacts(tmp_path: Path):
    run_dir = tmp_path / "run_001"
    _write_run_dir(run_dir)

    text = format_run_view(build_run_view(run_dir))

    assert "workflow: des" in text
    assert "rank=1 smiles_b=O" in text
    assert "report.txt" in text


def test_build_run_view_requires_manifest(tmp_path: Path):
    run_dir = tmp_path / "run_001"
    run_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="run manifest"):
        build_run_view(run_dir)


def test_build_run_view_rejects_malformed_run_json(tmp_path: Path):
    run_dir = tmp_path / "run_001"
    run_dir.mkdir()
    (run_dir / "run.manifest.json").write_text('{"json_filename":"run.json"}', encoding="utf-8")
    (run_dir / "run.json").write_text('{', encoding="utf-8")

    with pytest.raises(ValueError, match="Malformed run json"):
        build_run_view(run_dir)
