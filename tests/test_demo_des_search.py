from pathlib import Path

from examples import demo_des_search
from examples.demo_des_search import build_parser, resolve_defaults


def test_demo_parser_accepts_overrides():
    parser = build_parser()
    args = parser.parse_args(["--component-a", "CCO", "--n", "3"])
    assert args.component_a == "CCO"
    assert args.n == 3


def test_demo_resolve_defaults_returns_repo_paths():
    checkpoint_path, config_path, llm_config_path = resolve_defaults()
    assert checkpoint_path.name.endswith(".pt")
    assert config_path.name == "config.yaml"
    assert llm_config_path.name == "llm.example.yaml"


def test_tutorial_and_readme_links_exist():
    assert Path("docs/tutorial.md").exists()
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "docs/tutorial.md" in readme


def test_examples_readme_exists_and_links_tutorial():
    examples_readme = Path("examples/README.md")
    assert examples_readme.exists()
    text = examples_readme.read_text(encoding="utf-8")
    assert "docs/tutorial.md" in text


def test_demo_mock_mode_runs_without_real_pipeline(monkeypatch, capsys):
    monkeypatch.setattr(
        demo_des_search,
        "run_search_report",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("real pipeline should not run")),
    )
    demo_des_search.main(["--mock", "--component-a", "CCO", "--n", "2"])
    out = capsys.readouterr().out
    assert "trust=" in out
    assert "std=" in out
    assert "flag=" in out
    assert "Mock mode is using canned outputs" in out
    assert "OCCO" in out


def test_tutorial_shows_explicit_real_checkpoint():
    text = Path("docs/tutorial.md").read_text(encoding="utf-8")
    assert "DES_CHECKPOINT_PATH=ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt" in text


def test_mock_script_exists_and_is_simple():
    script = Path("scripts/demo-mock.sh")
    assert script.exists()
    contents = script.read_text(encoding="utf-8")
    assert 'python -m examples.demo_des_search --mock --component-a "CCO" --n 5' in contents


def test_real_script_exists_and_is_simple():
    script = Path("scripts/demo-real.sh")
    assert script.exists()
    contents = script.read_text(encoding="utf-8")
    assert 'CHECKPOINT_PATH="${DES_CHECKPOINT_PATH:-ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt}"' in contents
    assert 'python -m examples.demo_des_search --component-a "CCO" --n 5 --checkpoint-path "$CHECKPOINT_PATH"' in contents


def test_mock_script_runs_from_other_directory(tmp_path):
    import subprocess
    result = subprocess.run(["bash", str(Path("scripts/demo-mock.sh").resolve())], cwd=tmp_path, check=True, capture_output=True, text=True)
    assert "trust=" in result.stdout
    assert "std=" in result.stdout
    assert "flag=" in result.stdout
    assert "OCCO" in result.stdout
