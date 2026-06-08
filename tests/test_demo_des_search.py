from pathlib import Path

from examples import demo_des_search
from examples.demo_des_search import SearchOutcome, build_parser, resolve_defaults
from des_multi_agent.llm.schemas import CandidateReview


def test_demo_parser_accepts_overrides():
    parser = build_parser()
    args = parser.parse_args(["--component-a", "CCO", "--n", "3", "--discovery-path", "tests/fixtures/discovery"])
    assert args.component_a == "CCO"
    assert args.n == 3
    assert args.discovery_path == "tests/fixtures/discovery"


def test_demo_resolve_defaults_returns_repo_paths():
    checkpoint_path, config_path, llm_config_path = resolve_defaults()
    assert checkpoint_path.name.endswith(".pt")
    assert config_path.name == "config.yaml"
    assert llm_config_path.name == "llm.example.yaml"


def test_tutorial_and_readme_links_exist():
    assert Path("docs/tutorial.md").exists()
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "docs/tutorial.md" in readme
    assert "task-router" in readme
    assert "workflow=clarify" in readme
    assert "viscosity_template" in readme
    assert "ligand_binding_template" in readme
    assert "plain_language_gemma4_12b" in readme
    assert "plain_language_metal_binding_gemma4_12b" in readme


def test_examples_readme_exists_and_links_tutorial():
    examples_readme = Path("examples/README.md")
    assert examples_readme.exists()
    text = examples_readme.read_text(encoding="utf-8")
    assert "docs/tutorial.md" in text
    assert "viscosity_template" in text
    assert "ligand_binding_template" in text
    assert "plain_language_gemma4_12b" in text
    assert "plain_language_metal_binding_gemma4_12b" in text
    assert "task_router" in text


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
    assert "LLM candidate reviews" in out
    assert "Mock mode is using canned outputs" in out
    assert "OCCO" in out


def test_tutorial_shows_explicit_real_checkpoint():
    text = Path("docs/tutorial.md").read_text(encoding="utf-8")
    assert "DES_CHECKPOINT_PATH=ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt" in text
    assert "task-router" in text
    assert "workflow=clarify" in text
    assert "viscosity_template" in text
    assert "ligand_binding_template" in text
    assert "plain_language_gemma4_12b" in text
    assert "plain_language_metal_binding_gemma4_12b" in text


def test_llm_example_mentions_supported_ollama_models():
    text = Path("llm.example.yaml").read_text(encoding="utf-8")
    assert "provider: ollama" in text
    assert "qwen3.6" in text


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
    assert 'if [[ -n "${DES_DISCOVERY_PATH:-}" ]]; then' in contents
    assert '--discovery-path "$DES_DISCOVERY_PATH"' in contents


def test_model_specific_examples_exist():
    for folder, required_texts in [
        ("examples/des_viscosity", ["--viscosity-model-path", "Viscosity predictions"]),
        ("examples/viscosity_template", ["template-style DES viscosity example", "viscosity_model_path"]),
        ("examples/metal_binding", ["--workflow metal-binding", "log K"]),
        ("examples/ligand_binding_template", ["template-style metal-binding example", "stability_constant_model_path"]),
        ("examples/gemma4_12b", ["--llm-config", "CCO"]),
        ("examples/nemotron_3_nano", ["--llm-config", "CCO"]),
        ("examples/qwen3_6", ["--llm-config", "CCO"]),
        ("examples/lidocaine_gemma4_12b", ["--llm-config", "lidocaine"]),
        ("examples/plain_language_gemma4_12b", ["task-router", "plain-language", "Gemma 4-12B"]),
        ("examples/plain_language_metal_binding_gemma4_12b", ["task-router", "plain-language", "metal-binding", "Gemma 4-12B"]),
        ("examples/task_router", ["task-router", "component_a", "checkpoint_path"]),
        ("examples/des_run_memory_feedback", ["save-run-memory", "label-run", "reuse-run", "run.memory.json"]),
    ]:
        base = Path(folder)
        assert (base / "README.md").exists()
        assert (base / "run.sh").exists()
        assert (base / "input.txt").exists()
        assert (base / "output.txt").exists()
        readme_text = (base / "README.md").read_text(encoding="utf-8")
        run_text = (base / "run.sh").read_text(encoding="utf-8")
        output_text = (base / "output.txt").read_text(encoding="utf-8")
        for required in required_texts:
            assert required in readme_text or required in run_text or required in output_text


def test_mock_script_runs_from_other_directory(tmp_path):
    import subprocess
    result = subprocess.run(["bash", str(Path("scripts/demo-mock.sh").resolve())], cwd=tmp_path, check=True, capture_output=True, text=True)
    assert "trust=" in result.stdout
    assert "std=" in result.stdout
    assert "flag=" in result.stdout
    assert "OCCO" in result.stdout



def test_des_run_memory_feedback_example_mentions_feedback_loop():
    readme = Path("examples/des_run_memory_feedback/README.md").read_text(encoding="utf-8")
    assert "save-memory" not in readme  # sanity check for wording drift
    assert "save run memory" not in readme.lower() or "run.memory.json" in readme
    assert "label-run" in readme
    assert "reuse" in readme.lower()
    assert "CCO" in readme


def test_demo_renders_candidate_reviews(monkeypatch, capsys):
    fake_outcome = SearchOutcome(
        results=[],
        annotated_results=[],
        candidate_proposals=[],
        candidate_reviews=[
            CandidateReview(smiles="OCCO", decision="keep", confidence=0.9, rationale="demo review", notes=["stable"])
        ],
        brainstorm_candidates=[],
        explanation_notes=[],
        critique_notes=[],
        llm_warnings=[],
    )
    monkeypatch.setattr(demo_des_search, "run_search_report", lambda *args, **kwargs: fake_outcome)
    demo_des_search.main(["--component-a", "CCO", "--n", "1"])
    out = capsys.readouterr().out
    assert "LLM candidate reviews" in out
    assert "demo review" in out
