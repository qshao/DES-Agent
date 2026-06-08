from pathlib import Path

from tests.fixtures.example_benchmark_cases import (
    iter_benchmark_cases,
    normalize_benchmark_output,
)


def test_benchmark_cases_cover_current_examples():
    names = [case.name for case in iter_benchmark_cases()]
    assert "task_router" in names
    assert "plain_language_gemma4_12b" in names
    assert "plain_language_metal_binding_gemma4_12b" in names
    assert "des_viscosity" in names
    assert "metal_binding" in names


def test_normalize_benchmark_output_strips_warning_noise():
    raw = "\n".join([
        "  ",
        "DeprecationWarning: torch_geometric.distributed has been deprecated",
        "actual report line",
        "",
        "",
        "DeprecationWarning: `torch.jit.script` is deprecated.",
    ])
    normalized = normalize_benchmark_output(raw)
    assert normalized == "actual report line"


def test_benchmark_case_paths_exist():
    for case in iter_benchmark_cases():
        assert case.example_input.exists()
        assert case.example_output.exists()
        assert case.example_readme.exists()
        assert case.baseline_input.exists()
        assert case.baseline_output.exists()
        assert case.baseline_readme.exists()


def test_each_example_output_matches_frozen_baseline():
    for case in iter_benchmark_cases():
        actual = normalize_benchmark_output(case.example_output.read_text(encoding="utf-8"))
        expected = normalize_benchmark_output(case.baseline_output.read_text(encoding="utf-8"))
        assert actual == expected, case.name


def test_example_benchmark_score_is_perfect():
    cases = iter_benchmark_cases()
    matched = 0
    for case in cases:
        actual = normalize_benchmark_output(case.example_output.read_text(encoding="utf-8"))
        expected = normalize_benchmark_output(case.baseline_output.read_text(encoding="utf-8"))
        matched += int(actual == expected)
    score = matched / len(cases)
    assert score == 1.0, f"benchmark score was {score:.3f}"


def test_docs_mention_example_benchmark_suite():
    readme = Path("README.md").read_text(encoding="utf-8")
    tutorial = Path("docs/tutorial.md").read_text(encoding="utf-8")
    examples_readme = Path("examples/README.md").read_text(encoding="utf-8")
    assert "example benchmark" in readme.lower()
    assert "example benchmark" in tutorial.lower()
    assert "example benchmark" in examples_readme.lower()
