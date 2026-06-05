from __future__ import annotations

from des_multi_agent import cli
from des_multi_agent.uncertainty import UncertaintyPolicy


def test_cli_parser_accepts_uncertainty_arguments():
    parser = cli.build_parser()
    args = parser.parse_args([
        "--component-a", "CCO",
        "--n", "3",
        "--checkpoint-path", "ckpt.pt",
        "--uncertainty-mode", "filter",
        "--min-trust-score", "0.70",
        "--soft-penalty-weight", "0.20",
        "--std-high-threshold-k", "12.0",
        "--std-medium-threshold-k", "4.0",
    ])

    assert args.uncertainty_mode == "filter"
    assert args.min_trust_score == 0.70
    assert args.soft_penalty_weight == 0.20
    assert args.std_high_threshold_k == 12.0
    assert args.std_medium_threshold_k == 4.0


def test_cli_main_passes_uncertainty_policy(monkeypatch, tmp_path, capsys):
    checkpoint = tmp_path / "ckpt.pt"
    checkpoint.write_text("ckpt", encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text(
        """device: cpu
embedding:
  method: morgan
  morgan:
    radius: 2
    n_bits: 16
    use_chirality: false
""",
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    def fake_run_search_report(**kwargs):
        captured.update(kwargs)
        return type(
            "Outcome",
            (),
            {
                "results": [],
                "annotated_results": [],
                "brainstorm_candidates": [],
                "explanation_notes": [],
                "critique_notes": [],
                "llm_warnings": [],
            },
        )

    monkeypatch.setattr(cli, "run_search_report", fake_run_search_report)
    monkeypatch.setattr(cli, "format_report", lambda *args, **kwargs: "ok")

    cli.main([
        "--component-a", "CCO",
        "--n", "3",
        "--checkpoint-path", str(checkpoint),
        "--config-path", str(config),
        "--uncertainty-mode", "report_only",
        "--min-trust-score", "0.70",
        "--soft-penalty-weight", "0.20",
        "--std-high-threshold-k", "12.0",
        "--std-medium-threshold-k", "4.0",
    ])

    out = capsys.readouterr().out
    assert out.strip() == "ok"
    assert isinstance(captured["uncertainty_policy"], UncertaintyPolicy)
    assert captured["uncertainty_policy"].mode == "report_only"
    assert captured["uncertainty_policy"].min_trust_score == 0.70
    assert captured["uncertainty_policy"].soft_penalty_weight == 0.20
    assert captured["uncertainty_policy"].std_high_threshold_k == 12.0
    assert captured["uncertainty_policy"].std_medium_threshold_k == 4.0
