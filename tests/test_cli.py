from pathlib import Path
import json
from types import SimpleNamespace

import des_multi_agent.cli as cli_module
import des_multi_agent.exporting as exporting_module
from des_multi_agent.cli import build_parser


def test_cli_parser_accepts_component_a_and_n():
    parser = build_parser()
    args = parser.parse_args(["--component-a", "CCO", "--n", "5", "--checkpoint-path", "ckpt.pt"])
    assert args.component_a == "CCO"
    assert args.n == 5


def test_cli_parser_accepts_metal_binding_args():
    parser = build_parser()
    args = parser.parse_args(["--workflow", "metal-binding", "--metal-ion", "Cu2+", "--ligand-smiles", "NCCN"])
    assert args.workflow == "metal-binding"
    assert args.metal_ion == "Cu2+"
    assert args.ligand_smiles == "NCCN"


def test_metal_binding_cli_does_not_invoke_des_exports(monkeypatch, capsys):
    class _FakeOutcome:
        metal_ion = "Cu2+"
        ligand_smiles = "NCCN"
        prediction = type("P", (), {"value": 1.23, "units": "log K", "model_name": "mock", "source": "mock", "warnings": ()})()
        warnings = ()

    monkeypatch.setattr(cli_module, "run_metal_binding_workflow", lambda *args, **kwargs: _FakeOutcome())
    monkeypatch.setattr(cli_module, "format_metal_binding_report", lambda outcome: "METAL REPORT")
    monkeypatch.setattr(
        exporting_module,
        "export_des_run_bundle",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("DES export should not be called")),
    )
    cli_module.main(["--workflow", "metal-binding", "--metal-ion", "Cu2+", "--ligand-smiles", "NCCN"])
    out = capsys.readouterr().out.strip().splitlines()
    assert out[0] == "METAL REPORT"
    assert any(line.startswith("summary:") for line in out)


def test_cli_parser_accepts_task_router_subcommand():
    parser = build_parser()
    args = parser.parse_args(["task-router", "find DES partners for lidocaine"])
    assert args.command == "task-router"
    assert args.request == "find DES partners for lidocaine"


def test_cli_parser_accepts_task_execute_subcommand():
    parser = build_parser()
    args = parser.parse_args(["task-execute", "find DES partners for lidocaine"])
    assert args.command == "task-execute"
    assert args.request == "find DES partners for lidocaine"


def test_cli_parser_accepts_compare_runs_subcommand():
    parser = build_parser()
    args = parser.parse_args(["compare-runs", "runs/run_001", "runs/run_002"])
    assert args.command == "compare-runs"
    assert args.left == "runs/run_001"
    assert args.right == "runs/run_002"


def test_cli_parser_accepts_label_run_subcommand():
    parser = build_parser()
    args = parser.parse_args([
        "label-run",
        "--run",
        "runs/run_001",
        "--label",
        "O=good",
        "--label",
        "O=bad",
    ])
    assert args.command == "label-run"
    assert args.run == "runs/run_001"
    assert args.label == ["O=good", "O=bad"]


def test_task_router_subcommand_prints_json_and_summary(monkeypatch, capsys):
    class _FakeResponse:
        needs_clarification = True
        clarifying_questions = ["Which workflow?"]

        def to_json(self):
            return '{"workflow":"clarify","needs_clarification":true,"clarifying_questions":["Which workflow?"],"job":null}'

    monkeypatch.setattr(cli_module, "route_task", lambda request, provider=None: _FakeResponse())
    cli_module.main(["task-router", "find DES partners for lidocaine"])
    out = capsys.readouterr()
    assert out.out.strip().startswith("{")
    assert "clarifying_questions" in out.out
    assert "summary:" in out.err


def test_task_execute_subcommand_prints_report_and_summary(monkeypatch, capsys):
    monkeypatch.setattr(
        cli_module,
        "execute_task_request_detailed",
        lambda request, provider=None, llm_cfg=None: SimpleNamespace(needs_clarification=False, output="EXECUTED REPORT", summary_status="executed"),
    )
    cli_module.main(["task-execute", "find DES partners for lidocaine"])
    out = capsys.readouterr().out.strip().splitlines()
    assert out[0] == "EXECUTED REPORT"
    assert any(line.startswith("summary:") for line in out)


def test_task_execute_subcommand_prints_clarification_json_and_summary_to_stderr(monkeypatch, capsys):
    monkeypatch.setattr(
        cli_module,
        "execute_task_request_detailed",
        lambda request, provider=None, llm_cfg=None: SimpleNamespace(
            needs_clarification=True,
            output='{"workflow":"clarify","needs_clarification":true,"clarifying_questions":["Which workflow?"],"job":null}',
            summary_status="clarified",
        ),
    )
    cli_module.main(["task-execute", "find DES partners for lidocaine"])
    out = capsys.readouterr()
    assert out.out.strip().startswith("{")
    assert "summary:" in out.err


def test_compare_runs_subcommand_prints_report_and_summary(monkeypatch, capsys):
    monkeypatch.setattr(cli_module, "compare_saved_runs", lambda left, right: type("R", (), {"workflow": "des", "rows": []})())
    monkeypatch.setattr(cli_module, "format_compare_report", lambda result: "compare-runs report")
    cli_module.main(["compare-runs", "runs/run_001", "runs/run_002"])
    out = capsys.readouterr().out.strip().splitlines()
    assert out[0] == "compare-runs report"
    assert any(line.startswith("summary:") for line in out)


def test_compare_runs_subcommand_prints_json_summary(monkeypatch, capsys):
    class _FakeResult:
        workflow = "des"
        left_path = Path("runs/run_001")
        right_path = Path("runs/run_002")
        left_component_a = "CCO"
        right_component_a = "CCO"
        left_n = 5
        right_n = 6
        rows = []

    monkeypatch.setattr(cli_module, "compare_saved_runs", lambda left, right: _FakeResult())
    monkeypatch.setattr(cli_module, "format_compare_report", lambda result: "compare-runs report")
    monkeypatch.setattr(
        cli_module,
        "format_compare_json_text",
        lambda result: '{"workflow": "des", "counts": {"new": 0, "removed": 0, "moved": 0, "unchanged": 0}, "changed_candidates": []}',
    )
    cli_module.main(["compare-runs", "runs/run_001", "runs/run_002", "--json"])
    out = capsys.readouterr()
    assert json.loads(out.out.splitlines()[0])["workflow"] == "des"
    assert "compare-runs report" in out.err
    assert "summary:" in out.err


def test_doctor_subcommand_prints_report_and_summary(capsys):
    try:
        cli_module.main(["doctor"])
    except SystemExit as exc:
        assert exc.code in (0, 1)
    out = capsys.readouterr().out.strip().splitlines()
    assert out[0].startswith("doctor:")
    assert any(line.startswith("summary:") for line in out)


def test_label_run_subcommand_prints_message_and_summary(monkeypatch, capsys):
    monkeypatch.setattr(cli_module, "run_label_command", lambda run, labels: "UPDATED MEMORY")
    cli_module.main(["label-run", "--run", "runs/run_001", "--label", "O=good"])
    out = capsys.readouterr().out.strip().splitlines()
    assert out[0] == "UPDATED MEMORY"
    assert any(line.startswith("summary:") for line in out)


def test_cli_parser_supports_run_memory_flags():
    parser = build_parser()
    args = parser.parse_args([
        "--workflow",
        "des",
        "--component-a",
        "CCO",
        "--checkpoint-path",
        "ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt",
        "--save-run-memory",
        "runs/run_001/run.memory.json",
        "--reuse-run",
        "runs/run_000/run.memory.json",
    ])
    assert args.save_run_memory == "runs/run_001/run.memory.json"
    assert args.reuse_run == "runs/run_000/run.memory.json"


def test_cli_parser_accepts_doctor_subcommand():
    parser = build_parser()
    args = parser.parse_args(["doctor"])
    assert args.command == "doctor"


def test_cli_parser_accepts_output_dir():
    parser = build_parser()
    args = parser.parse_args([
        "--workflow",
        "des",
        "--component-a",
        "CCO",
        "--checkpoint-path",
        "ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt",
        "--output-dir",
        "runs/run_001",
    ])
    assert args.output_dir == "runs/run_001"


def test_des_cli_forwards_output_dir_to_run_search_report(monkeypatch, tmp_path, capsys):
    checkpoint_path = tmp_path / "ckpt.pt"
    checkpoint_path.write_text("ckpt", encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text("device: cpu\n", encoding="utf-8")
    captured = {}

    def fake_run_search_report(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            results=[],
            annotated_results=[],
            candidate_proposals=[],
            candidate_reviews=[],
            explanation_notes=[],
            critique_notes=[],
            brainstorm_candidates=[],
            llm_warnings=[],
            memory_notes=[],
            viscosity_predictions=[],
        )

    monkeypatch.setattr(cli_module, "run_search_report", fake_run_search_report)
    monkeypatch.setattr(cli_module, "format_report", lambda *args, **kwargs: "DES REPORT")

    cli_module.main([
        "--workflow",
        "des",
        "--component-a",
        "CCO",
        "--checkpoint-path",
        str(checkpoint_path),
        "--config-path",
        str(config_path),
        "--output-dir",
        str(tmp_path / "runs" / "run_001"),
    ])

    out = capsys.readouterr().out.strip().splitlines()
    assert out[0] == "DES REPORT"
    assert any(line.startswith("summary:") for line in out)
    assert captured["output_dir"] == str(tmp_path / "runs" / "run_001")


def test_compare_runs_subcommand_accepts_json_flag():
    parser = build_parser()
    args = parser.parse_args(["compare-runs", "runs/run_001", "runs/run_002", "--json"])
    assert args.json is True
