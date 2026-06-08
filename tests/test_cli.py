from des_multi_agent.cli import build_parser
import des_multi_agent.cli as cli_module


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


def test_task_router_subcommand_prints_json(monkeypatch, capsys):
    class _FakeResponse:
        def to_json(self):
            return "{\"workflow\":\"clarify\",\"needs_clarification\":true,\"clarifying_questions\":[\"Which workflow?\"],\"job\":null}"

    monkeypatch.setattr(cli_module, "route_task", lambda request, provider=None: _FakeResponse())
    cli_module.main(["task-router", "find DES partners for lidocaine"])
    out = capsys.readouterr().out.strip()
    assert out.startswith("{")
    assert "clarifying_questions" in out
    assert "Which workflow?" in out


def test_task_execute_subcommand_prints_report(monkeypatch, capsys):
    monkeypatch.setattr(cli_module, "execute_task_request", lambda request, provider=None: "EXECUTED REPORT")
    cli_module.main(["task-execute", "find DES partners for lidocaine"])
    out = capsys.readouterr().out.strip()
    assert out == "EXECUTED REPORT"


def test_compare_runs_subcommand_prints_report(monkeypatch, capsys):
    monkeypatch.setattr(cli_module, "compare_saved_runs", lambda left, right: type("R", (), {"workflow": "des", "rows": []})())
    monkeypatch.setattr(cli_module, "format_compare_report", lambda result: "compare-runs report")
    cli_module.main(["compare-runs", "runs/run_001", "runs/run_002"])
    out = capsys.readouterr().out.strip()
    assert out == "compare-runs report"


def test_doctor_subcommand_prints_report(capsys):
    try:
        cli_module.main(["doctor"])
    except SystemExit as exc:
        assert exc.code in (0, 1)
    out = capsys.readouterr().out.strip()
    assert out.startswith("doctor:")
    assert "doctor: ok" in out or "errors:" in out or "warnings:" in out


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
