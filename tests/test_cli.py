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
