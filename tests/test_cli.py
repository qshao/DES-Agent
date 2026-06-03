from des_multi_agent.cli import build_parser


def test_cli_parser_accepts_component_a_and_n():
    parser = build_parser()
    args = parser.parse_args(["--component-a", "CCO", "--n", "5", "--checkpoint-path", "ckpt.pt"])
    assert args.component_a == "CCO"
    assert args.n == 5
