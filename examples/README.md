# Examples

This folder contains the quickest way to try the DES multi-agent system end to end.

## Mock Demo

Run the fully offline mock demo:

```bash
./scripts/demo-mock.sh
```

Direct command if you prefer:

```bash
python -m examples.demo_des_search --mock --component-a "CCO" --n 5
```

## Real Demo

Run the deterministic demo against the shipped checkpoint:

```bash
./scripts/demo-real.sh
```

Direct command if you prefer:

```bash
python -m examples.demo_des_search --component-a "CCO" --n 5 --checkpoint-path ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt
```

To add local discovery, pass the directory explicitly:

```bash
python -m examples.demo_des_search --component-a "CCO" --n 5 --checkpoint-path ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt --discovery-path /path/to/discovery
```

## Optional LLM Mode

If you have an LLM provider configured, pass the sample config:

```bash
python -m examples.demo_des_search --component-a "CCO" --n 5 --llm-config llm.example.yaml
```

## More Detail

See [`docs/tutorial.md`](/home/qshao/DES-Agent/docs/tutorial.md) for a short explanation of the output and common issues.

## Uncertainty Controls

The library CLI [`des_multi_agent.cli`](/home/qshao/DES-Agent/des_multi_agent/cli.py) lets you tune how uncertainty affects filtering and ranking:

```bash
python -m des_multi_agent.cli --component-a "CCO" --n 5 --checkpoint-path ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt --uncertainty-mode filter --min-trust-score 0.70 --soft-penalty-weight 0.20
```

The default mode is `penalize`. Use `report_only` if you want to inspect the uncertainty columns without changing ranking.
