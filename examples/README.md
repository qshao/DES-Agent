# Examples

This folder contains the quickest way to try the DES multi-agent system end to end.

## Mock demo

Run the fully offline mock demo:

```bash
./scripts/demo-mock.sh
```

Direct command if you prefer:

```bash
python -m examples.demo_des_search --mock --component-a "CCO" --n 5
```

## Real demo

Run the default deterministic demo against a local checkpoint:

```bash
./scripts/demo-real.sh
```

Direct command if you prefer:

```bash
DES_CHECKPOINT_PATH=ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt python -m examples.demo_des_search --component-a "CCO" --n 5 --checkpoint-path "$DES_CHECKPOINT_PATH"
```

## Optional LLM mode

If you have an LLM provider configured, pass the sample config:

```bash
python -m examples.demo_des_search --component-a "CCO" --n 5 --llm-config llm.example.yaml
```

## More detail

See [`docs/tutorial.md`](/home/qshao/DES-Agent/docs/tutorial.md) for a short explanation of the output and common issues.

## Uncertainty Controls

The library CLI [`des_multi_agent.cli`](/home/qshao/DES-Agent/des_multi_agent/cli.py) lets you tune how uncertainty affects filtering and ranking. Example:

```bash
python -m des_multi_agent.cli --component-a "CCO" --n 5 --checkpoint-path ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt --uncertainty-mode filter --min-trust-score 0.70 --soft-penalty-weight 0.20
```

The default mode is `penalize`. Use `report_only` if you want to inspect the uncertainty columns without changing ranking.
