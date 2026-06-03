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
