# Dry Run Example

This example demonstrates `--dry-run`, which validates paths, config, and checkpoint compatibility then exits immediately — no ML predictions are run, no output files are written.

Use `--dry-run` to:

- Confirm a checkpoint file is loadable and matches the current config before committing to a long run
- Validate a new environment or container without spending compute
- Check SMILES, config path, and checkpoint path in a CI/CD pipeline
- Debug setup issues without running the full pipeline

## Input

- Component A: `CCO` (ethanol)
- Candidate search count: `5`
- Checkpoint: `ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt`
- Dry run: enabled
- Captured input: [`input.txt`](./input.txt)

## Run

```bash
./run.sh
```

The script exits with code `0` on success and `1` if any path or config check fails.

## Output

The file [`output.txt`](./output.txt) contains the single validation line:

```
[dry-run] Paths resolved, config parsed, checkpoint compatible — OK.
```

If a path is wrong or the checkpoint is incompatible, an error message is printed instead and the exit code is non-zero.

## How to Adapt

Add `--dry-run` to any DES command before your first real run:

```bash
python -m des_multi_agent.cli \
  --workflow des \
  --component-a "your-smiles" \
  --checkpoint-path /path/to/model.pt \
  --dry-run && echo "Setup OK — running full search now" && \
python -m des_multi_agent.cli \
  --workflow des \
  --component-a "your-smiles" \
  --checkpoint-path /path/to/model.pt
```
