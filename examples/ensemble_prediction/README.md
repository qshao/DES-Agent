# Ensemble Prediction Example

This example demonstrates `--ensemble`, which runs predictions across all fold checkpoints found in `ml_des_mp/runs/` and aggregates them into a single result with per-candidate uncertainty estimates.

Compared to a single-checkpoint run, ensemble mode:

- Uses every `*_best.pt` file found in `ml_des_mp/runs/` automatically — no `--checkpoint-path` needed
- Adds `ensemble_folds=N` and `ens_std=X K` to each candidate's rationale column
- May re-rank candidates when folds disagree on the best eutectic composition
- Produces slightly different Tm estimates than any single fold, reflecting the average across the ensemble

## Input

- Component A: `CCO` (ethanol)
- Candidate search count: `5`
- Ensemble mode: all fold checkpoints in `ml_des_mp/runs/` (10 folds in the bundled repo)
- Captured input: [`input.txt`](./input.txt)

## Run

```bash
./run.sh
```

## Output

The file [`output.txt`](./output.txt) shows the ensemble-ranked table. Notice the `ensemble_folds=10 ens_std=X K` annotation in the rationale column — higher `ens_std` values indicate greater disagreement between folds and should be treated with more caution.

## How to Adapt

- Swap in your own fold checkpoints by placing `*_best.pt` files in `ml_des_mp/runs/`.
- Combine with `--uncertainty-mode penalize` (see [`examples/uncertainty_controls/`](../uncertainty_controls)) to automatically down-rank candidates where folds disagree.
- If you only have a single checkpoint, use `--checkpoint-path` instead and omit `--ensemble`.
