# Uncertainty Controls Example

This example demonstrates the three `--uncertainty-mode` policies that control how low-trust predictions are handled in the ranked output.

| Mode | Effect |
|------|--------|
| `report_only` | Adds `trust` and `confidence` columns; ranking unchanged |
| `penalize` | Down-ranks candidates whose trust score falls below `--min-trust-score` by `--soft-penalty-weight` |
| `filter` | Removes candidates below `--min-trust-score` entirely; they do not appear in the table |

The script runs all three modes on the same ethanol (`CCO`) query so the ranking difference is visible in [`output.txt`](./output.txt).

## Input

- Component A: `CCO` (ethanol)
- Candidate search count: `5`
- Checkpoint: `ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt`
- Modes compared: `report_only`, `penalize` (min-trust 0.85, penalty 0.3), `filter` (min-trust 0.9)
- Captured input: [`input.txt`](./input.txt)

## Run

```bash
./run.sh
```

## Output

The file [`output.txt`](./output.txt) contains three sections. In the bundled heuristic run all candidates have `trust=0.80`, so:

- `report_only` shows the trust column but leaves ranking intact.
- `penalize` with `min-trust-score 0.85` keeps all candidates but applies a soft composite-score penalty to each (their scores decrease proportionally to `--soft-penalty-weight`).
- `filter` with `min-trust-score 0.9` removes all candidates because none reach the 0.90 threshold — the table is empty.

In a real ML-backed run trust scores vary by candidate, so only the low-trust ones are affected.

## How to Adapt

- Start with `report_only` to audit the trust landscape before deciding on a policy.
- Use `penalize` when you want uncertain candidates to stay visible but sorted lower.
- Use `filter` for production pipelines where low-confidence results should never appear.
- Combine with `--ensemble` to get meaningful per-candidate uncertainty from fold disagreement.
