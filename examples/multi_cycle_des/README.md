# Multi-Cycle Iterative Screening Example

This example demonstrates the `--n-cycles` flag, which runs iterative DES screening over multiple cycles. Top hits from each cycle seed the next brainstorm, and screening stops early when the top-K candidates converge across consecutive cycles.

## Input

- Component A: `CCO`
- Candidate search count: `5`
- Checkpoint: `ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt`
- Number of cycles: `3`
- Captured input: [`input.txt`](./input.txt)

## Run

```bash
./run.sh
```

## Output

The file [`output.txt`](./output.txt) contains the final DES screening table from the last cycle. Cycle-level progress lines (e.g. `[cycle 1/3] screened=5 des=3 top-K changes: +3 new, 0 dropped`) are printed to stderr during the run. If the top-K candidate set stabilises before all cycles complete, the run stops early.

The file [`trajectory.md`](./trajectory.md) is the durable per-cycle narrative written by `--output-dir`. It records how many candidates were screened and hit each cycle, which entered or left the shortlist, which chemical families were reinforced, and whether the search converged. A console summary is also printed to stderr during the run.
