# Leaderboard and History Example

This example shows how to accumulate multiple DES runs under one history directory, then use `leaderboard` to rank all compounds across runs and `history` to review a per-run summary.

## Input

- Component A: `CCO` (ethanol)
- Candidate search count: `5`
- Checkpoint: `ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt`
- History directory: `/tmp/des_history/` (two runs written here)
- Captured input: [`input.txt`](./input.txt)

## Run

```bash
./run.sh
```

## Output

The file [`output.txt`](./output.txt) contains the combined output from all four steps:

- **RUN 1 / RUN 2**: standard DES screening tables written to `run_01/` and `run_02/` under the history directory.
- **LEADERBOARD**: compounds ranked by best `min_tm_k` across all runs; the `runs` column shows in how many cycles a compound appeared.
- **HISTORY**: one row per run with `run_name`, timestamp, `component_a`, `n_screened`, `n_des`, top candidate, and its `min_tm_k`.

## How to Adapt

- Use different `component_a` values across runs to compare different targets in a single leaderboard.
- Add more runs by repeating the `--output-dir` pattern with a new subdirectory name (`run_03/`, etc.).
- Build a persistent history over time by pointing every DES run at the same `--output-dir` parent; the leaderboard and history commands read all subdirectories with valid `run.json` or `run.manifest.json` files.
