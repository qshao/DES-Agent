# Candidates File Example

This example demonstrates the `--candidates-file` flag, which bypasses LLM/heuristic candidate generation and screens a curated SMILES list directly. This is useful for targeted screening, literature compounds, or when you already know which candidates to evaluate — no LLM or internet connection is required.

## Input

- Component A: `CCO` (ethanol)
- Candidates file: [`candidates.smiles`](./candidates.smiles) — 6 curated SMILES, one per line
- Candidate count: `--n 6`
- Checkpoint: `ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt`
- Captured input: [`input.txt`](./input.txt)

## Run

```bash
./run.sh
```

## Output

The file [`output.txt`](./output.txt) contains the captured report output from the DES workflow, including the screening table ranked by `min_tm_k`.

## How to Adapt

- Edit [`candidates.smiles`](./candidates.smiles) to add or replace compounds with your own SMILES strings (one per line, no header).
- Set `--n` to match or exceed the number of lines in your candidates file so all entries are evaluated.
- The `--candidates-file` flag accepts any plain-text file with one valid SMILES string per line.
