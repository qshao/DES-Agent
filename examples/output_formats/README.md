# Output Formats Example

This example demonstrates the four `--format` modes for DES search output.

| Format | Best for |
|--------|---------|
| `table` | Human reading, terminal display (default) |
| `json` | Scripting, downstream processing, REST APIs |
| `csv` | Spreadsheets, pandas, data analysis tools |
| `prose` | LLM-downstream prompts, narrative reports |

The script runs the same ethanol (`CCO`) query four times — once per format — so you can see the exact shape of each output in [`output.txt`](./output.txt).

## Input

- Component A: `CCO` (ethanol)
- Candidate search count: `5`
- Checkpoint: `ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt`
- Formats compared: `table`, `json`, `csv`, `prose`
- Captured input: [`input.txt`](./input.txt)

## Run

```bash
./run.sh
```

## Output

The file [`output.txt`](./output.txt) shows all four formats. A few things to note:

- **`json`** includes full numeric precision and all fields; useful for piping into `jq` or Python.
- **`csv`** is spreadsheet-ready; pipe to a file with `> results.csv` for direct import.
- **`prose`** is a short natural-language summary only — no table. The `summary:` block is always printed to stderr, so `stdout` stays clean for machine consumers.

## How to Adapt

```bash
# Save JSON for later processing
python -m des_multi_agent.cli --workflow des --component-a "CCO" \
  --format json > results.json

# Pipe CSV directly into Python
python -m des_multi_agent.cli --workflow des --component-a "CCO" \
  --format csv | python -c "import sys,csv; rows=list(csv.DictReader(sys.stdin)); print(rows[0])"
```
