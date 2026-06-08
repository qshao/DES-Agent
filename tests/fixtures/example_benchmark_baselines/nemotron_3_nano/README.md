# Nemotron 3 Nano Example

Uses Ollama with Nemotron 3 Nano and the shared DES demo script.

## Input

- Component A: `CCO`
- Candidate search count: `20`
- Checkpoint: `ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt`
- LLM config: [`llm.nemotron_3_nano.yaml`](./llm.nemotron_3_nano.yaml)
- Captured input: [`input.txt`](./input.txt)

## Run

The wrapper saves stdout to `output.txt` and suppresses stderr so the captured artifact starts with the report table.

```bash
./run.sh
```

## Output

The file [`output.txt`](./output.txt) contains the captured report output from the demo, including:

- ranked DES results
- uncertainty annotations
- LLM brainstorm candidates
- explanation notes
- critique notes
