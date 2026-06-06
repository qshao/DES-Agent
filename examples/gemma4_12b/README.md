# Gemma 4-12B Example

Uses Ollama with Gemma 4-12B and the shared DES demo script.

## Input

- Component A: `CCO`
- Candidate search count: `20`
- Checkpoint: `ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt`
- LLM config: [`llm.gemma4_12b.yaml`](./llm.gemma4_12b.yaml)
- Captured input: [`input.txt`](./input.txt)

## Run

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
- warnings if the model cannot be reached locally
