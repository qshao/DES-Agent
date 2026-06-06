# Lidocaine + Gemma 4-12B Example

This folder records a real DES screening run for lidocaine using the shared multi-agent demo, the shipped `ml_des_mp` checkpoint, and Ollama Gemma 4-12B.

## Input

- Component A: `lidocaine` free base
- SMILES: `CCN(CC)CC(=O)Nc1c(C)cccc1C`
- Candidate search count: `5`
- Checkpoint: `ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt`
- LLM config: [`llm.gemma4_12b.yaml`](./llm.gemma4_12b.yaml)
- Captured input: [`input.txt`](./input.txt)

## Run

```bash
./run.sh
```

## Output

The file [`output.txt`](./output.txt) contains the captured report output from the real run, including:

- ranked DES results
- uncertainty annotations
- Gemma brainstorm candidates
- explanation notes
- critique notes
- warnings if the local Ollama model is unavailable
