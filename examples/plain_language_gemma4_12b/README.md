# Plain-Language Gemma 4-12B DES Example

This example shows how a natural-language request is routed into a DES job and then executed with Ollama Gemma 4-12B. It uses the same task-router style of normalization to convert the plain-language fields into the repo's DES job parameters.

## Input

- Plain-language request: see [`input.txt`](./input.txt)
- Model: Ollama Gemma 4-12B
- Checkpoint: `ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt`
- LLM config: [`llm.gemma4_12b.yaml`](./llm.gemma4_12b.yaml)
- Captured input: [`input.txt`](./input.txt)

## Run

The wrapper first uses the task router to translate the plain-language request into a JSON job, then runs the DES workflow and saves the combined output to [`output.txt`](./output.txt).

```bash
./run.sh
```

## Output

The file [`output.txt`](./output.txt) contains:

- the plain-language request
- the router JSON job
- the DES screening report

## How to Adapt

Use this folder as a template for your own plain-language DES workflow:

- Replace the request in [`input.txt`](./input.txt) with your own natural-language prompt.
- Update the molecule name or SMILES in the request if you want a different component A.
- If you want a different model, edit [`llm.gemma4_12b.yaml`](./llm.gemma4_12b.yaml) or swap in another local Ollama config.
- The example stays close to the real user workflow because the router decides the job fields first and the DES pipeline runs second.
- If you want to save the resulting DES run for later reuse, add `--save-run-memory runs/run_001/run.memory.json` to the underlying DES command; you can then label the saved memory in place with `python -m des_multi_agent.cli label-run --run runs/run_001 --label "O=good"` and later reuse that file with `--reuse-run`. If you keep several labeled runs under `runs/`, you can also point `--reuse-run` at the parent `runs/` directory to reuse the whole labeled history.
