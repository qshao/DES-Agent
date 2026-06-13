# Plain-Language Gemma 4-12B Metal-Binding Example

This example shows how a natural-language request is routed into a metal-binding job and then executed with Ollama Gemma 4-12B. It uses the same task-router style normalization to convert the plain-language fields into the repo's metal-binding job parameters.

This example is the plain-language counterpart to the standalone metal-binding workflow, so it is a good place to demonstrate how the router can turn chemistry intent into structured inputs before the prediction step.

## Input

- Plain-language request: see [`input.txt`](./input.txt)
- Model: Ollama Gemma 4-12B
- Stability model: `artifacts/stability_constants/model.json`
- LLM config: [`llm.gemma4_12b.yaml`](./llm.gemma4_12b.yaml)
- Captured input: [`input.txt`](./input.txt)

## Run

The wrapper first uses the task router to translate the plain-language request into a JSON job, then runs the metal-binding workflow and saves the combined output to [`output.txt`](./output.txt).

```bash
./run.sh
```

## Output

The file [`output.txt`](./output.txt) contains:

- the plain-language request
- the router JSON job
- the metal-binding prediction report

## How to Adapt

Use this folder as a template for your own plain-language metal-binding workflow:

- Replace the request in [`input.txt`](./input.txt) with your own natural-language prompt.
- Update the metal ion and ligand if you want a different binding pair.
- If you want a different model, edit [`llm.gemma4_12b.yaml`](./llm.gemma4_12b.yaml) or swap in another local Ollama config.
- The example stays close to the real user workflow because the router decides the job fields first and the metal-binding workflow runs second.
- The DES run-memory reuse feature is DES-only; it does not apply to this metal-binding workflow.
- Proposal-diversity controls are also DES-only, so this example remains a routing and prediction demo rather than a brainstorm demo.
