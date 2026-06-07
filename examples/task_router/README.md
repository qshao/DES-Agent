# Task Router Example

This example shows how to turn a plain-language request into a JSON job without running the workflow.

## Input

- Plain-language request: see [`input.txt`](./input.txt)
- The request is written the way a user would ask for help, not as CLI flags.

## Run

```bash
./run.sh
```

The wrapper resolves the repository root, calls the task router, and saves JSON-only output to [`output.txt`](./output.txt).

## Output

The captured output is a JSON job object with:

- `workflow`
- `needs_clarification`
- `clarifying_questions`
- `job`

## How to Adapt

Use this folder as a template for your own work:

- Change `input.txt` to describe your own molecule, metal ion, or workflow goal.
- If the router needs more information, it will return `workflow=clarify` and a list of questions.
- If the job is complete, copy the `job` fields into the matching workflow command.
- For DES runs, the router can populate `component_a`, `n`, `checkpoint_path`, and `config_path`.
- For metal-binding runs, the router can populate `metal_ion`, `ligand_smiles`, and `stability_constant_model_path`.

If you want the task router to use a different local Ollama model, edit [`llm.example.yaml`](../../llm.example.yaml) before running the command.
