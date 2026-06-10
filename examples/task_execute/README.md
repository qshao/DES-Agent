# Task-Execute Example

This example shows how `task-execute` translates a plain-language request and immediately runs the matching workflow in one step. It combines the `task-router` (translation step) with actual workflow execution — so a single command takes natural language in and produces a DES screening table out.

> **Requires Ollama** — start Ollama with `ollama serve` and pull a model (e.g. `ollama pull gemma4:12b`) before running. The task router calls the local LLM to parse the plain-language request; without a running Ollama service the command will fail.

## Input

- Plain-language request: see [`input.txt`](./input.txt)
- The request is written the way a user would ask for help, not as CLI flags.

## Run

```bash
./run.sh
```

The wrapper resolves the repository root, calls `task-execute` with the hardcoded request string, and saves output to [`output.txt`](./output.txt).

## Output

When Ollama is running, the output is a standard DES screening table for CCO with 20 candidates, including columns:

- `smiles_b` — SMILES of the candidate partner
- `is_des` — whether the pair forms a DES (Tm ≤ 260 K with ≥10% relative drop)
- `min_tm_k` — predicted minimum melting point (K)
- `source` — prediction source (e.g. `heuristic; id=rule-based-family-library`)
- `trust` — confidence score
- `mean_tm_k`, `spread_k`, `std_k` — ensemble statistics
- `uncertainty_flag` — `low` / `medium` / `high`
- `rationale` — short explanation of the prediction

The file [`output.txt`](./output.txt) is a placeholder — run `./run.sh` with Ollama active to capture live output.

## How to Adapt

Use this folder as a template for your own work:

- Change the request string in `run.sh` to describe a different molecule, candidate count, or workflow goal.
- The router understands both DES and metal-binding workflows, so requests like `"find metal binding partners for Cu2+ with EDTA"` will route to the metal-binding workflow automatically.
- To see only the routing step (JSON job object, no workflow execution), use [`examples/task_router/`](../task_router) instead.
- If the router needs more information, the underlying `task-router` command returns `workflow=clarify` with a list of questions; `task-execute` will surface these before running.

If you want `task-execute` to use a different local Ollama model, edit [`llm.example.yaml`](../../llm.example.yaml) before running the command.
