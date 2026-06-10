# Betaine DES Screening with Gemma 4-12B

This example searches for deep eutectic solvent (DES) partners for **betaine** (trimethylglycine, `C[N+](C)(C)CC(=O)[O-]`) using Ollama Gemma 4-12B as the LLM advisor. It combines multi-cycle iterative screening with viscosity-aware ranking and LLM-driven chemical validity checking.

## Chemistry

Betaine is a natural zwitterionic compound (quaternary N⁺, carboxylate O⁻) with a high pure-component melting point (~573 K). To form a DES, a partner molecule must:

- Be an **organic** compound (not a simple inorganic salt or ionic species)
- Carry **hydrogen-bond donor** groups (–OH, –NH, –COOH) or **hydrogen-bond acceptor** groups that can interact with betaine's carboxylate oxygen and quaternary nitrogen
- Produce a mixture melting point depression to below **340 K**

The LLM enforces these constraints in two ways:
1. **Two-stage brainstorm (H6)**: Gemma first selects 4–6 chemical families (polyols, carboxylic acids, amides, amino alcohols, …) that satisfy the organic + H-bonding requirement for betaine, then distributes 20 candidate SMILES across those families.
2. **Contradiction detection (H3)**: After ML predictions, Gemma reviews each DES-positive result and reports `agree`, `conflict`, or `uncertain` — flagging cases where the predicted DES-former lacks H-bonding groups or is chemically implausible.

## Requirements

Ollama must be running with Gemma 4-12B available:

```bash
ollama serve          # start the Ollama daemon
ollama pull gemma4:12b
```

## Input

- Component A: `C[N+](C)(C)CC(=O)[O-]` (betaine, zwitterionic form)
- Candidate search count per cycle: `20`
- Checkpoint: `ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt`
- Tm ceiling: `340 K` (`--abs-tm-threshold 340`)
- Minimum relative Tm drop: `5%` (`--rel-drop-min 0.05`)
- Viscosity model: `artifacts/designsolvents/viscosity/model.json`
- Viscosity weight: `0.7` (viscosity dominates composite ranking)
- Max cycles: `5`
- LLM: Ollama Gemma 4-12B (`llm.gemma4_12b.yaml`)
- Captured input: [`input.txt`](./input.txt)

## Run

```bash
./run.sh
```

## Output

The file [`output.txt`](./output.txt) contains the captured report from a real run with Gemma 4-12B, including:

- **DES screening table**: LLM-brainstormed candidates ranked by composite viscosity + Tm score; candidates with `source=llm; id=brainstorm` were proposed by Gemma across chemical families.
- **LLM candidate reviews**: Gemma's per-candidate assessment; compounds that are inorganic, ionic, or lack H-bonding groups are rejected with reasoning.
- **LLM brainstorm**: the family-grouped candidate list showing Gemma's two-stage selection.
- **LLM contradiction analysis**: `agree` / `conflict` / `uncertain` verdict per DES-positive candidate, with chemical reasoning about H-bond compatibility with betaine's carboxylate.
- **Viscosity predictions**: predicted mixture viscosity for each DES-positive pair.

## How to Adapt

- Lower `--viscosity-weight` toward `0.3` to give more weight to Tm depression; raise it toward `1.0` to rank almost entirely by viscosity.
- Tighten `--abs-tm-threshold` to `280` for a stricter pharmaceutical window or relax it further for industrial applications.
- Add `--save-run-memory runs/betaine_run_001/run.memory.json` and use `label-run` to feed good/bad labels back into the next screening cycle.
- Compare two runs (e.g., with different viscosity weights) using `python -m des_multi_agent.cli compare-runs run_a.json run_b.json`.
