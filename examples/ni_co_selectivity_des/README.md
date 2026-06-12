# Ni²⁺/Co²⁺ Selectivity-DES Example

This example runs the full **selectivity-DES pipeline** to find ligands that:

1. Bind Ni²⁺ selectively over Co²⁺ (Phase 1 — metal-ion selectivity screening)
2. Form a deep eutectic solvent with a low-viscosity, low-melting partner (Phase 2 — DES search)

Gemma4-12B (Ollama) is used as the LLM for two-stage brainstorming in both phases.

## Scientific goal

Selective extraction of Ni²⁺ from a Co²⁺-containing solution is industrially important (battery recycling, hydrometallurgy). A ligand that is both selective *and* DES-compatible enables solvent-extraction processes that are greener and lower-cost than conventional organic solvents.

## Pipeline architecture

```
Outer loop (2 cycles)
├── Phase 1: Ni²⁺/Co²⁺ selectivity screening
│   ├── LLM brainstorms selective ligand candidates (20 per cycle, 3 cycles)
│   ├── Heuristic stability-constant model scores each ligand
│   ├── Score = 0.4 × log K(Ni²⁺) + 0.6 × ΔlogK   (selectivity-weighted)
│   └── Top 3 ligands (ΔlogK ≥ 0.3) pass to Phase 2
│
└── Phase 2: DES partner search for each shortlisted ligand
    ├── 20 candidates per cycle × 3 cycles per ligand
    ├── ChemBERTa ML model predicts eutectic melting temperature
    ├── Viscosity model gates candidates above 200 cP
    ├── Acceptance criteria: Tm ≤ 350 K, viscosity ≤ 200 cP
    └── DES-compatible ligands feed back to Phase 1 as hints
```

## Prerequisites

- Ollama running locally with `gemma4:12b` loaded — check with:
  ```bash
  curl -s http://localhost:11434/api/tags | python3 -c "import sys,json; [print(m['name']) for m in json.load(sys.stdin)['models']]"
  ```
- ChemBERTa checkpoint: `ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt`
- Stability-constant artifact: `artifacts/stability_constants/model.json`
- Viscosity artifact: `artifacts/designsolvents/viscosity/model.json`
- Verify everything with:
  ```bash
  python -m des_multi_agent.cli doctor --check checkpoint --check artifacts
  ```

## Run

```bash
./run.sh
# or with a custom output directory:
./run.sh runs/my_ni_co_run
```

Expected wall time: **10–25 minutes** (dominated by Gemma4-12B LLM calls).

Progress is streamed to stderr during the run:

```
[outer 1/2] phase 1: selectivity screening
[cycle 1/3] screened=18 top_score=5.99
...
[outer 1/2] phase 2: DES search for 3 ligand(s)
[outer 1/2] phase 2: ligand 1/3 — <SMILES>
[1/6] Generating candidates for <SMILES>...
...
[outer 2/2] DES-compatible set stable — converged early
```

## Output

The file [`output.txt`](./output.txt) contains two sections:

**Section 1 — Selectivity table** (Phase 1 final results)

| Column | Meaning |
|--------|---------|
| `ligand` | Candidate SMILES |
| `log_k_target` | Predicted log K for Ni²⁺ |
| `log_k_competitor` | Predicted log K for Co²⁺ |
| `delta_log_k` | `log K(Ni²⁺) − log K(Co²⁺)`; positive = selective for Ni²⁺ |
| `score` | Composite ranking score |
| `des_compatible` | Whether a DES partner was found in Phase 2 |

**Section 2 — DES partner blocks** (Phase 2 results per ligand)

Each ligand gets its own DES screening table with Tm, viscosity, and rationale columns.

## Key parameters and how to tune them

| Parameter | Flag | Default here | Effect |
|-----------|------|-------------|--------|
| Selectivity emphasis | `--selectivity-weight` | 0.6 | Increase to prioritize ΔlogK over absolute affinity |
| Ligands passed to DES | `--top-ligands` | 3 | Increase to explore more ligands at higher compute cost |
| Selectivity gate | `--min-delta-log-k` | 0.3 | Raise to require stronger Ni²⁺/Co²⁺ discrimination |
| Tm ceiling | `--abs-tm-threshold` | 350 K | Lower for room-temperature liquids |
| Viscosity gate | `--viscosity-threshold` | 200 cP | Lower for more flowable solvents |
| Outer cycles | `--n-outer-cycles` | 2 | Increase if you want more feedback-loop refinement |

## LLM config

[`llm.ni_co_selectivity.yaml`](../../llm.ni_co_selectivity.yaml) points to `gemma4:12b` on Ollama at `localhost:11434`. Edit `model_name` to switch to `nemotron-3-nano:latest` or `qwen3.6` without changing anything else.

## How to adapt for other metal pairs

```bash
python -m des_multi_agent.cli \
  --workflow selectivity-des \
  --target-metal-ion "Cu2+" \
  --competitor-metal-ion "Fe3+" \
  --n 20 --n-cycles 3 \
  --n-des-candidates 20 --n-des-cycles 3 \
  --n-outer-cycles 2 --top-ligands 3 --min-delta-log-k 0.5 \
  --abs-tm-threshold 350 \
  --viscosity-model-path artifacts/designsolvents/viscosity/model.json \
  --viscosity-threshold 200 --viscosity-weight 0.4 \
  --checkpoint-path ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt \
  --config-path ml_des_mp/config.yaml \
  --stability-constant-model-path artifacts/stability_constants/model.json \
  --llm-config llm.ni_co_selectivity.yaml
```
