# Ni²⁺/Co²⁺ Selectivity-DES Example — Nemotron-3-Nano

This example runs the full **selectivity-DES pipeline** using **Nemotron-3-Nano** (Ollama) to find:

1. Five Ni²⁺-selective HBD/HBA ligands (Phase 1)
2. Deep eutectic solvents containing the top three of those ligands, with Tm < 350 K, viscosity ≤ 200 cP, and hydrophobic character (Phase 2)

## Scientific goal

Selective extraction of Ni²⁺ from Co²⁺-containing solutions is industrially relevant (battery recycling, hydrometallurgy). This example uses Nemotron-3-Nano as a fast, lightweight LLM alternative and shortlists **five** ligands in Phase 1 for a wider selectivity pool before DES screening.

## Pipeline architecture

```
Outer loop (2 cycles)
├── Phase 1: Ni²⁺/Co²⁺ selectivity screening
│   ├── Nemotron-3-Nano brainstorms HBD/HBA candidates (20/cycle × 3 cycles)
│   ├── Heuristic stability-constant model scores each ligand
│   ├── Score = 0.4 × log K(Ni²⁺) + 0.6 × ΔlogK
│   └── Top 5 ligands shortlisted; top 3 pass to Phase 2
│
└── Phase 2: DES partner search for top-3 ligands
    ├── 20 candidates per cycle × 3 cycles per ligand
    ├── ChemBERTa ML model predicts eutectic Tm
    ├── Viscosity model gates candidates above 200 cP
    ├── Acceptance: Tm ≤ 350 K, viscosity ≤ 200 cP
    └── DES-compatible ligands feed back to Phase 1
```

## Prerequisites

- Ollama running locally with `nemotron-3-nano:latest` loaded
- ChemBERTa checkpoint: `ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt`
- Artifacts: `artifacts/stability_constants/model.json`, `artifacts/designsolvents/viscosity/model.json`

## Run

```bash
./run.sh
./run.sh runs/my_nemotron_run   # custom output dir
```

Expected wall time: **15–30 minutes** (Nemotron-3-Nano is 24 GB — generation is slower than Gemma4-12B despite the "nano" name).

## LLM config

[`llm.ni_co_nemotron.yaml`](../../llm.ni_co_nemotron.yaml) — change `model_name` to switch LLMs.

## Related examples

- [`ni_co_selectivity_des/`](../ni_co_selectivity_des) — same task with Gemma4-12B, top-3 ligands
- [`ni_co_selectivity_des_qwen36/`](../ni_co_selectivity_des_qwen36) — same task with Qwen 3.6 (requires `ollama pull qwen3.6`)
