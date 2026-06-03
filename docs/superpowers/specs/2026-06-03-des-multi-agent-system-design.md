# DES Multi-Agent System Design

**Goal:** Build a multi-agent workflow that takes a user-proposed component `A`, proposes `N` chemically plausible partner components `B` from general chemistry knowledge, uses the trained model in `ml_des_mp` to predict melting temperature across a molar-fraction grid from `0.1` to `0.9`, and classifies whether each pair is likely to form a deep eutectic solvent.

**Architecture:** The first version will be a deterministic orchestration layer with a small set of focused agents. A candidate-generation agent proposes partner components from rule-based chemistry heuristics, a chemistry-filter agent removes invalid or implausible candidates, a property-resolution agent estimates or validates neat-component melting points, a prediction agent wraps the existing `ml_des_mp` checkpoint and inference code, and an evaluation agent applies configurable DES criteria using both an absolute melting-point cutoff and a relative reduction threshold. A reporting agent packages the ranked results, ratio curve summaries, and rationale for downstream use or human review.

**Tech Stack:** Python, RDKit for chemical validation and basic property checks, PyTorch for model inference, the existing `ml_des_mp` code and checkpoints, and a lightweight orchestration layer built inside the current repository.

---

## Scope

This design covers the first working version only.

- Input: one component `A` as SMILES, an integer `N`, and optional constraints such as allowed element classes or desired partner type.
- Output: a ranked list of candidate pairs `(A, B)` with predicted melting curves on the `0.1` to `0.9` molar-fraction grid, DES classification, and concise explanations.
- Out of scope for v1: autonomous self-refinement loops, external database search, synthesis planning, and wet-lab recommendation policies.

## Existing Assets

The repository already contains the model stack needed for prediction:

- `ml_des_mp/predict.py` loads a trained checkpoint and predicts `Tm` for a provided CSV row.
- `ml_des_mp/src/train.py` defines the thermodynamic prediction formula used by training and inference.
- `ml_des_mp/src/models/model.py` and `ml_des_mp/src/models/physics_core.py` define the model and melting-curve computation.
- `ml_des_mp/config.yaml` defines the active embedding profile and training settings.

The multi-agent system should reuse these assets rather than reimplementing the model math.

## System Roles

### 1. Candidate Generator Agent

Purpose: propose plausible partner components `B` for a user-supplied component `A`.

Behavior:

- Uses general chemistry heuristics instead of an external database.
- Generates candidates from broad families that are commonly associated with deep eutectic solvents, such as hydrogen-bond donors, hydrogen-bond acceptors, quaternary ammonium salts, amides, alcohols, carboxylic acids, and polyols.
- Avoids returning obviously unstable or nonsensical structures.
- Produces SMILES strings plus a short rationale for each proposal.

### 2. Chemistry Filter Agent

Purpose: remove invalid or low-quality candidates before model scoring.

Behavior:

- Validates SMILES with RDKit.
- Rejects duplicates, self-pairs, and candidates that fail basic chemistry sanity checks.
- Applies lightweight heuristics such as element whitelist, charge sanity, and simple functional-group plausibility.
- Preserves traceability by returning the reason each candidate was kept or removed.

### 3. Prediction Agent

Purpose: evaluate each remaining `(A, B)` pair with the existing trained model.

Behavior:

- Loads a checkpoint from `ml_des_mp/runs/`.
- Reuses the repository's embedding and thermodynamic prediction pipeline.
- Predicts melting temperature across a fixed molar-fraction grid covering `0.1` to `0.9`.
- Produces a melting-curve summary rather than only a single-point score.

Implementation note:

- The first version should treat the existing predictor as the source of truth.
- For non-GNN checkpoints, inference can follow the same pattern as `ml_des_mp/predict.py`.
- The agent should expose a pair-level API so the orchestrator can score many candidates in batch.

### 4. Property-Resolution Agent

Purpose: provide neat-component melting points for the ML model and evaluation logic.

Behavior:

- Accepts a component SMILES string and returns a neat-component melting-point estimate in Kelvin.
- Uses a local heuristic or cached lookup table first.
- Allows user-supplied values to override estimates when known.
- Emits a confidence flag so downstream logic can distinguish measured, cached, and estimated values.

Implementation note:

- The first version can use a simple rule-based estimator plus an override hook.
- The orchestrator should keep the source of the temperature explicit in the result payload.

### 5. DES Evaluation Agent

Purpose: decide whether a candidate pair is likely to be a DES.

Behavior:

- Applies two rules together:
  - an absolute predicted melting-temperature cutoff
  - a relative reduction threshold versus the neat-component melting points
- Evaluates the full predicted curve across the ratio grid and records where the criteria are satisfied.
- Returns a boolean label plus an explanation that references the thresholds used.

Recommended interpretation:

- Absolute cutoff: the predicted minimum or ratio-averaged melting temperature must be below a configurable threshold.
- Relative reduction: the predicted melting temperature must drop by at least a configurable percentage relative to the lower of the two neat-component melting points, or relative to both neat components if the user wants stricter screening.

The exact threshold values should live in configuration, not hard-coded in the agent logic.

### 6. Ranking and Reporting Agent

Purpose: present results in a usable form.

Behavior:

- Sorts candidate pairs by predicted DES strength, lowest predicted melting point, and consistency across the ratio window.
- Produces a compact table with:
  - component `A`
  - candidate `B`
  - predicted curve summary
  - DES classification
  - threshold outcomes
  - rationale from the generation agent
- Emits machine-readable output for downstream automation and a human-readable summary for review.

## Data Flow

1. The user provides component `A` as SMILES and requests `N` candidate partners.
2. The candidate generator proposes a larger pool of plausible `B` structures.
3. The chemistry filter removes invalid or redundant candidates.
4. The property-resolution agent obtains neat-component melting points for `A` and each candidate `B`.
5. The prediction agent scores each remaining pair across the molar-fraction grid `0.1` to `0.9`.
6. The evaluation agent applies the DES rules.
7. The ranking agent orders the surviving pairs and returns the top results.

This should run as a single orchestration request, but each agent must remain independently testable.

## Interfaces

The first implementation should expose a small internal API:

- `generate_candidates(component_a: str, n: int, constraints: dict | None) -> list[CandidateProposal]`
- `filter_candidates(component_a: str, candidates: list[CandidateProposal]) -> list[CandidateProposal]`
- `resolve_melting_point(component: str, override_k: float | None = None) -> MeltingPointEstimate`
- `predict_curve(component_a: str, component_b: str, checkpoint_path: str) -> CurvePrediction`
- `classify_des(curve: CurvePrediction, thresholds: DesThresholds) -> DesResult`
- `rank_results(results: list[DesResult]) -> list[DesResult]`

Each structure should include the SMILES strings, the predicted ratio grid, the predicted melting temperatures, the neat-component temperatures used for inference, and a text rationale.

## Error Handling

- Invalid input SMILES should fail early with a clear error message.
- If the checkpoint is missing or incompatible with the selected embedding profile, inference should stop before candidate scoring begins.
- If a candidate cannot be embedded or parsed, the system should skip it and record the reason.
- If a neat-component melting point cannot be resolved, the system should fail fast unless the user provided an explicit override.
- If fewer than `N` valid candidates remain after filtering, the system should return the available candidates and report the shortfall.
- If the absolute or relative threshold configuration is missing, the run should fail fast rather than silently using defaults.

## Testing Strategy

The first implementation should be covered by focused tests at the agent boundary.

- Candidate generation tests should verify that the generator returns chemically plausible SMILES strings for a simple example input.
- Filter tests should verify that invalid SMILES, duplicates, and self-pairs are removed.
- Prediction tests should verify that the orchestrator can load an existing checkpoint and produce a curve over the `0.1` to `0.9` grid.
- DES classification tests should verify that both the absolute and relative criteria must be satisfied.
- End-to-end tests should verify that a short request with a known `A` returns ranked candidates and a structured report.

## Implementation Boundaries

- The multi-agent layer should live alongside the existing project rather than replacing it.
- The `ml_des_mp` package should remain the prediction backend.
- The first version should not depend on external services, browser sessions, or database access.
- The candidate generator should be rule-based and deterministic unless a later revision explicitly adds stochastic sampling.

## Risks and Mitigations

- Risk: rule-based candidate generation may be narrow.
  - Mitigation: make the generator modular so new chemistry families can be added without rewriting the orchestrator.
- Risk: the DES criterion may be too strict or too loose.
  - Mitigation: keep thresholds configurable and return the curve evidence used for classification.
- Risk: the existing model may not generalize to out-of-distribution candidates.
  - Mitigation: surface prediction uncertainty or score stability in later versions, but keep v1 deterministic.
- Risk: using only general chemistry rules may miss valid partners.
  - Mitigation: allow the candidate generator to be expanded later with database-backed or LLM-assisted search.

