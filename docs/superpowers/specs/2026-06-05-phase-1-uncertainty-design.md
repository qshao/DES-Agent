# Phase 1 Uncertainty Layer Design

**Goal:** Add an uncertainty layer to the DES multi-agent system that estimates confidence for the predicted minimum melting temperature of each candidate pair, using both a heuristic trust score and a 3-pass model-based uncertainty estimate. The uncertainty must be visible in the report and must influence filtering and ranking.

**Architecture:** The deterministic DES pipeline remains the source of truth for the final minimum melting-temperature prediction and DES classification. Phase 1 adds a post-processing uncertainty layer that wraps the existing pair-level predictor: it runs the prediction three times per pair, aggregates the minimum melting-temperature statistics, computes a normalized trust score in the range `0.0` to `1.0`, and applies configurable filtering or ranking penalties based on the resulting uncertainty. The uncertainty layer is advisory for the model output but authoritative for candidate triage: low-trust candidates can be filtered out or demoted before the final ranked report is produced.

**Tech Stack:** Python, RDKit for chemistry sanity checks, the existing `des_multi_agent` orchestration code, the existing `ml_des_mp` predictor and checkpoints, and `pytest` for verification.

---

## Scope

This design is limited to Phase 1 and to the minimum melting-temperature prediction target.

- Input: component `A` as SMILES, candidate `B` as SMILES, existing configuration, and the existing `ml_des_mp` checkpoint path.
- Output: per-candidate uncertainty metadata for the minimum predicted melting temperature, including repeated-prediction statistics, a trust score, and an uncertainty flag.
- In scope: heuristic trust score, model-based repeated prediction, uncertainty-aware filtering and ranking, and report formatting.
- Out of scope: full melting-curve uncertainty, ensemble retraining, active learning, literature search, external database integration, and any change to the final deterministic DES label logic.

## Existing Assets

The current repository already provides the core deterministic pipeline:

- `des_multi_agent/prediction.py` predicts the melting curve for a candidate pair.
- `des_multi_agent/evaluation.py` derives the DES label from the curve and thresholds.
- `des_multi_agent/orchestrator.py` coordinates candidate generation, filtering, property resolution, prediction, and ranking.
- `des_multi_agent/reporting.py` formats the final CLI output.
- `des_multi_agent/chemistry_filter.py` already performs canonicalization and basic plausibility checks.

Phase 1 should extend these assets rather than replace them.

## Design Principles

- Keep the deterministic DES label unchanged.
- Treat uncertainty as a first-class output, not a hidden implementation detail.
- Use a numeric trust score in `[0.0, 1.0]` where `1.0` means fully trusted.
- Run the model-based uncertainty estimate exactly three times per candidate pair.
- Make the uncertainty policy configurable so users can choose hard filtering or soft ranking penalties.

## System Roles

### 1. Uncertainty Prediction Agent

Purpose: estimate instability in the predicted minimum melting temperature.

Behavior:

- Runs the existing pair-level predictor three times for the same `(A, B)` pair.
- Computes:
  - `tm_min_pred_k_mean`
  - `tm_min_pred_k_std`
  - `tm_min_pred_k_min`
  - `tm_min_pred_k_max`
- Derives an uncertainty band from the spread of the three minimum-Tm values.
- Exposes the raw repeated values for debugging and testability.

Implementation notes:

- The repeated passes should reuse the existing inference path and checkpoint loading logic.
- If the backend is deterministic in a given environment, the spread can collapse to zero; the design still requires three passes so the model-based pathway is explicit and testable.
- The API should return a single structured object so the orchestrator can consume it directly.

### 2. Heuristic Trust Agent

Purpose: compute a normalized confidence score for a candidate pair.

Behavior:

- Produces a trust score in `[0.0, 1.0]`.
- Starts from a score of `1.0` and subtracts penalties for:
  - chemically unusual or borderline candidate structures
  - low similarity to candidate families already seen in the deterministic generator
  - large spread in the 3-pass minimum-Tm estimate
  - unresolved or low-confidence neat-component melting-point estimates
- Produces a short explanation for the final score.

Recommended interpretation:

- `1.0` means highly trusted.
- `0.7` to `0.9` means usable but worth reviewing.
- `0.4` to `0.69` means caution.
- below `0.4` means low trust.

### 3. Uncertainty Filter and Ranker

Purpose: use uncertainty to triage candidates before final presentation.

Behavior:

- Applies a configurable minimum trust threshold.
- Can operate in one of three modes:
  - `filter`: remove candidates below threshold
  - `penalize`: keep all candidates but demote low-trust ones in the final ranking
  - `report_only`: do not affect ranking, but still show uncertainty in the report
- Applies a soft penalty to ranking score when uncertainty is above a configured limit.
- Keeps the deterministic DES label untouched.

### 4. Reporting Agent

Purpose: expose uncertainty in the CLI and demo output.

Behavior:

- Shows the repeated minimum-Tm values or a compact summary of them.
- Shows the mean and spread for the minimum-Tm estimate.
- Shows the trust score and the uncertainty band/flag.
- Includes a brief rationale for the trust score and any filtering or demotion applied.

## Data Model

The following structures should be available to the orchestrator and reporter:

- `UncertaintyEstimate`
  - `tm_min_values: list[float]`
  - `tm_min_mean_k: float`
  - `tm_min_std_k: float`
  - `tm_min_min_k: float`
  - `tm_min_max_k: float`
  - `trust_score: float`
  - `uncertainty_flag: str`
  - `explanation: str`
- `UncertaintyPolicy`
  - `mode: str`
  - `min_trust_score: float`
  - `soft_penalty_weight: float`
  - `std_high_threshold_k: float`
  - `std_medium_threshold_k: float`

## Data Flow

1. The orchestrator generates and filters candidate partners as today.
2. For each surviving pair, the uncertainty prediction agent runs the minimum-Tm predictor three times.
3. The heuristic trust agent computes a normalized trust score from chemistry and stability features.
4. The uncertainty policy decides whether to filter, penalize, or only report the uncertainty.
5. The ranking step combines the deterministic DES result and the uncertainty policy outcome.
6. The reporting step prints the uncertainty metadata alongside the existing DES summary.

## Interfaces

The first implementation should expose a small internal API:

- `estimate_min_tm_uncertainty(component_a: str, component_b: str, checkpoint_path: str, config_path: str) -> UncertaintyEstimate`
- `score_candidate_trust(component_a: str, component_b: str, tm_uncertainty: UncertaintyEstimate, neat_component_status: dict) -> float`
- `apply_uncertainty_policy(results: list[DesResult], uncertainty: dict[str, UncertaintyEstimate], policy: UncertaintyPolicy) -> list[DesResult]`
- `format_uncertainty(estimate: UncertaintyEstimate) -> str`

The new uncertainty objects should be attached to candidate-level result records so downstream consumers can inspect them without re-running the predictor.

## Policy and Thresholds

The policy must be configurable and should not be hard-coded in the orchestration logic.

Recommended defaults:

- `mode = "penalize"`
- `min_trust_score = 0.55`
- `soft_penalty_weight = 0.35`
- `std_medium_threshold_k = 5.0`
- `std_high_threshold_k = 15.0`

Interpretation:

- If `trust_score < min_trust_score` and `mode == "filter"`, drop the candidate.
- If `trust_score < min_trust_score` and `mode == "penalize"`, keep the candidate but demote it.
- If `mode == "report_only"`, keep the candidate and do not alter rank order.

## Error Handling

- If the three prediction passes cannot all be run, the system should mark the uncertainty as unavailable and fall back to a conservative low-trust classification.
- If the candidate cannot be embedded or the checkpoint cannot be loaded, the orchestrator should fail fast before scoring begins, as it does today.
- If the trust score cannot be computed, the system should treat the candidate as low trust rather than high trust.
- If the uncertainty policy is missing or malformed, the run should fail fast rather than silently ignoring uncertainty.

## Testing Strategy

The first implementation should be covered by focused tests.

- Repeated-prediction tests should verify that three runs are executed and aggregated into mean, standard deviation, and range values.
- Trust-score tests should verify that the score is normalized to `[0.0, 1.0]` and that stronger instability lowers the score.
- Policy tests should verify:
  - filtering below threshold
  - demotion under penalize mode
  - no ranking change under report-only mode
- Reporting tests should verify that the uncertainty summary appears in the CLI output and the demo output.
- End-to-end tests should verify that uncertainty metadata is present for each candidate and affects the final ordering when enabled.

## Implementation Boundaries

- The uncertainty layer should live alongside the current deterministic orchestration code.
- The minimum-Tm predictor should remain the source of truth for the actual temperature estimate.
- Phase 1 should not add new external service dependencies.
- Phase 1 should not change the DES classification formula itself.

## Risks and Mitigations

- Risk: three repeated passes may still be deterministic in some environments.
  - Mitigation: keep the three-pass structure for explicitness, and surface zero spread as a valid low-variance outcome.
- Risk: a heuristic trust score can become arbitrary.
  - Mitigation: base the score on concrete, explainable features and keep the formula configurable.
- Risk: uncertainty-based filtering may hide useful candidates.
  - Mitigation: allow `report_only` mode and keep the threshold configurable.
- Risk: the uncertainty output could clutter the report.
  - Mitigation: format it compactly and attach only the essential summary values by default.

