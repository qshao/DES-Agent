# Optional LLM Candidate Brainstorming Design

**Goal:** Add an optional LLM-assisted layer to the DES multi-agent system that can brainstorm additional candidate partners, generate human-readable explanations, and provide advisory critique of the deterministic DES ranking without changing the final model-based score.

**Architecture:** The system will keep the current deterministic pipeline as the source of truth for candidate filtering, melting-temperature prediction, and DES classification. A new optional LLM layer will sit beside it and provide three services: candidate brainstorming, explanation generation, and advisory critique of ranked results. The LLM will be selected by configuration, with support for either a hosted API provider or a local model provider through a shared adapter interface. If the LLM fails or returns malformed output, the system falls back to the deterministic path.

**Tech Stack:** Python, the existing `des_multi_agent` and `ml_des_mp` packages, optional provider SDKs for local or hosted LLMs, JSON parsing for structured output validation, and `pytest` for regression tests.

---

## Scope

This design covers an optional, non-authoritative LLM layer.

- Input: component `A`, requested candidate count `N`, existing deterministic candidate/ranking results, and optional user constraints.
- Output: optional LLM-generated candidate suggestions, explanation text, and critique notes attached to deterministic results.
- Out of scope: letting the LLM assign the final DES label, replacing the thermodynamic model, or making the pipeline dependent on network access.

## Existing Assets

The repository already contains the deterministic workflow:

- `des_multi_agent/candidate_generation.py` proposes rule-based partner candidates.
- `des_multi_agent/chemistry_filter.py` filters invalid or duplicated candidates.
- `des_multi_agent/prediction.py` scores candidate pairs using `ml_des_mp`.
- `des_multi_agent/evaluation.py` and `des_multi_agent/ranking.py` assign the final DES result.
- `des_multi_agent/reporting.py` formats the final output.

The optional LLM layer should augment these assets rather than replace them.

## System Roles

### 1. LLM Candidate Brainstormer

Purpose: propose additional candidate partners beyond the deterministic rule-based families.

Behavior:

- Takes component `A`, optional constraints, and a short chemistry context summary.
- Returns a bounded list of candidate SMILES strings with a rationale for each one.
- Never claims a final DES label.
- Must obey the same chemistry sanity filters as deterministic candidates before scoring.

Recommended use:

- Augment the deterministic candidate pool, not replace it.
- Cap the number of LLM-proposed candidates per run to avoid combinatorial drift.

### 2. LLM Explanation Generator

Purpose: write concise human-readable explanations for deterministic candidate rankings and DES outcomes.

Behavior:

- Consumes the final ranked results and their deterministic evidence.
- Produces a short per-candidate summary explaining why a candidate ranked well or poorly.
- Can explain the absolute and relative threshold outcomes, but must not invent new scores.

### 3. LLM Critique Agent

Purpose: provide advisory review of the deterministic ranking.

Behavior:

- Reviews the top-ranked results for obvious chemistry concerns, suspicious outliers, or likely out-of-distribution candidates.
- Produces advisory notes such as “high confidence”, “possible false positive”, or “chemically unusual”.
- Cannot change the final ranking, DES label, or prediction values.

## Provider Interface

The LLM layer should use a provider abstraction so the rest of the system does not depend on whether the model is local or remote.

Supported provider modes:

- `disabled`: no LLM calls, pure deterministic pipeline
- `local`: local/open-source model served on the machine
- `hosted`: external API model selected by config

Required adapter methods:

- `brainstorm_candidates(component_a, constraints, context) -> list[CandidateProposal]`
- `generate_explanations(results, context) -> list[ExplanationNote]`
- `critique_results(results, context) -> list[CritiqueNote]`

## Data Flow

1. The deterministic candidate generator proposes baseline partner candidates.
2. If enabled, the LLM brainstormer proposes supplemental candidates.
3. The chemistry filter removes invalid or duplicate candidates from both sources.
4. The prediction agent scores all surviving candidates using `ml_des_mp`.
5. The evaluation agent applies the DES thresholds and the ranking agent sorts results.
6. If enabled, the LLM explanation and critique agents annotate the final ranked output.
7. The final report includes deterministic scores plus optional LLM notes.

## Structured Output

LLM responses should be treated as structured data, not free-form text.

Expected fields:

- candidate brainstorming: `smiles`, `rationale`, `family`
- explanation notes: `smiles`, `summary`, `key_evidence`
- critique notes: `smiles`, `assessment`, `concerns`

The system should reject malformed output and fall back to deterministic-only output for that run.

## Configuration

The feature should be controlled by configuration values, not hard-coded branching.

Recommended config keys:

- `llm.enabled`
- `llm.provider` with values `disabled`, `local`, or `hosted`
- `llm.model_name`
- `llm.api_base_url`
- `llm.api_key_env`
- `llm.max_candidates`
- `llm.max_tokens`
- `llm.temperature`
- `llm.timeout_seconds`

## Error Handling

- If the provider is disabled, the deterministic pipeline runs unchanged.
- If the provider returns invalid JSON or missing required fields, the system discards the LLM contribution for that step and continues.
- If the provider times out or raises an API error, the system logs the failure and falls back to deterministic-only output.
- If the LLM proposes a duplicate or invalid SMILES string, the chemistry filter removes it before scoring.

## Testing Strategy

The first implementation should be covered by focused tests.

- Provider selection tests should verify that config maps to the right adapter class.
- Brainstorming tests should verify that valid LLM candidates are merged with deterministic candidates and invalid ones are filtered.
- Explanation tests should verify that the LLM note is attached to the correct ranked result and does not modify the score.
- Critique tests should verify that critique notes are advisory metadata only.
- Fallback tests should verify that malformed LLM output or provider errors still return deterministic results.

## Implementation Boundaries

- The LLM layer should live in a separate package or module tree so it can be disabled cleanly.
- Deterministic scoring and DES classification remain the final authority.
- The optional LLM layer should not be required for existing CLI usage.
- The system should work with either local or hosted providers selected by config.

## Risks and Mitigations

- Risk: LLM output may be malformed or non-chemical.
  - Mitigation: require structured output and apply RDKit validation before use.
- Risk: the LLM could bias ranking too strongly.
  - Mitigation: keep ranking and DES labels deterministic; LLM only annotates results.
- Risk: hosted providers may introduce network or cost dependency.
  - Mitigation: keep a local provider mode and a disabled mode.
- Risk: prompt drift could make explanations inconsistent with the actual scores.
  - Mitigation: feed the LLM only the deterministic evidence and validate outputs against the source results.

