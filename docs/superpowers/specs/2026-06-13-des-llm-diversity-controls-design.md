# DES LLM Diversity Controls Design

## Goal

Add explicit controls for how much chemical variety the LLM should preserve when brainstorming DES partner candidates, especially across iterative cycles that already have prior successful families.

The default behavior should be `balanced`: keep some diversity across families while still biasing toward previously productive families.

## Scope

This design applies only to the DES LLM brainstorming path.

In scope:

- DES family selection prompt behavior
- DES candidate brainstorm prompt behavior
- LLM config fields for diversity behavior
- iterative-cycle family bias handling for DES workflows
- tests and user documentation

Out of scope:

- deterministic heuristic candidate generation
- metal-binding or metal-selectivity brainstorming controls
- run-memory ranking bias
- ranking, filtering, or prediction changes after candidates are generated

## User-Facing Behavior

Add three DES brainstorming modes:

- `explore`
- `balanced`
- `exploit`

Add two numeric controls:

- `max_families`
- `family_bias_strength`

Behavior:

- `explore`
  - prefer broader family spread
  - weak prior-family bias
  - encourage underused or chemically distinct families
- `balanced`
  - preserve family spread while still reusing prior productive families
  - this is the default
- `exploit`
  - strongly prefer prior productive families
  - allow diversity to narrow in later cycles

The deterministic DES predictor, uncertainty logic, reuse memory behavior, and final ranking remain unchanged.

## Configuration

Extend the LLM config with DES-only diversity settings:

- `diversity_mode: str = "balanced"`
- `max_families: int = 6`
- `family_bias_strength: float = 0.5`

Constraints:

- `diversity_mode` must be one of `explore`, `balanced`, `exploit`
- `max_families` must be a positive integer
- `family_bias_strength` must be in `[0.0, 1.0]`

These settings live in the shared LLM config because brainstorming already flows through the provider layer. They affect DES brainstorming only for now.

## Prompting Design

### Family selection

Update the DES family-selection prompt to include:

- diversity mode
- maximum number of families
- prior productive families, if any
- bias-strength instruction

Prompt intent by mode:

- `explore`: explicitly ask for chemically distinct families and limited reuse of prior top families
- `balanced`: ask for a mix of productive and novel families
- `exploit`: ask for families close to prior productive families unless strong chemistry suggests otherwise

### Candidate brainstorm

Update the DES candidate brainstorm prompt to include:

- diversity mode
- maximum number of families already chosen
- prior productive families, if any
- bias-strength instruction
- explicit distribution request across selected families

The prompt should stay JSON-only and keep the current two-stage structure.

## Iterative Context Handling

The orchestrator already passes prior top results and family ledgers into the DES brainstorming context.

Refine that behavior so the context clearly distinguishes:

- prior top candidate molecules
- prior productive families
- whether the next cycle should explore, balance, or exploit those families

No hard filtering is added. The prior family information remains advisory prompt context only.

## Architecture Changes

### `des_multi_agent/llm/config.py`

Add fields:

- `diversity_mode`
- `max_families`
- `family_bias_strength`

Validate each field.

### `des_multi_agent/llm/prompts.py`

Update:

- `family_selection_prompt`
- `candidate_brainstorm_prompt`

Add small helper text generators if needed so the mode-specific instructions stay readable and testable.

### `des_multi_agent/llm/base.py`

Pass the new config controls into:

- `select_candidate_families`
- `brainstorm_candidates`

Keep the public flow unchanged otherwise.

### `des_multi_agent/orchestrator.py`

Pass structured prior-family guidance into the DES brainstorming calls.

Prefer small helper functions rather than embedding mode logic into the main run path.

### Documentation

Update:

- `docs/tutorial.md`
- `examples/README.md` if needed

Explain how to choose between `explore`, `balanced`, and `exploit`.

## Testing

Add tests for:

- config parsing and validation for the three new fields
- prompt text containing the correct diversity-mode instructions
- prompt text containing prior-family guidance when available
- default behavior being `balanced`
- provider flow passing the new values through correctly

Do not add fragile tests that depend on live LLM outputs.

## Risks and Mitigations

Risk:

- more prompt complexity may reduce JSON reliability on weaker local models

Mitigation:

- keep instructions compact
- preserve the current raw-JSON-only contract
- avoid adding too many independent knobs

Risk:

- users may assume this changes final ranking behavior directly

Mitigation:

- document clearly that these settings affect brainstorming only

## Success Criteria

The feature is complete when:

- users can configure DES brainstorming with `explore`, `balanced`, or `exploit`
- `balanced` is the default
- prompts include the intended diversity behavior and prior-family context
- iterative DES runs can bias brainstorming without collapsing diversity by default
- docs and tests cover the new controls
