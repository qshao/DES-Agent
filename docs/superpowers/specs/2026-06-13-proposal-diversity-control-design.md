# Proposal Diversity Control Design

## Goal

Add a reusable proposal-diversity control that prevents the system from returning near-duplicate chemical proposals too often.

The first target is DES candidate generation. The controller should:

- reject exact duplicates and near-duplicates before ranking
- preserve useful chemical diversity inside a promising family
- optionally fall back to a nearby family when too many similar proposals are found
- remain reusable for metal-binding and selectivity workflows later

This feature should improve proposal quality without changing the deterministic prediction model or the final ranking logic.

## Scope

This design applies to the candidate-proposal stage before scoring and ranking.

In scope:

- exact duplicate suppression
- near-duplicate suppression using structural similarity
- family-aware fallback for DES proposal generation
- reusable diversity policy for future workflows
- tests and documentation

Out of scope:

- changing the deterministic DES predictor
- changing final ranking formulas
- adding autonomous multi-step planning
- adding memory-driven reranking bias
- adding a general search agent for retrosynthesis or synthesis planning

## User-Facing Behavior

The proposal-diversity controller should make candidate sets look more chemically useful in two ways:

1. It should remove trivial copies of the same proposal.
2. It should keep a core chemical family but reserve part of the proposal budget for nearby alternatives.

For DES workflows, the default behavior should be:

- keep the strongest family signal from the generator
- suppress exact duplicates
- suppress near-duplicates above a similarity threshold
- if too many similar proposals are suppressed, ask for nearby family alternatives rather than repeating the same motif

For metal-binding workflows later, the same control should at least suppress exact and near-duplicate proposals even if family fallback is not used.

## Architecture

### Reusable diversity controller

Add a small reusable controller that operates on `CandidateProposal` objects before final filtering and ranking.

The controller should expose a simple policy boundary such as:

- `max_similarity`
- `duplicate_policy`
- `family_fallback`
- `per_family_budget`

The exact method names can be adjusted during implementation, but the controller must clearly answer:

- is this proposal too similar to one we already have?
- should this proposal be suppressed?
- if suppressed, should the generator be asked for a nearby family alternative?

### Similarity handling

The controller should compare candidate structures using canonical SMILES and a lightweight structural similarity measure.

Required behavior:

- exact duplicates are removed first using canonical SMILES
- near-duplicates are then compared against the accepted set
- if a proposal is too similar to an accepted proposal, it is suppressed

The design should prefer a simple, testable fingerprint-based approach rather than a heavyweight chemistry model.

### DES family-aware fallback

For DES candidate generation, the controller may suggest a nearby family when it suppresses too many proposals from the same motif.

This fallback should stay advisory and bounded:

- it should preserve the core family when it is chemically useful
- it should reserve part of the budget for nearby families
- it should not turn into a general LLM planning loop

### Workflow reuse

The same controller should be reusable by metal-binding and selectivity workflows.

Reuse means:

- same duplicate and similarity suppression logic
- same candidate-policy interface
- workflow-specific fallback behavior can differ

## Configuration

Add proposal-diversity settings to the shared chemistry proposal layer or workflow config.

Suggested settings:

- `max_similarity: float = 0.85`
- `deduplicate_exact: bool = True`
- `deduplicate_near: bool = True`
- `family_fallback: bool = True` for DES
- `per_family_budget: int | None = None`

Constraints:

- `max_similarity` must be in `(0.0, 1.0]`
- `per_family_budget`, if set, must be positive

The controller should not depend on LLM availability. It should work for heuristic, discovery, and LLM-suggested proposals alike.

## Integration Points

### Candidate generation

Apply the controller after proposals are collected from all sources and before final filtering.

That means it should see:

- heuristic candidates
- discovery candidates
- LLM brainstorm candidates

This is important because duplicate suppression should work across sources, not just inside one source.

### DES workflow

In DES runs, if near-duplicate suppression removes too many candidates from a family, the controller should preserve the remaining budget by encouraging nearby chemical families.

The final scoring logic remains unchanged.

### Metal-binding workflows

For later reuse, metal-binding workflows can use the same controller in a simpler mode that only performs exact and near-duplicate suppression.

## Testing

Add tests for:

- exact duplicate suppression by canonical SMILES
- near-duplicate suppression by structural similarity
- cross-source duplicate suppression
- DES family-aware fallback keeping nearby families in the accepted set
- configuration validation for similarity threshold and family budget
- reuse of the same controller on a non-DES proposal list

Tests should not require live LLM calls.

## Risks and Mitigations

Risk:
A similarity threshold that is too aggressive could remove chemically meaningful alternatives.

Mitigation:
Keep the threshold configurable, use a conservative default, and test that family-level variety remains available.

Risk:
The controller could be mistaken for ranking logic.

Mitigation:
Document clearly that it only controls proposal diversity before final ranking.

Risk:
A family-aware fallback could become another hidden planner.

Mitigation:
Keep fallback bounded and advisory, and do not let it execute multiple loops.

## Success Criteria

The feature is complete when:

- exact duplicates are removed before ranking
- near-duplicates are suppressed across all proposal sources
- DES runs still preserve at least one productive family while allowing nearby alternatives
- the same controller can be reused by metal-binding workflows later
- docs and tests describe the proposal-diversity behavior clearly
