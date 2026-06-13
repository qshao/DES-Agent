# Chemistry Advisor Design

## Goal

Add a reusable LLM-based chemistry-advisor layer that improves how the system proposes, reviews, and explains chemical candidates.

The first target is the DES workflow. The advisor should:

- propose more chemically intuitive candidates
- explain why candidates look plausible or implausible
- flag likely failure modes
- suggest the next chemistry-aware search step
- reuse successful reasoning patterns from prior runs when available

This feature should increase chemistry intuition without replacing deterministic property prediction or making the system dependent on online services.

## Scope

This design applies to the DES workflow first, but the advisor should be reusable by other workflows later, especially metal-binding and selectivity tasks.

In scope:

- LLM-assisted candidate proposal refinement
- chemistry rationales for candidate choices
- chemistry warnings for likely failure modes
- hybrid next-step suggestions in the report
- optional use of run memory as a soft prior for reasoning patterns
- tests and documentation

Out of scope:

- changing the deterministic DES prediction model
- replacing numerical ranking with LLM judgment
- autonomous multi-step planning without user review
- online-only dependencies
- metal-binding implementation details beyond future reuse boundaries

## User-Facing Behavior

The advisor should be able to support three kinds of chemistry-intuitive output:

1. `proposal`
   - improve candidate generation
   - prefer chemically plausible, constraint-aware candidates
   - reject obviously poor chemistry early

2. `reasoning`
   - explain why a candidate is plausible or risky
   - summarize motifs, donor/acceptor patterns, and chemistry caveats

3. `next-step guidance`
   - recommend one safe follow-up action
   - recommend one broader chemistry alternative

For the first version, all three should be available inside the DES workflow.

Desired behavior in reports:

- top candidates should include short chemistry rationales
- obvious risks should be called out explicitly
- the final report should include one conservative next step and one exploratory next step
- if run memory has useful prior reasoning, the advisor may reuse that pattern as a soft prompt prior

## Architecture

### New reusable advisor component

Add a reusable advisor layer, likely as a small module under the LLM package or a shared chemistry-advice module.

The advisor should expose structured methods such as:

- `score_candidate_chemistry(...)`
- `explain_candidate(...)`
- `flag_candidate_risks(...)`
- `suggest_next_step(...)`

The exact method names can be adjusted during implementation, but the boundary should stay clear:

- input: candidate, context, workflow state, prior reasoning memory
- output: short structured chemistry guidance
- dependency: existing LLM provider interface

### Workflow integration

The DES workflow should call the advisor in three places:

1. after candidate generation
   - to improve or filter proposal quality

2. after ranking
   - to explain the top candidates and identify caveats

3. during report generation
   - to add hybrid next-step advice

The advisor should not replace the numeric predictor or uncertainty logic. It should annotate and guide the workflow.

### Memory integration

Run memory should be used as a soft prior only.

The advisor may read prior successful reasoning patterns from memory, but it should not:

- force a previous conclusion onto a new run
- override current chemistry evidence
- depend on memory being present

If memory is absent, the advisor should still work normally.

## Design Choices

### Candidate proposal layer

The proposal layer should favor chemically plausible candidates that satisfy the user’s constraints and look consistent with known DES chemistry.

The advisor should be able to identify patterns such as:

- donor/acceptor balance
- functional groups that are likely to participate in hydrogen bonding
- obvious instability or incompatibility issues
- clearly implausible partner combinations

The output should remain compact and structured so it can be added to reports without overwhelming them.

### Reasoning layer

The reasoning layer should produce short chemistry explanations in plain language.

Good outputs include:

- why a candidate is chemically sensible
- what motif makes it plausible
- what chemical uncertainty remains

The reasoning layer should be advisory, not authoritative. It should not claim certainty when the evidence is weak.

### Next-step guidance layer

The next-step guidance layer should output two suggestions:

- one conservative step
- one exploratory step

Examples of conservative steps:

- tighten the candidate family set
- rerank with narrower constraints
- rerun with a smaller family budget

Examples of exploratory steps:

- shift donor/acceptor families
- relax one constraint and rescan nearby space
- try a chemically adjacent family set

## Implementation Boundaries

### DES workflow only first

The first implementation should ship inside the DES workflow.

That keeps the feature bounded and easy to verify while still creating a reusable advisor boundary for later workflows.

### Reusable advisor, not full planner

The advisor should remain a chemistry-advice component, not a general-purpose autonomous planner.

It may recommend the next step, but it should not execute the next step automatically.

### Structured outputs

Outputs should be structured enough to support both reports and tests.

Preferred output fields are things like:

- candidate SMILES
- short rationale
- risk notes
- suggested next action

## Testing

Add tests for:

- advisor prompt construction
- advisor output parsing
- report integration for rationales and warnings
- next-step suggestions containing both conservative and exploratory advice
- optional memory-aware behavior when prior reasoning is present
- fallback behavior when memory is absent or empty

Tests should not require live LLM calls.

## Risks and Mitigations

Risk:
The advisor may become too verbose and drown out the core DES results.

Mitigation:
Keep the outputs short, structured, and limited to the top candidates and the final report.

Risk:
The advisor may overstate chemistry confidence.

Mitigation:
Force it to phrase output as guidance, not truth, and keep deterministic prediction as the source of the final score.

Risk:
Memory could bias the system too strongly toward past patterns.

Mitigation:
Treat memory as a soft prior and make the current-run evidence dominant.

## Success Criteria

The feature is complete when:

- the DES workflow can attach LLM chemistry rationales to proposals and top results
- the DES workflow can surface chemical warnings and caveats
- the report includes one safe next step and one broader chemistry alternative
- the advisor can reuse useful prior reasoning patterns without depending on them
- the feature is reusable enough to support future metal-binding and selectivity workflows
- tests cover the new advisor behavior
