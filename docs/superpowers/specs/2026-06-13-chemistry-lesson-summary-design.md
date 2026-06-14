# Chemistry Lesson Summary Design

## Goal

Add a `ChemistryLessonSummary` layer that turns prior DES predictions into short chemistry lessons for the next cycle and the final report. The summary should explain what the previous evidence suggests chemically, what to avoid, and what to try next, using the same evidence already available in the workflow.

The goal is not to replace ranking, proposal generation, or the existing chemistry-advisor layer. Instead, the lesson summary should give those components a compact, chemically grounded memory of what the run has learned so far.

## Current Context

The project already has the core pieces needed for this feature:

- `ChemicalPatternMemory` turns prior predictions and labels into compact lessons and bounded ranking bias.
- `run_memory.py` stores saved run records and user `good` / `bad` labels.
- `orchestrator.py` already builds chemistry-advisor prompts, run-memory notes, and LLM brainstorm context.
- `multi_cycle.py` carries evidence forward between cycles.
- `reporting.py` formats the final human-readable run report.

The missing piece is a dedicated lesson-summary object that makes the chemistry signal explicit and readable in both the report and the next cycle context.

## Evidence Sources

The lesson summary should be derived from the evidence the system already has, with user labels weighted most strongly:

- DES-positive and DES-negative results from the current cycle
- candidate family labels from the brainstorm stage
- melting-point trend information when available
- viscosity predictions when available
- uncertainty annotations and trust scores when available
- chemistry-advisor warnings and rationales when available
- saved run memory labels (`good` / `bad`) when the user reuses a run
- prior pattern-memory notes when a previous cycle already extracted lessons

The summary should prefer repeated patterns over single observations. A lone candidate should not create a strong lesson unless the user labeled it or the evidence repeats across cycles.

## Outputs

`ChemistryLessonSummary` should provide a small structured object with two scopes:

- `cycle_summary`: short notes for the current cycle
- `run_summary`: a roll-up across all cycles so far

Each summary should include:

- `productive_patterns`: families or motifs that look promising
- `avoid_patterns`: families, motifs, or properties that repeatedly look poor
- `next_steps`: one or two concrete follow-up suggestions
- `warnings`: brief notes about likely failure modes or uncertainty
- `representative_examples`: a small bounded set of good and bad examples
- `confidence`: low, medium, or high
- `notes`: user-visible text that can be shown in the report and reused by prompts

Example notes:

- `Short diols and small amides were repeatedly productive for this component.`
- `Bulky aromatic acids and high-viscosity triols kept appearing in the avoid set.`
- `A safe next step is to stay near the productive families; an exploratory next step is to try adjacent aliphatic donors.`

## Data Flow

1. During a cycle, the orchestrator collects annotated results, candidate proposals, chemistry-advisor notes, and any reused run memory.
2. A lesson summary is built from that evidence after ranking is complete.
3. The summary is attached to the search outcome so the report can display it.
4. The same summary is fed into the next cycle's LLM brainstorming and chemistry-advisor prompts.
5. If `--reuse-run` is active, saved memory contributes to the run summary so later cycles can reuse lessons from earlier runs.
6. Multi-cycle runs should carry the most recent lesson summary forward so later cycles can summarize what changed since the previous cycle.

## Report Behavior

The final report should show the lesson summary as a compact block, not as a new section that duplicates the full candidate table.

The report should make clear:

- what chemical patterns looked productive
- what chemical patterns should be avoided
- which next action is conservative versus exploratory
- whether the lesson is based on strong or weak evidence

The report should keep the language short and chemistry-facing. It should not restate every candidate. Instead, it should highlight the chemical lesson from the run.

## Prompt Behavior

The LLM should receive the lesson summary as a short chemistry note, not a long history dump.

The prompt should ask the LLM to:

- reuse productive motifs when appropriate
- avoid repeating the most obvious failures
- suggest one safe next step and one broader chemistry alternative
- keep the advice aligned with the current diversity mode and pattern memory

The number of representative examples should be small and bounded so prompts stay stable across runs.

## Guardrails

- Do not override deterministic ranking with the lesson summary.
- Do not invent a lesson when the evidence is weak; say that the pattern is tentative instead.
- Do not let the summary grow into a second full report.
- Do not reuse lessons across different component-A inputs unless the user explicitly opts in.
- Do not store the entire candidate table in the summary; only keep compact lessons and a few examples.
- If the lesson summary cannot be built, continue the run and omit the summary with a warning.

## User Controls

The first version should be mostly automatic.

Recommended behavior:

- cycle summaries are always produced when there is enough evidence
- final run summaries are always produced for multi-cycle or reuse-enabled runs
- the user does not need a new CLI flag for the first version
- reuse behavior follows the existing `--reuse-run` and multi-cycle settings

If a later version needs more control, the system can expose a toggle for lesson-summary verbosity, but the initial design should stay simple.

## Testing

Tests should cover:

- cycle summary generation from annotated results
- run summary roll-up across multiple cycles
- inclusion of saved good/bad labels in the lesson text
- bounded representative examples
- conservative behavior when evidence is sparse or contradictory
- prompt note generation for the advisor and brainstorm context
- report formatting that shows the lesson summary without duplicating the full table
- graceful fallback when no usable lesson can be built

## Non-Goals

- Do not create a new autonomous planner.
- Do not add internet-backed chemistry knowledge.
- Do not train a model on lessons.
- Do not replace the existing chemistry-advisor or pattern-memory layers.
- Do not add a separate manual annotation workflow for lessons.

## Open Implementation Notes

The likely implementation shape is a new helper in `des_multi_agent/chemical_lesson_summary.py` or a closely related module that can consume the existing pattern-memory object and annotated results.

The first implementation should focus on DES runs, where the lesson summary can reuse the current cycle/ranking structure. If the design works well there, the same abstraction can later be reused for metal-binding and selectivity workflows.
