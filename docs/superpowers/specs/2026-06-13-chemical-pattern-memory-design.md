# Chemical Pattern Memory Design

## Goal

Add a `ChemicalPatternMemory` layer that uses prior predictions and user feedback to make later DES screening cycles more chemically intuitive. The layer should summarize what prior cycles suggest chemically, then use those summaries to guide the next cycle's LLM proposals, chemistry-advisor explanations, and ranking bias.

The goal is not to replace the current predictive models. Current melting-point, viscosity, uncertainty, and selectivity predictions remain the primary evidence. Pattern memory is a bounded prior that helps the system search nearby chemical space more intelligently.

## Current Context

The project already has several pieces that can support this feature:

- `run_memory.py` stores prior ranked candidates and user `good` / `bad` labels.
- `multi_cycle.py` passes prior top results and an accumulated family ledger into later cycles.
- `orchestrator.py` already threads prior-family context into LLM brainstorming and run-memory notes into chemistry-advisor prompts.
- `proposal_diversity.py` prevents exact duplicates, near-duplicates, and family collapse.
- The chemistry-advisor layer can generate rationale, warnings, and next-step suggestions from structured context.

The missing piece is a structured memory layer that converts prior predictions into compact chemical lessons instead of passing only raw top hits or exact labels.

## Evidence Sources

The pattern memory layer should use both short-term and long-term evidence.

Short-term evidence comes from the active multi-cycle run:

- candidate family
- candidate SMILES
- DES pass/fail status
- minimum predicted melting temperature
- relative melting-point depression
- viscosity prediction when available
- uncertainty flag and trust score
- chemistry-advisor warning text when available

Long-term evidence comes from saved run memory when the user opts in with reuse:

- prior ranked candidates
- user `good` / `bad` labels
- prior source and source id
- prior component A compatibility
- prior top-ranked examples

User labels should carry more weight than unlabeled predictions. Repeated patterns across cycles should carry more weight than one-off hits.

## Outputs

`ChemicalPatternMemory` should produce a small structured object with:

- `productive_families`: families repeatedly associated with accepted DES candidates
- `avoid_families`: families repeatedly associated with high Tm, high viscosity, bad labels, or strong warnings
- `good_examples`: representative SMILES examples from good labels or strong predicted hits
- `bad_examples`: representative SMILES examples from bad labels or repeated failures
- `prompt_notes`: compact chemistry lessons suitable for LLM prompts
- `ranking_bias_by_smiles`: capped exact-candidate bonuses or penalties
- `ranking_bias_by_family`: capped family-level bonuses or penalties
- `confidence`: low, medium, or high memory confidence
- `notes`: user-visible notes explaining memory effects

Example prompt notes:

- `Prior cycles found short diols and small amides productive for ethanol.`
- `Triol-like candidates had acceptable Tm but tended to carry viscosity risk.`
- `User-labeled bad examples suggest avoiding bulky aromatic carboxylic acids for this component.`

## Data Flow

1. During each DES cycle, `orchestrator.py` collects the ranked results, candidate proposals, viscosity predictions, uncertainty annotations, and advisor notes.
2. `multi_cycle.py` passes the current accumulated pattern memory into the next `run_search_report` call.
3. If `--reuse-run` is provided, saved run memory is loaded and converted into long-term pattern evidence.
4. The LLM brainstorm context receives `prompt_notes`, plus a few representative good and bad examples.
5. Proposal diversity still enforces similarity caps and per-family budgets after generation.
6. Ranking receives bounded exact-candidate and family-level bias from the pattern memory.
7. Chemistry-advisor prompts receive the same compact pattern notes, so explanations can mention prior chemical lessons.
8. Reports include short notes describing which memory patterns were applied.

## Ranking Behavior

Memory should affect ranking only through capped bonuses and penalties.

Recommended initial caps:

- exact user good label: up to `+0.20`
- exact user bad label: down to `-0.20`
- repeated productive family: up to `+0.10`
- repeated avoided family: down to `-0.10`
- uncertainty penalty multiplier: reduce memory effect by 50% when prior evidence has low trust

The ranking adjustment should never make a clearly non-DES candidate outrank a strong DES candidate solely because memory liked its family. Memory can break ties, nudge uncertain cases, and steer exploration; it should not override model predictions.

## LLM Prompt Behavior

The LLM should receive pattern memory as a short chemistry lesson block, not as a full result table.

The prompt should ask the LLM to:

- reuse some productive motifs
- propose adjacent alternatives to productive motifs
- avoid repeating known failures
- preserve diversity according to the existing diversity mode
- explain when it intentionally explores outside prior productive families

The number of example structures should be limited, for example three good examples and three bad examples.

## Guardrails

- Do not hard-exclude a family unless the user explicitly excludes it or repeatedly labels it bad.
- Do not use memory from a different `component_a` unless the user explicitly asks for cross-component reuse.
- Reduce memory influence when prior evidence is sparse, uncertain, or contradictory.
- Keep prompt notes short enough to avoid drowning out the current task.
- Record memory influence in run notes so users can inspect why a candidate was favored or penalized.
- If memory parsing fails, continue the run without pattern memory and emit a warning.

## User Controls

Initial controls should be minimal:

- `--chemical-pattern-memory off|soft|adaptive`
- `--pattern-memory-max-examples N`

Default should be `adaptive` when `--n-cycles > 1` or `--reuse-run` is used, and effectively inactive when no prior evidence exists.

Future controls can expose family-level thresholds, but the first version should avoid too many knobs.

## Testing

Tests should cover:

- pattern extraction from ranked DES results
- stronger weighting for user good/bad labels than unlabeled prior ranks
- reduced influence when uncertainty is low-trust
- no cross-component reuse by default
- prompt-note generation with bounded length
- ranking bias caps
- multi-cycle handoff from one cycle to the next
- saved run memory reuse through `--reuse-run`
- graceful behavior when no memory exists

## Non-Goals

- Do not train or fine-tune a model from prior runs.
- Do not add autonomous experiment planning in this feature.
- Do not make LLM advice override deterministic prediction outputs.
- Do not require internet access or external chemistry databases.

## Open Implementation Notes

The likely implementation module is `des_multi_agent/chemical_pattern_memory.py`. It should stay independent from `run_memory.py`: run memory stores prior run records, while chemical pattern memory interprets those records into chemical lessons and ranking priors.

The first implementation should focus on DES workflows. Metal-binding and selectivity workflows can reuse the same pattern-memory abstraction later, but they may need separate family semantics and scoring rules.
