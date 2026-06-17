# DES-Agent Architecture

This diagram summarizes the main user entry points, workflow agents, predictive models, memory tools, and output artifacts in DES-Agent.

![DES-Agent multi-agent architecture](assets/des-agent-architecture.png)

The system starts from plain-language requests, SMILES inputs, metal ions, or local files. The CLI and task layer route those inputs into DES screening, viscosity-aware ranking, metal-binding, metal-selectivity, or selectivity-DES workflows. Optional LLM roles assist with brainstorming, candidate review, explanations, and contradiction checks. Local model artifacts and run-memory files keep the system usable offline and reproducible across runs.

## New Modules (2026)

**`des_multi_agent/chemistry/partner_registry.py`**
Builds the known-compound registry (curated `common_names.json` ∪ auto-role-tagged `experimental.json`) and exposes `is_known()`, `structural_sanity()`, and `known_partner_menu(role, limit)` for reality-anchored partner proposals.

**`des_multi_agent/chemistry/claim_grounding.py`**
The single LLM-agnostic surface for chemistry grounding. Source-side: `structural_facts(smiles).as_prompt_block()` injects computed H-bond and coordination facts into prompts. Output-side: `GroundingVerdict` (verified/contradicted/unverifiable) from four checkers (family, DES plausibility, coordination, selectivity) and `PartnerVerdict` (known/novel_plausible/novel_implausible) from `ground_partner_reality`.

**`des_multi_agent/chemistry/name_resolution.py`**
Resolves common molecule names to canonical SMILES via `artifacts/molecule_names/common_names.json`. Used by all CLI entry points; falls back to treating the input as SMILES if no name match.
