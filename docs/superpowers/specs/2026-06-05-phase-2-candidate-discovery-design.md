# Phase 2 Candidate Discovery Design

**Goal:** Add a local candidate-discovery layer to the DES multi-agent system so the user can ask for component `A` and receive better candidate partners before prediction. Phase 2 focuses on local retrieval only: literature-like lookup from curated files and similarity-based retrieval from local candidate libraries. The existing heuristic generator remains the fallback when local discovery is empty or incomplete.

**Architecture:** Phase 2 inserts a deterministic discovery layer ahead of the current candidate generator. The discovery layer loads local reference data, retrieves known or similar partners, normalizes and deduplicates them, and returns candidates with provenance metadata. The orchestrator merges those results with the existing rule-based generator, then continues through the current filter, property resolution, prediction, uncertainty, and ranking pipeline unchanged.

**Tech Stack:** Python, the existing `des_multi_agent` orchestration code, local data files in JSON/CSV/YAML or similar tabular formats, RDKit for optional similarity computation, and `pytest` for verification.

---

## Scope

This design is limited to local candidate discovery.

- Input: component `A` as SMILES, requested candidate count `n`, and optional local source paths.
- Output: candidate partners with provenance metadata indicating whether they came from literature-like lookup, similarity search, or the heuristic fallback generator.
- In scope: local literature lookup, local similarity search, deterministic merge/deduplication, provenance reporting, and fallback to the existing generator when discovery sources are absent or empty.
- Out of scope: live external databases, web scraping, availability screening, safety screening, prediction changes, uncertainty changes, active learning, or any change to DES classification.

## Existing Assets

The repository already has the deterministic downstream pipeline:

- `des_multi_agent/candidate_generation.py` creates rule-based candidate partners.
- `des_multi_agent/chemistry_filter.py` removes invalid or duplicate candidates.
- `des_multi_agent/property_resolution.py` resolves neat-component melting points.
- `des_multi_agent/prediction.py` predicts the melting curve.
- `des_multi_agent/evaluation.py` classifies DES behavior.
- `des_multi_agent/orchestrator.py` coordinates the end-to-end search.
- `des_multi_agent/reporting.py` formats the output.

Phase 2 should extend these assets rather than replace them.

## Design Principles

- Keep discovery deterministic and local.
- Prefer explicit provenance over opaque ranking.
- Fall back to the existing heuristic generator if local sources are empty, malformed, or unavailable.
- Normalize molecules before deduplication so equivalent representations collapse to one candidate.
- Preserve the downstream prediction and uncertainty pipeline unchanged.

## System Roles

### 1. Local Library Loader

Purpose: load and normalize local discovery sources.

Behavior:

- Reads curated files containing local DES references and local candidate libraries.
- Normalizes molecule strings and any metadata fields required for ranking or provenance.
- Rejects malformed records at load time with a clear error.
- Provides a lightweight in-memory view so retrieval can be deterministic and testable.

Recommended source shape:

- Literature-like references: `component_a`, `component_b`, `source`, `note`, `reference_id`
- Similarity library entries: `smiles`, `family`, `source`, `note`

### 2. Literature Lookup

Purpose: return known or near-known partner candidates from curated local references.

Behavior:

- Searches the local reference set for records matching the input `component_a` or a canonicalized equivalent.
- Returns candidate partners that are explicitly documented in the local reference file.
- Attaches provenance fields describing the source record and why the candidate was selected.
- Allows exact matches and close analogs, but keeps the matching logic deterministic.

### 3. Similarity Search

Purpose: retrieve structurally similar candidate partners from the local candidate library.

Behavior:

- Scores each library entry against `component_a` using a deterministic similarity metric.
- Ranks candidates by similarity score and optionally by simple family heuristics.
- Returns a bounded number of top candidates so the result set remains manageable.
- Attaches the similarity score and a short rationale to each candidate.

Recommended similarity approach:

- Use RDKit fingerprints if available in the environment.
- Keep a simple fallback metric or fingerprint-free approximation if RDKit is unavailable, but prefer RDKit when possible.

### 4. Candidate Merger

Purpose: combine local discovery results with the existing heuristic generator.

Behavior:

- Merges literature lookup, similarity search, and heuristic generator outputs.
- Canonicalizes candidates before deduplication so the same compound is not repeated under different SMILES strings.
- Preserves the provenance of the first source that introduced a candidate.
- Produces at most the requested `n` candidates, unless the local sources and generator collectively cannot supply that many.

### 5. Orchestrator Integration

Purpose: use local discovery before the prediction pipeline.

Behavior:

- Calls local literature lookup and similarity search first.
- Falls back to the existing heuristic generator if the discovery layer returns too few candidates.
- Passes the merged candidate set through the existing chemistry filter, property resolution, prediction, uncertainty, and ranking steps.
- Exposes provenance metadata in the final report so users can tell why a candidate appeared.

## Data Model

The discovery layer should return structured candidate records with these core fields:

- `smiles`: canonical or canonicalizable candidate SMILES
- `family`: coarse chemical family or category
- `source`: `literature`, `similarity`, or `heuristic`
- `source_id`: local record identifier or library key
- `rationale`: short human-readable explanation
- `similarity_score`: optional numeric score for similarity hits
- `reference_note`: optional free-text note for literature hits

The orchestrator should be able to merge these into the existing candidate proposal records without losing provenance.

## Data Flow

1. The orchestrator receives component `A` and the requested count `n`.
2. The local library loader reads curated discovery files.
3. `literature_lookup()` extracts known DES-like partners or close analogs from the reference file.
4. `similarity_search()` ranks local library entries against `component_a`.
5. The merger combines discovery outputs with heuristic proposals and removes duplicates.
6. The chemistry filter removes invalid or self-matching candidates.
7. The existing property resolution, prediction, uncertainty, and ranking pipeline runs unchanged.
8. The report shows candidate provenance alongside the usual DES summary.

## Interfaces

The first implementation should expose a small internal API:

- `load_discovery_library(path: str) -> DiscoveryLibrary`
- `literature_lookup(component_a: str, library: DiscoveryLibrary) -> list[CandidateProposal]`
- `similarity_search(component_a: str, library: DiscoveryLibrary, limit: int) -> list[CandidateProposal]`
- `merge_discovery_candidates(*candidate_groups) -> list[CandidateProposal]`

The returned candidate proposal objects should include provenance fields so the orchestrator and reporter can surface them without recomputing anything.

## Error Handling

- If a local discovery file is missing, malformed, or empty, the orchestrator should log or surface a warning and fall back to the existing heuristic generator.
- If similarity computation cannot run, the discovery layer should still return literature hits and/or heuristic candidates.
- If a record is duplicated across local sources, the merger should keep one canonical candidate and preserve the best provenance available.
- If discovery returns fewer than `n` candidates, the orchestrator should continue with the available set rather than fail the run.

## Testing Strategy

The first implementation should be covered by focused tests.

- Library loader tests should verify that local records are parsed and normalized correctly.
- Literature lookup tests should verify that known pairs are returned with provenance.
- Similarity search tests should verify ranking, limit handling, and deterministic ordering.
- Merger tests should verify deduplication, source precedence, and fallback behavior.
- Orchestrator tests should verify that discovery results flow into the existing prediction pipeline and that provenance appears in the report.
- Regression tests should cover malformed discovery files and empty discovery sources.

## Implementation Boundaries

- Phase 2 should add local discovery only.
- Phase 2 should not change the predictor, uncertainty, or DES classification logic.
- Phase 2 should not require live network access.
- Phase 2 should not add availability or safety screening yet; those remain future phases.

## Risks and Mitigations

- Risk: local sources may be sparse.
  - Mitigation: preserve the existing heuristic generator as a fallback.
- Risk: duplicate candidates may appear from multiple sources.
  - Mitigation: canonicalize before deduplication and keep provenance metadata.
- Risk: similarity scores could be hard to reproduce across environments.
  - Mitigation: prefer deterministic fingerprints and fixed ordering.
- Risk: provenance can clutter the report.
  - Mitigation: keep provenance compact and attach only the minimum necessary metadata.
