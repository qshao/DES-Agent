# Natural Language Normalization Design

## Goal

Add a normalization layer that improves how plain-language requests are turned into router jobs for DES and metal-binding workflows. The layer should resolve or clarify compound names, salts, free bases, and ambiguous phrasing before execution, while keeping the existing CLI/job schema unchanged.

## Scope

This first version focuses on router-facing normalization only:

- normalize compound names and common aliases
- detect likely salt/free-base ambiguities
- canonicalize workflow hints like "DES", "ligand binding", or "metal extraction"
- ask clarification questions when the request cannot be safely normalized
- preserve the current JSON job schema and execution flow

This version does not:

- change the DES prediction models
- change the metal-binding predictor
- add new workflow types
- add live chemistry database lookups

## Architecture

Add a lightweight normalization layer in front of the existing task router. The layer should produce normalized request hints that help the router fill the same CLI-style job schema more reliably.

The layer should:

- inspect the raw request text
- extract candidate compound names and workflow intent
- map obvious aliases to canonical names when safe
- flag ambiguous chemical forms for clarification
- pass the normalized request downstream to the existing router prompt

The router itself remains responsible for producing the final JSON job object. Normalization should only improve the quality of the inputs to the router and the questions it asks.

## Components

- `des_multi_agent/request_normalization.py`
  - parses the plain-language request
  - identifies chemical names, salts, and workflow hints
  - returns normalized hints and clarification flags

- `des_multi_agent/task_router_prompts.py`
  - updated to include normalized hints in the routing prompt
  - requests clarification when the normalizer marks a request ambiguous

- `des_multi_agent/task_router.py`
  - calls the normalization helper before prompting the LLM
  - preserves the existing `RouterResponse` JSON contract

- `des_multi_agent/task_executor.py`
  - uses the same normalized request path so execution matches routing behavior

- `tests/test_request_normalization.py`
  - unit tests for alias handling, salt/free-base ambiguity, and workflow hint extraction

- `tests/test_llm_parser.py`
  - router parsing tests for normalized requests and clarification flow

- `README.md`, `docs/tutorial.md`, `examples/README.md`
  - document how the router handles ambiguous names and when it asks for clarification

## Data Flow

1. The user enters a plain-language request.
2. The normalization layer inspects the text for compound names and workflow intent.
3. If the request is obviously resolvable, the normalizer returns canonical hints.
4. If the request contains an ambiguous salt/free-base or an unclear compound name, the normalizer flags it for clarification.
5. The router prompt receives the normalized hints and produces a JSON job or clarification questions.
6. `task-execute` uses the same path so execution and routing stay consistent.

## Error Handling

- If the request does not mention any chemical input, the router should ask for the missing compound or ligand name.
- If the request mentions both a salt form and a free-base form without enough context, the router should ask a clarification question instead of guessing.
- If the normalizer cannot recognize the compound name, it should leave the name unchanged and let the router ask for clarification.
- If a workflow hint is inconsistent with the text, the router should favor clarification over silent reinterpretation.

## Testing

- Unit tests for canonical alias mapping on common compound naming patterns
- Unit tests for salt/free-base ambiguity detection
- Unit tests for workflow hint extraction from phrases like:
  - "find DES partners"
  - "metal extraction ligands"
  - "predict stability constant"
- Router integration tests that confirm normalized hints still produce the existing JSON schema
- CLI tests that show the router asks for clarification rather than inventing a missing chemical form

## Success Criteria

The normalization layer is successful if:

- plain-language DES and metal-binding requests are interpreted more reliably
- salt/free-base ambiguity is surfaced as clarification instead of being guessed
- the router and task-execute commands keep the same JSON schema
- no existing workflows need to change their prediction or reporting logic
