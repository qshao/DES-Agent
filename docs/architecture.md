# DES-Agent Architecture

This diagram summarizes the main user entry points, workflow agents, predictive models, memory tools, and output artifacts in DES-Agent.

![DES-Agent multi-agent architecture](assets/des-agent-architecture.png)

The system starts from plain-language requests, SMILES inputs, metal ions, or local files. The CLI and task layer route those inputs into DES screening, viscosity-aware ranking, metal-binding, metal-selectivity, or selectivity-DES workflows. Optional LLM roles assist with brainstorming, candidate review, explanations, and contradiction checks. Local model artifacts and run-memory files keep the system usable offline and reproducible across runs.
