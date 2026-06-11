"""
Screen ligands for Ni2+ selectivity over Co2+, requiring HBD and HBA capability.

Three-cycle run: cycle 1 uses the heuristic library; cycles 2 and 3 use LLM
brainstorming when an llm_config path is provided as the first CLI argument.
Without an LLM the loop exits early after the heuristic candidates are exhausted.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running from the repo root without installing the package.
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from des_multi_agent.reporting import format_metal_selectivity_report
from des_multi_agent.workflows.metal_binding_selectivity import run_metal_selectivity_screen

# Constraints: require at least one HBD and one HBA group.
# These are forwarded to the LLM brainstorm prompt so the model focuses on
# ligands with -OH, -NH, -COOH, or similar donor/acceptor functionality.
CONSTRAINTS = {
    "min_hbd": 1,
    "min_hba": 1,
}

llm_provider = None
if len(sys.argv) > 1:
    llm_config_path = sys.argv[1]
    from des_multi_agent.llm.factory import build_llm_provider
    from des_multi_agent.llm.config import load_llm_config
    llm_cfg = load_llm_config(llm_config_path)
    llm_provider = build_llm_provider(llm_cfg)

outcome = run_metal_selectivity_screen(
    target_metal="Ni2+",
    competitor_metal="Co2+",
    n=20,
    model_path=str(_REPO / "artifacts" / "stability_constants" / "model.json"),
    llm_provider=llm_provider,
    constraints=CONSTRAINTS,
    n_cycles=3,
    w_affinity=0.5,
    w_selectivity=0.5,
)

print(format_metal_selectivity_report(outcome))
