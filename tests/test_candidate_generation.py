from des_multi_agent import candidate_generation
from des_multi_agent.candidate_generation import generate_candidates
from des_multi_agent.chemistry_filter import canonicalize_smiles, filter_candidates
from des_multi_agent.schemas import CandidateProposal
import pytest


def test_generation_and_filtering_returns_plausible_smiles():
    proposals = generate_candidates("CCO", n=5, constraints=None)
    filtered = filter_candidates("CCO", proposals)

    assert len(proposals) >= 5
    assert all(p.smiles for p in proposals)
    assert all(p.smiles != "CCO" for p in filtered)
    assert len({p.smiles for p in filtered}) == len(filtered)


def test_generate_candidates_raises_when_constraints_make_generation_impossible(monkeypatch):
    monkeypatch.setattr(
        candidate_generation,
        "_FAMILY_LIBRARY",
        (("alcohol", "hydrogen-bond donor", "O"),),
    )
    with pytest.raises(ValueError, match="Unable to generate"):
        generate_candidates("CCO", n=1, constraints={"allowed_families": ["nonexistent"]})


def test_filter_candidates_rejects_canonical_duplicates():
    proposals = [CandidateProposal(smiles="CCO", rationale="same molecule", family="alcohol")]
    filtered = filter_candidates("C(C)O", proposals)
    assert filtered == []
    assert canonicalize_smiles("C(C)O") == canonicalize_smiles("CCO")
