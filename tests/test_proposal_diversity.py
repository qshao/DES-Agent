from des_multi_agent.proposal_diversity import ProposalDiversityConfig, apply_proposal_diversity
from des_multi_agent.schemas import CandidateProposal


def test_apply_proposal_diversity_removes_exact_duplicates():
    proposals = [
        CandidateProposal(smiles="OCCO", rationale="polyol", family="polyol"),
        CandidateProposal(smiles="C(CO)O", rationale="same canonical molecule", family="polyol"),
    ]
    result = apply_proposal_diversity(
        "CCO",
        proposals,
        config=ProposalDiversityConfig(max_similarity=0.85),
    )
    assert [item.smiles for item in result.accepted] == ["OCCO"]
    assert any(item.smiles == "C(CO)O" for item in result.suppressed)


def test_apply_proposal_diversity_suppresses_near_duplicates():
    proposals = [
        CandidateProposal(smiles="OCCO", rationale="polyol", family="polyol"),
        CandidateProposal(smiles="OCCCO", rationale="near duplicate", family="polyol"),
    ]
    result = apply_proposal_diversity(
        "CCN",
        proposals,
        config=ProposalDiversityConfig(max_similarity=0.80, deduplicate_exact=True, deduplicate_near=True),
    )
    assert [item.smiles for item in result.accepted] == ["OCCO"]
    assert any(item.smiles == "OCCCO" for item in result.suppressed)


def test_apply_proposal_diversity_suggests_adjacent_families_when_des_budget_collapses():
    proposals = [
        CandidateProposal(smiles="OCCO", rationale="polyol", family="polyol"),
        CandidateProposal(smiles="C(CO)O", rationale="duplicate polyol", family="polyol"),
        CandidateProposal(smiles="CC(=O)N", rationale="amide", family="amide"),
    ]
    result = apply_proposal_diversity(
        "CCO",
        proposals,
        config=ProposalDiversityConfig(max_similarity=0.85, family_fallback=True, per_family_budget=1),
    )
    assert result.accepted
    assert result.suggested_families
    assert any(family in {"amide", "urea", "carboxylic acid"} for family in result.suggested_families)
