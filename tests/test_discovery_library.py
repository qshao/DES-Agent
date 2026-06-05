from pathlib import Path

from des_multi_agent.discovery import DiscoveryLibrary, load_discovery_library
from des_multi_agent.schemas import CandidateProposal


def test_load_discovery_library_parses_literature_and_library():
    fixtures = Path(__file__).parent / "fixtures" / "discovery"
    library = load_discovery_library(fixtures)

    assert isinstance(library, DiscoveryLibrary)
    assert len(library.literature) == 1
    assert len(library.candidate_library) == 3
    assert library.literature[0].component_b == "OCCO"
    assert library.candidate_library[0].smiles == "O"


def test_candidate_proposal_carries_provenance_fields():
    proposal = CandidateProposal(
        smiles="OCCO",
        rationale="Known local hit",
        family="literature",
        source="literature",
        source_id="LIT-001",
        similarity_score=0.91,
        reference_note="Curated local record",
    )

    assert proposal.source == "literature"
    assert proposal.source_id == "LIT-001"
    assert proposal.similarity_score == 0.91
    assert proposal.reference_note == "Curated local record"
