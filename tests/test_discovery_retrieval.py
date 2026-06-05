from pathlib import Path

from des_multi_agent.discovery import literature_lookup, load_discovery_library, merge_discovery_candidates, similarity_search


def test_discovery_returns_literature_and_similarity_hits():
    library = load_discovery_library(Path(__file__).parent / "fixtures" / "discovery")
    literature = literature_lookup("CCO", library)
    similar = similarity_search("CCO", library, limit=2)
    merged = merge_discovery_candidates(literature, similar)

    assert literature
    assert similar
    assert merged
    assert {candidate.source for candidate in merged} <= {"literature", "similarity"}
    assert merged[0].smiles


def test_merge_discovery_candidates_deduplicates_canonical_smiles():
    library = load_discovery_library(Path(__file__).parent / "fixtures" / "discovery")
    literature = literature_lookup("CCO", library)
    similar = similarity_search("CCO", library, limit=3)
    merged = merge_discovery_candidates(literature, similar)

    canonical = {candidate.smiles for candidate in merged}
    assert len(canonical) == len(merged)
