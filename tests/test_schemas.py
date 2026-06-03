from des_multi_agent.schemas import CandidateProposal, DesThresholds, MeltingPointEstimate


def test_thresholds_and_candidate_schema_round_trip():
    thresholds = DesThresholds(absolute_tm_max_k=300.0, relative_drop_min=0.25)
    proposal = CandidateProposal(
        smiles="O",
        rationale="small hydrogen-bond donor",
        family="alcohol",
    )
    estimate = MeltingPointEstimate(component="O", tm_k=273.15, source="heuristic", confidence=0.5)

    assert thresholds.absolute_tm_max_k == 300.0
    assert thresholds.relative_drop_min == 0.25
    assert proposal.smiles == "O"
    assert proposal.family == "alcohol"
    assert estimate.tm_k == 273.15
