from des_multi_agent.orchestrator import _grade_partner_reality


def test_grade_partitions_into_keep_demote_drop():
    # component_a = butane (no H-bond donors/acceptors) so that the branched
    # alkane gets label="none" → demote, and a Si-containing molecule fails the
    # structural-sanity gate → drop.  Urea is in the known set → keep.
    branched_alkane = "CCCC(CC)CCCC"  # 4-ethyloctane
    drop_candidate = "CCCC[Si](C)(C)C"  # silane — disallowed element
    llm_smiles = {"NC(N)=O", branched_alkane, drop_candidate, "CCO"}
    verdicts, penalties, drops = _grade_partner_reality(
        component_a="CCCC",          # butane — no H-bond capability
        candidate_smiles=["NC(N)=O", branched_alkane, drop_candidate, "CCO"],
        llm_smiles=llm_smiles,
    )
    # urea (known) and ethanol (known) → keep, no penalty, not dropped
    assert "NC(N)=O" not in penalties and "NC(N)=O" not in drops
    assert "CCO" not in penalties and "CCO" not in drops
    # branched alkane → no complementarity with butane (label=none) → demote
    assert penalties.get(branched_alkane, 0.0) == 0.25
    # silane → fails structural sanity (disallowed element) → drop
    assert drop_candidate in drops
    # one verdict per candidate
    assert len(verdicts) == 4


def test_grade_skips_non_llm_smiles():
    verdicts, penalties, drops = _grade_partner_reality(
        component_a="CCO",
        candidate_smiles=["CCCC[Si](C)(C)C"],  # would drop if graded
        llm_smiles=set(),                        # but it is not LLM-sourced
    )
    assert verdicts == [] and penalties == {} and drops == set()
