from des_multi_agent.trajectory import (
    CycleSnapshot,
    SearchTrajectory,
    TopEntry,
    shortlist_delta,
)


def test_top_entry_fields():
    e = TopEntry(label="ethylene glycol (OCCO)", metric_name="min_tm_k",
                 metric_value=201.8, secondary="Δ22.6%, high confidence")
    assert e.metric_name == "min_tm_k"
    assert e.metric_value == 201.8


def test_cycle_snapshot_defaults():
    s = CycleSnapshot(cycle=1, n_screened=5, n_hits=5, top_entries=[])
    assert s.new_entrants == [] and s.dropouts == []
    assert s.family_ledger == {}
    assert s.converged is False
    assert s.convergence_reason == "" and s.notable_warnings == []
    # frozen-default lists are independent instances
    s2 = CycleSnapshot(cycle=2, n_screened=1, n_hits=0, top_entries=[])
    assert s.new_entrants is not s2.new_entrants


def test_search_trajectory_fields():
    t = SearchTrajectory(workflow="des", headline="DES partners for ethanol (CCO)",
                         metric_label="min Tm (K)", snapshots=[], total_cycles=0,
                         converged=False, convergence_reason="", final_summary=[])
    assert t.workflow == "des"


def test_shortlist_delta_sorted_set_diff():
    new, dropped = shortlist_delta(["water (O)", "urea (NC(N)=O)"],
                                   ["urea (NC(N)=O)", "glycerol (OCC(O)CO)"])
    assert new == ["glycerol (OCC(O)CO)"]
    assert dropped == ["water (O)"]


def test_shortlist_delta_first_cycle_all_new():
    new, dropped = shortlist_delta([], ["a", "b"])
    assert new == ["a", "b"] and dropped == []
