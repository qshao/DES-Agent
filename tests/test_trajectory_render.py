from des_multi_agent.trajectory import (
    CycleSnapshot,
    SearchTrajectory,
    TopEntry,
    format_trajectory_console,
    format_trajectory_report,
)


def _eg(metric=201.8, sec="Δ22.6%, high confidence"):
    return TopEntry("ethylene glycol (OCCO)", "min_tm_k", metric, sec)


def _traj():
    c1 = CycleSnapshot(
        cycle=1, n_screened=5, n_hits=5, top_entries=[_eg()],
        family_ledger={"diol": 1, "amide": 1},
    )
    c2 = CycleSnapshot(
        cycle=2, n_screened=5, n_hits=5, top_entries=[_eg()],
        new_entrants=["1,2-propanediol (CC(O)CO)"], dropouts=["water (O)"],
        family_ledger={"diol": 2},
    )
    c3 = CycleSnapshot(
        cycle=3, n_screened=5, n_hits=5, top_entries=[_eg()],
        converged=True, convergence_reason="top-5 shortlist identical to previous cycle",
    )
    return SearchTrajectory(
        workflow="des", headline="DES partners for ethanol (CCO)",
        metric_label="min Tm (K)", snapshots=[c1, c2, c3], total_cycles=3,
        converged=True, convergence_reason="top-5 shortlist identical to previous cycle",
        final_summary=[_eg()],
    )


def test_report_has_title_and_cycle_headings():
    md = format_trajectory_report(_traj())
    assert "# Search Trajectory — DES partners for ethanol (CCO)" in md
    assert "## Cycle 1 — 5 screened, 5 hits" in md
    assert "## Cycle 3 — 5 screened, 5 hits  ✓ converged" in md
    assert "Converged: yes" in md


def test_report_cycle1_omits_change_line_and_shows_families():
    md = format_trajectory_report(_traj())
    cycle1 = md.split("## Cycle 1")[1].split("## Cycle 2")[0]
    assert "Shortlist change" not in cycle1
    assert "Families reinforced: diol (2), amide (1)" not in cycle1  # cycle1 ledger order
    assert "Families reinforced:" in cycle1


def test_report_change_line_and_final_shortlist():
    md = format_trajectory_report(_traj())
    assert "+1 entered (1,2-propanediol (CC(O)CO))" in md
    assert "-1 left (water (O))" in md
    assert "## Final shortlist" in md
    assert "ethylene glycol (OCCO) — min Tm (K) 201.8" in md


def test_report_empty_snapshots():
    t = SearchTrajectory("des", "x", "min Tm (K)", [], 0, False, "", [])
    assert "_No cycles recorded._" in format_trajectory_report(t)


def test_console_one_block_per_cycle():
    txt = format_trajectory_console(_traj())
    assert "Trajectory — DES partners for ethanol (CCO)  (3 cycles, converged)" in txt
    lines = [ln for ln in txt.splitlines() if ln.strip().startswith("cycle ")]
    assert len(lines) == 3
    assert "converged" in lines[2]
