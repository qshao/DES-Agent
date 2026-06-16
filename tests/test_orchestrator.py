from des_multi_agent import orchestrator
from des_multi_agent.prediction import CurvePrediction
from des_multi_agent.property_resolution import MeltingPointEstimate


def test_orchestrator_returns_ranked_results(monkeypatch, tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        """
device: cpu
embedding:
  method: morgan
  morgan:
    radius: 2
    n_bits: 16
    use_chirality: false
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        orchestrator,
        "resolve_melting_point",
        lambda comp, override_k=None: MeltingPointEstimate(component=comp, tm_k=298.15, source="heuristic", confidence=0.5),
    )

    def fake_predict_curve(component_a, component_b, t1_k, t2_k, checkpoint_path, config_path="ml_des_mp/config.yaml"):
        return CurvePrediction(
            smiles_a=component_a,
            smiles_b=component_b,
            ratios=[0.1, 0.5, 0.9],
            tm_pred_k=[250.0, 245.0, 255.0],
            t1_k=t1_k,
            t2_k=t2_k,
            checkpoint_path=checkpoint_path,
        )

    monkeypatch.setattr(orchestrator, "predict_curve", fake_predict_curve)
    results = orchestrator.run_search(
        component_a="CCO",
        n=3,
        checkpoint_path="ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt",
        config_path=str(cfg_path),
    )
    assert len(results) > 0
    assert all(hasattr(result, "rationale") for result in results)
    # the resolved melting-point provenance must be threaded onto each result
    top = results[0]
    assert top.t1_source == "heuristic"
    assert top.t2_source == "heuristic"
    assert top.t1_confidence == 0.5
    assert top.t2_confidence == 0.5


def test_candidates_file_accepts_molecule_names(tmp_path):
    """_load_candidates_file should resolve molecule names to SMILES."""
    from des_multi_agent.orchestrator import _load_candidates_file

    f = tmp_path / "candidates.txt"
    f.write_text("urea\ncholine chloride\n# this is a comment\nNC(N)=O\n")
    proposals = _load_candidates_file(str(f))
    smiles_list = [p.smiles for p in proposals]
    # urea → NC(N)=O; choline chloride → its SMILES; comment skipped; NC(N)=O passes through
    assert "NC(N)=O" in smiles_list
    # choline chloride should be resolved — check it's a valid SMILES
    from rdkit import Chem
    for s in smiles_list:
        assert Chem.MolFromSmiles(s) is not None, f"Invalid SMILES in output: {s!r}"


def test_candidates_file_skips_unknown_names_with_warning(tmp_path, capsys):
    """Unknown names in candidates file print a warning and are skipped."""
    from des_multi_agent.orchestrator import _load_candidates_file

    f = tmp_path / "candidates.txt"
    f.write_text("urea\nnot_a_real_molecule_xyz\n")
    proposals = _load_candidates_file(str(f))
    smiles_list = [p.smiles for p in proposals]
    assert len(smiles_list) == 1  # only urea
    captured = capsys.readouterr()
    assert "not_a_real_molecule_xyz" in captured.err


def test_run_search_report_collects_grounding_verdicts(monkeypatch, tmp_path):
    """claim_verdicts is populated on SearchOutcome even when LLM is disabled."""
    from des_multi_agent.chemistry.claim_grounding import GroundingVerdict
    from des_multi_agent.orchestrator import SearchOutcome

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        """
device: cpu
embedding:
  method: morgan
  morgan:
    radius: 2
    n_bits: 16
    use_chirality: false
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        orchestrator,
        "resolve_melting_point",
        lambda comp, override_k=None: MeltingPointEstimate(
            component=comp, tm_k=298.15, source="heuristic", confidence=0.5
        ),
    )

    def fake_predict_curve(component_a, component_b, t1_k, t2_k, checkpoint_path, config_path="ml_des_mp/config.yaml"):
        return CurvePrediction(
            smiles_a=component_a,
            smiles_b=component_b,
            ratios=[0.1, 0.5, 0.9],
            tm_pred_k=[250.0, 245.0, 255.0],
            t1_k=t1_k,
            t2_k=t2_k,
            checkpoint_path=checkpoint_path,
        )

    monkeypatch.setattr(orchestrator, "predict_curve", fake_predict_curve)

    outcome = orchestrator.run_search_report(
        component_a="CCO",
        n=3,
        checkpoint_path="ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt",
        config_path=str(cfg_path),
    )
    assert isinstance(outcome, SearchOutcome)
    assert isinstance(outcome.claim_verdicts, list)
    assert all(isinstance(v, GroundingVerdict) for v in outcome.claim_verdicts)


def test_search_outcome_claim_verdicts_defaults_to_empty_list():
    """SearchOutcome.claim_verdicts has a backward-compatible default."""
    from des_multi_agent.orchestrator import SearchOutcome
    from des_multi_agent.evaluation import DesResult
    from des_multi_agent.prediction import CurvePrediction

    # Build minimal SearchOutcome without supplying claim_verdicts
    curve = CurvePrediction(
        smiles_a="CCO",
        smiles_b="NC(N)=O",
        ratios=[0.5],
        tm_pred_k=[240.0],
        t1_k=298.15,
        t2_k=406.0,
        checkpoint_path="x",
    )
    result = DesResult(curve=curve, min_tm_k=240.0, is_des=True, rationale="test", absolute_pass=True, relative_pass=True)
    outcome = SearchOutcome(
        results=[result],
        annotated_results=[],
        candidate_proposals=[],
        candidate_reviews=[],
        brainstorm_candidates=[],
        explanation_notes=[],
        critique_notes=[],
        llm_warnings=[],
    )
    # claim_verdicts must exist and default to empty list
    assert outcome.claim_verdicts == []
