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
