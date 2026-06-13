from __future__ import annotations

from dataclasses import dataclass
from statistics import pstdev

import pytest

from des_multi_agent.prediction import CurvePrediction
from des_multi_agent.property_resolution import MeltingPointEstimate
from des_multi_agent.uncertainty import model


@dataclass(frozen=True)
class _FakeCurve:
    tm_pred_k: list[float]


def test_estimate_min_tm_uncertainty_runs_three_predictions(monkeypatch):
    calls: list[tuple[str, str, float, float, str, str]] = []

    monkeypatch.setattr(
        model,
        "resolve_melting_point",
        lambda component, override_k=None: MeltingPointEstimate(
            component=component,
            tm_k=300.0 if component == "CCO" else 275.0,
            source="heuristic",
            confidence=0.5,
        ),
    )

    def fake_predict_curve(component_a, component_b, t1_k, t2_k, checkpoint_path, config_path="ml_des_mp/config.yaml", mc_dropout=False):
        assert mc_dropout is True  # uncertainty uses MC-dropout for real spread
        calls.append((component_a, component_b, t1_k, t2_k, str(checkpoint_path), str(config_path)))
        curves = [
            _FakeCurve(tm_pred_k=[300.0, 240.0, 310.0]),
            _FakeCurve(tm_pred_k=[305.0, 260.0, 315.0]),
            _FakeCurve(tm_pred_k=[290.0, 250.0, 300.0]),
        ]
        return curves[len(calls) - 1]

    monkeypatch.setattr(model, "predict_curve", fake_predict_curve)

    result = model.estimate_min_tm_uncertainty(
        component_a="CCO",
        component_b="O",
        checkpoint_path="ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt",
        config_path="ml_des_mp/config.yaml",
    )

    assert len(calls) == 3
    assert calls == [
        ("CCO", "O", 300.0, 275.0, "ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt", "ml_des_mp/config.yaml"),
        ("CCO", "O", 300.0, 275.0, "ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt", "ml_des_mp/config.yaml"),
        ("CCO", "O", 300.0, 275.0, "ml_des_mp/runs/chemberta_random_row_fold01of05_best.pt", "ml_des_mp/config.yaml"),
    ]
    assert result.repeated_values == (240.0, 260.0, 250.0)
    assert result.min_tm_k == 240.0
    assert result.max_tm_k == 260.0
    assert result.mean_tm_k == pytest.approx(250.0)
    assert result.std_tm_k == pytest.approx(pstdev([240.0, 260.0, 250.0]))
    assert 0.0 <= result.trust_score <= 1.0
    assert result.uncertainty_flag in {"low", "medium", "high"}
    assert result.explanation


def _run_with_input_confidence(monkeypatch, confidence: float):
    monkeypatch.setattr(
        model,
        "resolve_melting_point",
        lambda component, override_k=None: MeltingPointEstimate(
            component=component, tm_k=300.0, source="heuristic", confidence=confidence,
        ),
    )

    def fake_predict_curve(component_a, component_b, t1_k, t2_k, checkpoint_path, config_path="ml_des_mp/config.yaml", mc_dropout=False):
        return _FakeCurve(tm_pred_k=[300.0, 250.0, 310.0])

    monkeypatch.setattr(model, "predict_curve", fake_predict_curve)
    return model.estimate_min_tm_uncertainty(
        component_a="CCO", component_b="O",
        checkpoint_path="ckpt.pt", config_path="cfg.yaml",
    )


def test_trust_score_reflects_input_melting_point_confidence(monkeypatch):
    # identical prediction spread; only the input Tm confidence differs
    high = _run_with_input_confidence(monkeypatch, 0.95)
    low = _run_with_input_confidence(monkeypatch, 0.35)
    assert high.std_tm_k == pytest.approx(low.std_tm_k)
    assert high.trust_score > low.trust_score
    assert 0.0 <= low.trust_score <= 1.0
