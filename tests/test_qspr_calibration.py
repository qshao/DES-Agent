"""QSPR conformal-calibration plumbing (A1): the model carries the calibrated
std scale + conformal multiplier and reports a calibrated interval."""
from __future__ import annotations

from types import SimpleNamespace

import torch

from des_multi_agent.predictors.melting_point import MeltingPointQSPR


class _Member:
    def __init__(self, value):
        self._v = value

    def __call__(self, x):
        return torch.tensor([self._v])


def test_predict_reports_conformal_interval_and_scale():
    embedder = SimpleNamespace(embed=lambda s: [[0.0, 0.0]])
    model = MeltingPointQSPR(
        members=[_Member(0.4), _Member(0.6)],  # spread -> nonzero std
        feat_mean=torch.zeros(2), feat_std=torch.ones(2),
        tm_mean=300.0, tm_std=50.0,
        embedder=embedder, device=torch.device("cpu"),
        std_scale_k=30.0, conformal_q=6.0,
    )
    p = model.predict("CCO")
    assert model.std_scale_k == 30.0
    assert p.std_k > 0
    assert p.ci_k == 6.0 * p.std_k  # calibrated half-width = q * ensemble std


def test_missing_calibration_leaves_interval_none():
    embedder = SimpleNamespace(embed=lambda s: [[0.0, 0.0]])
    model = MeltingPointQSPR(
        members=[_Member(0.4), _Member(0.6)],
        feat_mean=torch.zeros(2), feat_std=torch.ones(2),
        tm_mean=300.0, tm_std=50.0,
        embedder=embedder, device=torch.device("cpu"),
    )
    p = model.predict("CCO")
    assert model.std_scale_k is None
    assert p.ci_k is None
