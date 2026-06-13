"""R2: flag physically implausible eutectics (predicted Tm above both pure
components) in classification and reporting."""
from __future__ import annotations

from des_multi_agent.evaluation import DesResult, classify_des
from des_multi_agent.prediction import CurvePrediction
from des_multi_agent.schemas import DesThresholds
from des_multi_agent import reporting


def _curve(tm_pred, t1, t2):
    return CurvePrediction(
        smiles_a="CCO", smiles_b="O",
        ratios=[0.25, 0.5, 0.75], tm_pred_k=tm_pred,
        t1_k=t1, t2_k=t2, checkpoint_path="ckpt.pt",
    )


_THRESH = DesThresholds(absolute_tm_max_k=400.0, relative_drop_min=0.0)


def test_classify_flags_physical_eutectic():
    # min predicted Tm (240) is below both pure components (260, 290) -> physical
    res = classify_des(_curve([250.0, 240.0, 255.0], 260.0, 290.0), _THRESH)
    assert res.eutectic_physical is True


def test_classify_flags_nonphysical_eutectic():
    # min predicted Tm (305) exceeds both pure components (260, 290) -> impossible
    res = classify_des(_curve([320.0, 305.0, 330.0], 260.0, 290.0), _THRESH)
    assert res.eutectic_physical is False


def test_report_warns_on_nonphysical_eutectic():
    res = classify_des(_curve([320.0, 305.0, 330.0], 260.0, 290.0), _THRESH)
    text = reporting.format_report([res], resolve_names=False)
    assert "non-physical" in text.lower()


def test_report_no_warning_when_all_physical():
    res = classify_des(_curve([250.0, 240.0, 255.0], 260.0, 290.0), _THRESH)
    text = reporting.format_report([res], resolve_names=False)
    assert "non-physical" not in text.lower()
