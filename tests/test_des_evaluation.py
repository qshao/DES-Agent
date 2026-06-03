from des_multi_agent.evaluation import classify_des
from des_multi_agent.prediction import CurvePrediction
from des_multi_agent.schemas import DesThresholds


def test_des_classification_requires_both_thresholds():
    curve = CurvePrediction(
        smiles_a="CCO",
        smiles_b="O",
        ratios=[0.1, 0.5, 0.9],
        tm_pred_k=[250.0, 245.0, 255.0],
        t1_k=298.0,
        t2_k=273.0,
        checkpoint_path="ckpt.pt",
    )
    thresholds = DesThresholds(absolute_tm_max_k=260.0, relative_drop_min=0.10)
    result = classify_des(curve, thresholds)

    assert result.is_des is True
    assert result.absolute_pass is True
    assert result.relative_pass is True
