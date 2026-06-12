"""The DES report and JSON export surface pure-component melting-point
provenance (experimental / qspr / heuristic) when results carry it, and stay
unchanged when they do not."""
from __future__ import annotations

import json

from dataclasses import replace

from des_multi_agent.evaluation import DesResult
from des_multi_agent.prediction import CurvePrediction
from des_multi_agent import reporting


def _result(with_provenance: bool) -> DesResult:
    curve = CurvePrediction(
        smiles_a="CCO", smiles_b="O",
        ratios=[0.1, 0.5, 0.9], tm_pred_k=[250.0, 245.0, 255.0],
        t1_k=200.0, t2_k=273.15, checkpoint_path="ckpt.pt",
    )
    r = DesResult(
        curve=curve, absolute_pass=True, relative_pass=True, is_des=True,
        rationale="demo", min_tm_k=245.0, eutectic_ratio_b=0.5,
    )
    if with_provenance:
        r = replace(
            r,
            t1_source="experimental", t1_confidence=0.95,
            t2_source="qspr", t2_confidence=0.70,
        )
    return r


def test_text_report_shows_tm_provenance_when_present():
    text = reporting.format_report([_result(with_provenance=True)], resolve_names=False)
    assert "Melting-point inputs:" in text
    assert "experimental" in text
    assert "qspr" in text


def test_text_report_omits_section_when_no_provenance():
    text = reporting.format_report([_result(with_provenance=False)], resolve_names=False)
    assert "Melting-point inputs:" not in text


def test_json_report_includes_tm_provenance_when_present():
    payload = json.loads(reporting.format_report_json([_result(with_provenance=True)], resolve_names=False))
    row = payload[0]
    assert row["t1_source"] == "experimental"
    assert row["t2_source"] == "qspr"
    assert row["t1_confidence"] == 0.95
    assert row["t2_confidence"] == 0.70


def test_json_report_omits_tm_provenance_when_absent():
    payload = json.loads(reporting.format_report_json([_result(with_provenance=False)], resolve_names=False))
    assert "t1_source" not in payload[0]
