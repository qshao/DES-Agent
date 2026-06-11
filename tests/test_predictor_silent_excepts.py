"""Verify that unexpected exceptions are NOT silently swallowed in predictors."""
from __future__ import annotations

import pytest

from des_multi_agent.predictors.stability_constants import _predict_from_model as _sc_predict
from des_multi_agent.predictors.designsolvents import _predict_from_model as _ds_predict


class _BadModel:
    """Model whose .predict() raises MemoryError — should NOT be swallowed."""
    def predict(self, X):
        raise MemoryError("out of memory")


# Minimal feature dict — any non-empty dict works; the helper sorts keys
# to build the feature vector before calling model.predict().
_FEATURES = {"a": 1.0, "b": 2.0}


def test_stability_constants_propagates_memory_error():
    with pytest.raises(MemoryError):
        _sc_predict(_BadModel(), _FEATURES)


def test_designsolvents_propagates_memory_error():
    with pytest.raises(MemoryError):
        _ds_predict(_BadModel(), _FEATURES)
