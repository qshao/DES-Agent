import pandas as pd
import pytest

from ml_des_mp.src.splits import kfold_strict_pair, split_strict_pair


def test_split_strict_pair_rejects_single_key():
    df = pd.DataFrame({"pair_key": ["A", "A"], "value": [1, 2]})
    with pytest.raises(ValueError, match="at least two unique pair keys"):
        split_strict_pair(df, "pair_key", test_size=0.5, seed=42)


def test_kfold_strict_pair_rejects_too_many_folds_for_unique_keys():
    df = pd.DataFrame({"pair_key": ["A", "B"], "value": [1, 2]})
    with pytest.raises(ValueError, match="at least k unique pair keys"):
        kfold_strict_pair(df, "pair_key", k=3, seed=42)
