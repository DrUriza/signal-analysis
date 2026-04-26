import numpy as np
import pandas as pd
import pytest

from signal_analysis.filters import Regressor


def test_regressor_returns_prediction_and_degree():
    df = pd.DataFrame({"close": [10.0, 10.5, 11.0, 11.2, 11.8, 12.0]})

    y_pred, degree = Regressor(df)

    assert isinstance(y_pred, np.ndarray)
    assert len(y_pred) == len(df)
    assert isinstance(degree, int)
    assert degree >= 1


def test_regressor_raises_with_missing_close_column():
    df = pd.DataFrame({"price": [1.0, 2.0, 3.0]})

    with pytest.raises(ValueError, match="close"):
        Regressor(df)
