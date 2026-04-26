import pandas                          as pd
from signal_analysis.adapters.patterns import compute_candlestick_patterns


def test_compute_candlestick_patterns_returns_expected_columns():
    ohlc = pd.DataFrame(
        {
            "open": [10.0, 10.0, 10.0],
            "high": [11.0, 10.2, 12.0],
            "low": [9.0, 9.8, 9.5],
            "close": [10.8, 10.01, 11.6],
        }
    )

    out = compute_candlestick_patterns(ohlc)

    assert list(out.columns) == [
        "candle_class",
        "doji",
        "hammer",
        "bullish_engulfing",
        "bearish_engulfing",
    ]
    assert len(out) == len(ohlc)
