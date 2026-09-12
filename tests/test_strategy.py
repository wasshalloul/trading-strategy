"""
Unit tests for the MA crossover strategy and backtest engine.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from strategy.ma_crossover import MovingAverageCrossover
from backtest.engine import Backtester


def approx_equal(a, b, tol=1e-6):
    return abs(a - b) < tol


def test_invalid_windows_raise():
    try:
        MovingAverageCrossover(short_window=50, long_window=20)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_uptrend_produces_golden_cross_and_stays_long():
    dates = pd.date_range("2020-01-01", periods=300, freq="D")
    prices = pd.Series(np.linspace(100, 200, 300), index=dates)
    strat = MovingAverageCrossover(short_window=20, long_window=50)
    signals = strat.generate_signals(prices)

    assert (signals["signal"] == 1).sum() >= 1
    # once long, should stay long through a clean uptrend
    assert signals["position"].iloc[-1] == 1


def test_downtrend_never_goes_long():
    dates = pd.date_range("2020-01-01", periods=300, freq="D")
    prices = pd.Series(np.linspace(200, 100, 300), index=dates)
    strat = MovingAverageCrossover(short_window=20, long_window=50)
    signals = strat.generate_signals(prices)

    assert signals["position"].iloc[-1] == 0


def test_no_lookahead_bias():
    # Position should be executed the day AFTER the signal, never same-day
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    prices = pd.Series([100, 100, 100, 100, 100, 110, 120, 130, 140, 150], index=dates)
    df = pd.DataFrame({"price": prices, "position": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]})

    bt = Backtester(transaction_cost_pct=0.0)
    result = bt.run(df)

    # On the day position first becomes 1 (index 5), executed_position should
    # still be 0 (yesterday's position), only becoming 1 the NEXT day
    assert result["executed_position"].iloc[5] == 0
    assert result["executed_position"].iloc[6] == 1


def test_transaction_costs_reduce_returns():
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    prices = pd.Series([100, 105, 100, 105, 100, 105, 100, 105, 100, 105], index=dates)
    df = pd.DataFrame({"price": prices, "position": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]})

    bt_no_cost = Backtester(transaction_cost_pct=0.0)
    bt_with_cost = Backtester(transaction_cost_pct=0.01)

    result_no_cost = bt_no_cost.run(df)
    result_with_cost = bt_with_cost.run(df)

    assert result_with_cost["strategy_equity"].iloc[-1] < result_no_cost["strategy_equity"].iloc[-1]


def test_flat_market_zero_return():
    # Constant prices -> zero returns regardless of position
    dates = pd.date_range("2020-01-01", periods=20, freq="D")
    prices = pd.Series([100] * 20, index=dates)
    df = pd.DataFrame({"price": prices, "position": [1] * 20})

    bt = Backtester(transaction_cost_pct=0.0)
    result = bt.run(df)
    assert approx_equal(result["strategy_equity"].iloc[-1], 1.0)


def test_performance_summary_keys_present():
    dates = pd.date_range("2020-01-01", periods=300, freq="D")
    rng = np.random.default_rng(1)
    prices = pd.Series(100 * np.cumprod(1 + rng.normal(0.0003, 0.01, 300)), index=dates)

    strat = MovingAverageCrossover(short_window=20, long_window=50)
    signals = strat.generate_signals(prices)
    bt = Backtester()
    result = bt.run(signals)
    summary = bt.performance_summary(result)

    expected_keys = {"total_return", "buy_hold_return", "cagr", "annualized_volatility",
                      "sharpe_ratio", "max_drawdown", "win_rate", "num_trades",
                      "total_transaction_costs_pct"}
    assert expected_keys.issubset(summary.keys())


if __name__ == "__main__":
    tests = [
        test_invalid_windows_raise,
        test_uptrend_produces_golden_cross_and_stays_long,
        test_downtrend_never_goes_long,
        test_no_lookahead_bias,
        test_transaction_costs_reduce_returns,
        test_flat_market_zero_return,
        test_performance_summary_keys_present,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"PASS: {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL: {t.__name__} -> {e}")
    print(f"\n{passed}/{len(tests)} tests passed")
