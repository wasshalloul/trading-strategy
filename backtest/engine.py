"""
Backtesting Engine
---------------------
Simulates trading a position signal against historical prices, applying
transaction costs, and computes standard performance metrics.

Design choices (stated explicitly, since these assumptions materially
affect results):
- Trades execute at the NEXT day's close after a signal (no lookahead bias --
  you can't trade on today's close using today's crossover, since the
  crossover isn't confirmed until the market closes)
- Transaction costs are modeled as a fixed % of trade value (covers
  commission + slippage combined, a common simplification)
- No leverage: position is either 100% invested or 100% cash
"""

import numpy as np
import pandas as pd


class Backtester:
    def __init__(self, transaction_cost_pct=0.001, risk_free_rate=0.02):
        """
        Parameters
        ----------
        transaction_cost_pct : float
            Cost per trade as a fraction of trade value (0.001 = 10 bps,
            a reasonable estimate for commission + slippage on a liquid ETF)
        risk_free_rate : float
            Annualized risk-free rate, used for Sharpe ratio calculation
        """
        self.transaction_cost_pct = transaction_cost_pct
        self.risk_free_rate = risk_free_rate

    def run(self, signals_df: pd.DataFrame) -> pd.DataFrame:
        """
        Parameters
        ----------
        signals_df : pd.DataFrame with columns 'price' and 'position'
            (as produced by a strategy's generate_signals method)

        Returns
        -------
        pd.DataFrame with daily returns, equity curve, and trade costs
        """
        df = signals_df.copy()

        # Shift position by 1 day: trade executes the day AFTER the signal
        # is confirmed. This avoids lookahead bias.
        df["executed_position"] = df["position"].shift(1).fillna(0)

        # Daily market return
        df["market_return"] = df["price"].pct_change().fillna(0)

        # Strategy return = market return while in position, 0 while flat
        df["strategy_return_gross"] = df["executed_position"] * df["market_return"]

        # Transaction costs: charged whenever executed_position changes
        # (i.e. we actually entered or exited a trade)
        df["trade_occurred"] = df["executed_position"].diff().abs().fillna(0)
        df["transaction_cost"] = df["trade_occurred"] * self.transaction_cost_pct

        df["strategy_return_net"] = df["strategy_return_gross"] - df["transaction_cost"]

        # Equity curves (starting at 1.0 = 100%)
        df["strategy_equity"] = (1 + df["strategy_return_net"]).cumprod()
        df["buy_hold_equity"] = (1 + df["market_return"]).cumprod()

        return df

    def performance_summary(self, backtest_df: pd.DataFrame, periods_per_year=252) -> dict:
        """
        Compute standard performance metrics from a completed backtest.
        """
        returns = backtest_df["strategy_return_net"]
        equity = backtest_df["strategy_equity"]

        total_return = equity.iloc[-1] - 1
        n_years = len(returns) / periods_per_year
        cagr = (equity.iloc[-1] ** (1 / n_years)) - 1 if n_years > 0 else np.nan

        ann_return = returns.mean() * periods_per_year
        ann_vol = returns.std() * np.sqrt(periods_per_year)
        sharpe = (ann_return - self.risk_free_rate) / ann_vol if ann_vol > 0 else np.nan

        # Max drawdown
        running_max = equity.cummax()
        drawdown = (equity - running_max) / running_max
        max_drawdown = drawdown.min()

        # Win rate: fraction of trading days with a position that were profitable
        active_returns = returns[backtest_df["executed_position"] == 1]
        win_rate = (active_returns > 0).mean() if len(active_returns) > 0 else np.nan

        n_trades = int(backtest_df["trade_occurred"].sum())
        total_costs = backtest_df["transaction_cost"].sum()

        buy_hold_total_return = backtest_df["buy_hold_equity"].iloc[-1] - 1

        return {
            "total_return": round(total_return * 100, 2),
            "buy_hold_return": round(buy_hold_total_return * 100, 2),
            "cagr": round(cagr * 100, 2),
            "annualized_volatility": round(ann_vol * 100, 2),
            "sharpe_ratio": round(sharpe, 3),
            "max_drawdown": round(max_drawdown * 100, 2),
            "win_rate": round(win_rate * 100, 2) if not np.isnan(win_rate) else None,
            "num_trades": n_trades,
            "total_transaction_costs_pct": round(total_costs * 100, 2),
        }


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__) + "/..")
    from strategy.ma_crossover import MovingAverageCrossover

    # Synthetic test: trending market with noise
    dates = pd.date_range("2020-01-01", periods=1000, freq="D")
    rng = np.random.default_rng(42)
    returns = rng.normal(0.0005, 0.015, 1000)  # slight upward drift, realistic daily vol
    prices = pd.Series(100 * np.cumprod(1 + returns), index=dates)

    strat = MovingAverageCrossover(short_window=50, long_window=200)
    signals = strat.generate_signals(prices)

    bt = Backtester(transaction_cost_pct=0.001, risk_free_rate=0.02)
    result = bt.run(signals)
    summary = bt.performance_summary(result)

    print("Performance Summary (synthetic data):")
    for k, v in summary.items():
        print(f"  {k}: {v}")
