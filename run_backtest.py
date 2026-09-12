"""
Run the MA Crossover strategy on real historical data.

Usage:
    python run_backtest.py --ticker SPY --short 50 --long 200 --years 10
"""

import sys
import os
import argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import yfinance as yf
from strategy.ma_crossover import MovingAverageCrossover
from backtest.engine import Backtester

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--short", type=int, default=50)
    parser.add_argument("--long", type=int, default=200)
    parser.add_argument("--years", type=int, default=10)
    parser.add_argument("--cost", type=float, default=0.001, help="Transaction cost as fraction (0.001 = 10bps)")
    parser.add_argument("--rf", type=float, default=0.02, help="Risk-free rate for Sharpe ratio")
    args = parser.parse_args()

    print(f"Downloading {args.years}y of data for {args.ticker}...")
    data = yf.download(args.ticker, period=f"{args.years}y", progress=False)

    if data.empty:
        raise ValueError(f"No data returned for {args.ticker}. Check the ticker symbol.")

    prices = data["Close"]
    if isinstance(prices, pd.DataFrame):  # yfinance sometimes returns multi-index columns
        prices = prices.iloc[:, 0]

    strat = MovingAverageCrossover(short_window=args.short, long_window=args.long)
    signals = strat.generate_signals(prices)

    bt = Backtester(transaction_cost_pct=args.cost, risk_free_rate=args.rf)
    result = bt.run(signals)
    summary = bt.performance_summary(result)

    print(f"\n{args.ticker} | MA({args.short}/{args.long}) crossover | {args.years} years")
    print("=" * 55)
    for k, v in summary.items():
        print(f"  {k:30s}: {v}")

    out_path = os.path.join(os.path.dirname(__file__), "data", f"{args.ticker}_backtest.csv")
    result.to_csv(out_path)
    print(f"\nSaved full backtest data to {out_path}")
