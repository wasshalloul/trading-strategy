"""
Plot the backtest results: equity curve vs buy-and-hold, drawdown, and
the price with MA crossover signals marked.

Usage:
    python plot_backtest.py --ticker SPY
"""

import sys
import os
import argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="SPY")
    args = parser.parse_args()

    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", f"{args.ticker}_backtest.csv")
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True,
                              gridspec_kw={"height_ratios": [2, 1.2, 1]})

    # --- Equity curve ---
    axes[0].plot(df.index, df["strategy_equity"], label="MA Crossover Strategy", color="#2563eb", linewidth=1.5)
    axes[0].plot(df.index, df["buy_hold_equity"], label="Buy & Hold", color="#6b7280", linewidth=1.5, linestyle="--")
    axes[0].set_title(f"{args.ticker} -- Strategy vs Buy & Hold Equity Curve")
    axes[0].set_ylabel("Growth of $1")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # --- Drawdown ---
    running_max = df["strategy_equity"].cummax()
    drawdown = (df["strategy_equity"] - running_max) / running_max * 100
    axes[1].fill_between(df.index, drawdown, 0, color="#dc2626", alpha=0.4)
    axes[1].plot(df.index, drawdown, color="#dc2626", linewidth=0.8)
    axes[1].set_title("Strategy Drawdown")
    axes[1].set_ylabel("Drawdown (%)")
    axes[1].grid(alpha=0.3)

    # --- Price with position (long/flat) shading ---
    axes[2].plot(df.index, df["price"], color="black", linewidth=0.8)
    in_position = df["executed_position"] == 1
    axes[2].fill_between(df.index, df["price"].min(), df["price"].max(),
                          where=in_position, color="#16a34a", alpha=0.15, label="Long")
    axes[2].set_title(f"{args.ticker} Price (green = long position)")
    axes[2].set_ylabel("Price")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), f"{args.ticker}_backtest.png")
    plt.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}")
