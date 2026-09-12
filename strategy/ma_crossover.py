"""
Moving Average Crossover Strategy
------------------------------------
Classic trend-following strategy: go long when the short-term moving
average crosses above the long-term moving average ("golden cross"), and
exit / go flat when it crosses back below ("death cross").

This is intentionally simple. The point of this project isn't to claim
this strategy beats the market -- it's to build a correct, honest backtest
with realistic costs, and report the real performance, good or bad.
"""

import numpy as np
import pandas as pd


class MovingAverageCrossover:
    def __init__(self, short_window=50, long_window=200):
        """
        Parameters
        ----------
        short_window : int   Short moving average lookback (days)
        long_window : int    Long moving average lookback (days)
        """
        if short_window >= long_window:
            raise ValueError("short_window must be less than long_window")
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, prices: pd.Series) -> pd.DataFrame:
        """
        Parameters
        ----------
        prices : pd.Series   Close prices, indexed by date

        Returns
        -------
        pd.DataFrame with columns: price, short_ma, long_ma, signal, position
            signal: +1 on golden cross day, -1 on death cross day, 0 otherwise
            position: 1 if long (short_ma > long_ma), 0 if flat
        """
        df = pd.DataFrame({"price": prices})
        df["short_ma"] = df["price"].rolling(self.short_window).mean()
        df["long_ma"] = df["price"].rolling(self.long_window).mean()

        # position = 1 (long) whenever short MA is above long MA, else 0 (flat)
        df["position"] = np.where(df["short_ma"] > df["long_ma"], 1, 0)
        # only valid once both MAs are computed
        df.loc[df["long_ma"].isna(), "position"] = 0

        # signal marks the day the position CHANGES (the actual crossover event)
        df["signal"] = df["position"].diff().fillna(0)

        return df


if __name__ == "__main__":
    # Quick sanity check with synthetic data: an uptrend should trigger a
    # golden cross and stay long
    dates = pd.date_range("2020-01-01", periods=300, freq="D")
    trend = np.linspace(100, 200, 300) + np.random.default_rng(0).normal(0, 2, 300)
    prices = pd.Series(trend, index=dates)

    strat = MovingAverageCrossover(short_window=20, long_window=50)
    signals = strat.generate_signals(prices)

    n_golden = (signals["signal"] == 1).sum()
    n_death = (signals["signal"] == -1).sum()
    print(f"Golden crosses: {n_golden}, Death crosses: {n_death}")
    print(f"Days spent long: {signals['position'].sum()} / {len(signals)}")
    print(signals.tail())
