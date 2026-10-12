"""
Hand-verification of the HYG-SPY quintile hit-rate claim.

The research CLI reported: HYG +0.19% today, 80% historical match rate
vs a 55% base rate (n=200), for a +25 point lift. This script re-derives
that number independently using yfinance, so you can confirm the tool's
math matches reality before trusting it.

Run: python verify_hyg_claim.py
Requires: pip install yfinance pandas numpy
"""
import yfinance as yf
import pandas as pd
import numpy as np

LOOKBACK_PERIOD = "4y"  # matches the report's documented methodology (quintiles over ~4 years)
TODAY_HYG_MOVE = 0.0019  # +0.19%, from the report — update if re-running later


def pct_change(ticker: str, period: str = LOOKBACK_PERIOD) -> pd.Series:
    data = yf.download(ticker, period=period, progress=False)["Close"]
    if isinstance(data, pd.DataFrame):
        # Newer yfinance returns a DataFrame with the ticker as a column
        # even for a single symbol -- squeeze it down to a plain Series.
        data = data.squeeze("columns")
    return data.pct_change().dropna()


def main():
    spy = pct_change("SPY")
    hyg = pct_change("HYG")

    df = pd.concat([spy, hyg], axis=1, keys=["SPY", "HYG"]).dropna()
    # NOTE: do NOT truncate to a recent window here -- quintile boundaries
    # and bucket membership need to be computed over the same ~4 year
    # history the app uses, or bucket sizes (and thus hit rates) won't
    # be comparable. This was the bug in the first version of this script.

    # Base rate: how often does SPY move up on any day in this window?
    base_rate = (df["SPY"] > 0).mean()
    print(f"SPY base rate (up days), n={len(df)}: {base_rate:.1%}")

    # Bucket HYG's daily moves into quintiles, same method as the report
    df["HYG_quintile"] = pd.qcut(df["HYG"], 5, labels=False, duplicates="drop")

    # Find which quintile today's +0.19% HYG move falls into
    boundaries = df["HYG"].quantile([0.2, 0.4, 0.6, 0.8]).values
    today_quintile = int(np.searchsorted(boundaries, TODAY_HYG_MOVE))
    print(f"Today's HYG move ({TODAY_HYG_MOVE:+.2%}) falls in quintile {today_quintile} (0-4)")

    # Hit rate: on days HYG was in this same quintile, how often did SPY move up?
    bucket = df[df["HYG_quintile"] == today_quintile]
    hit_rate = (bucket["SPY"] > 0).mean()
    n = len(bucket)
    lift = hit_rate - base_rate

    print(f"\nHit rate in this quintile: {hit_rate:.1%} (n={n})")
    print(f"Lift vs base rate: {lift:+.1%}")
    print(f"\nReport claimed: 80% hit rate, +25 point lift, n=200")
    print(f"Your independent check: {hit_rate:.1%} hit rate, {lift:+.1%} lift, n={n}")
    print("\n(Small differences are expected — exact date range, split-adjustment,")
    print("and vendor data can differ slightly. Look for the same ballpark, not exact match.)")


if __name__ == "__main__":
    main()