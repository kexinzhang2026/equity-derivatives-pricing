"""Download today's full SPX option chain from Yahoo Finance and save one CSV snapshot.

Run once per trading day, ideally at the same time (e.g. ~30 min before the US close):
    python scripts/fetch_spx_chain.py            # writes data/raw/spx_YYYYMMDD.csv

To automate on macOS/Linux, add a cron line such as (weekdays, 15:30 New York time):
    30 15 * * 1-5  cd /path/to/repo && python scripts/fetch_spx_chain.py

Caveats worth knowing (and saying in an interview):
  - Yahoo quotes are delayed ~15 min and the spot and option quotes are not synchronous,
    so the forward is later implied from put-call parity instead of trusting `spot`.
  - Yahoo's own impliedVolatility column is not used; we recompute IV ourselves.
"""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf


def fetch(ticker: str = "^SPX") -> pd.DataFrame:
    tk = yf.Ticker(ticker)
    spot = float(tk.history(period="1d")["Close"].iloc[-1])
    today = pd.Timestamp(date.today())
    frames = []
    for expiry in tk.options:
        chain = tk.option_chain(expiry)
        for side, is_call in [(chain.calls, True), (chain.puts, False)]:
            frames.append(pd.DataFrame({
                "quote_date": today,
                "expiry": pd.Timestamp(expiry),
                "strike": side["strike"],
                "is_call": is_call,
                "bid": side["bid"],
                "ask": side["ask"],
                "volume": side["volume"],
                "open_interest": side["openInterest"],
                "spot": spot,
            }))
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="^SPX")
    parser.add_argument("--out", default=Path(__file__).resolve().parents[1] / "data" / "raw")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = fetch(args.ticker)
    path = out / f"spx_{date.today():%Y%m%d}.csv"
    df.to_csv(path, index=False)
    print(f"saved {len(df)} quotes across {df['expiry'].nunique()} expiries to {path}")
