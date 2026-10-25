"""Synthetic implied vol surfaces generated from a known model.

Calibration code is first tested here: if the optimiser cannot recover the
parameters that generated a noise-free surface, nothing it says about real data
can be trusted. Adding bid/ask-sized noise then shows which parameters remain
identifiable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from derivpricing.calibration import heston_calib
from derivpricing.models import black_scholes as bs
from derivpricing.models.heston import HestonParams


def heston_surface(p: HestonParams, S0=100.0, r=0.04, q=0.012,
                   expiries=(0.08, 0.25, 0.5, 1.0, 2.0), n_strikes=15, z_range=(-3.0, 2.0),
                   noise_iv=0.0, half_spread_iv=0.005, seed=0) -> pd.DataFrame:
    """Surface in the same format as surface.build_surface, priced with Heston-COS.

    Strikes per expiry span z_range standardised moneyness ln(K/F)/(sqrt(theta) sqrt(T)).
    noise_iv: std of Gaussian noise added to each implied vol (e.g. 0.002 = 0.2 vol pt).
    """
    rng = np.random.default_rng(seed)
    rows = []
    for T in expiries:
        F, D = S0 * np.exp((r - q) * T), np.exp(-r * T)
        z = np.linspace(*z_range, n_strikes)
        K = F * np.exp(z * np.sqrt(p.theta * T))
        rows.append(pd.DataFrame({"T": T, "K": K, "F": F, "D": D, "is_call": K >= F}))
    s = pd.concat(rows, ignore_index=True)
    s["quote_date"] = pd.Timestamp("2026-01-02")
    s["expiry"] = s["quote_date"] + pd.to_timedelta(np.round(s["T"] * 365), unit="D")
    s["k"] = np.log(s["K"] / s["F"])

    price = heston_calib.model_prices(p, s, N=512)
    S_eff, rr = (s["D"] * s["F"]).to_numpy(), (-np.log(s["D"]) / s["T"]).to_numpy()
    args = (S_eff, s["K"].to_numpy(), s["T"].to_numpy(), rr, 0.0, s["is_call"].to_numpy())
    iv = bs.implied_vol_vec(price, *args) + noise_iv * rng.standard_normal(len(s))
    s["iv"], s["iv_bid"], s["iv_ask"] = iv, iv - half_spread_iv, iv + half_spread_iv
    s["mid"] = bs.price(args[0], args[1], args[2], args[3], iv, 0.0, args[5])
    s["bid"] = bs.price(args[0], args[1], args[2], args[3], s["iv_bid"], 0.0, args[5])
    s["ask"] = bs.price(args[0], args[1], args[2], args[3], s["iv_ask"], 0.0, args[5])
    s["vega"] = bs.greeks(args[0], args[1], args[2], args[3], iv, 0.0)["vega"]
    return s
