"""Calibrate Heston to a whole implied volatility surface with the COS engine.

Objective (derivation: docs/derivations/06_calibration.md):
    min_p  Σ_i [ (C_model(K_i, T_i; p) - C_mkt,i) / vega_i ]^2

Dividing the price error by the Black-Scholes vega turns it, to first order,
into an implied-vol error (ΔC ≈ vega Δσ), without inverting model prices to
IVs inside the optimiser. So every option counts in vol points, not dollars:
otherwise long-dated ATM options, worth hundreds of dollars, would dominate.

The problem is non-convex and κ and θ are only weakly identified separately
(a fast reversion to a low level and a slow one to a high level produce similar
term structures), so we run a local least-squares solver from several starting
points and keep every result: the spread of the near-optimal solutions *is* the
identifiability analysis.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from derivpricing.engines import cos
from derivpricing.models import black_scholes as bs
from derivpricing.models.heston import HestonParams, char_func

#                 v0     kappa  theta  xi    rho
LOWER = np.array([1e-4, 1e-2, 1e-4, 1e-2, -0.999])
UPPER = np.array([1.0, 20.0, 1.0, 5.0, 0.999])
NAMES = ["v0", "kappa", "theta", "xi", "rho"]


@dataclass
class HestonFit:
    params: HestonParams
    rmse_iv: float                 # root-mean-square implied vol error (vol units, 0.01 = 1 pt)
    surface: pd.DataFrame          # input surface with model_price, model_iv, iv_error columns
    starts: list = field(default_factory=list)   # (start, fitted params, cost) for each start


def model_prices(p: HestonParams, surface: pd.DataFrame, N: int = 192) -> np.ndarray:
    """Heston prices for every row of the surface, one vectorised COS call per expiry."""
    out = np.empty(len(surface))
    for T, idx in surface.groupby("T").indices.items():
        rows = surface.iloc[idx]
        F, D = rows["F"].iloc[0], rows["D"].iloc[0]
        r = -np.log(D) / T
        cf = partial(char_func, T=T, r=r, q=0.0, p=p)
        K = rows["K"].to_numpy()
        is_call = rows["is_call"].to_numpy()
        call = np.atleast_1d(cos.price(cf, D * F, K, T, r, 0.0, True, N=N))
        put = call - D * (F - K)                  # parity, same forward and discount
        out[idx] = np.where(is_call, call, put)
    return out


def _residuals(x_free, free, x_full, surface, mkt, vega):
    x = x_full.copy()
    x[free] = x_free
    return (model_prices(HestonParams(*x), surface) - mkt) / vega


def calibrate(surface: pd.DataFrame, starts=None, seed: int = 0, n_random: int = 4,
              fixed: dict | None = None) -> HestonFit:
    """Fit Heston to one quote date's surface (output of surface.build_surface).

    starts: optional list of 5-vectors; by default one 'typical equity' start plus
    n_random points drawn uniformly inside a plausible box.
    fixed: e.g. {"kappa": 2.0} holds parameters fixed; used for profile-likelihood
    plots (fix kappa on a grid, refit the rest, look at how flat the error is).
    """
    fixed = fixed or {}
    free = np.array([n not in fixed for n in NAMES])
    mkt = surface["mid"].to_numpy()
    # Floor tiny wing vegas so a single deep-OTM quote cannot dominate the objective.
    vega = surface["vega"].to_numpy()
    vega = np.maximum(vega, 0.01 * np.median(vega))
    atm_var = float(np.median(surface.loc[surface["k"].abs() < 0.05, "iv"]) ** 2) \
        if (surface["k"].abs() < 0.05).any() else 0.04
    if starts is None:
        rng = np.random.default_rng(seed)
        starts = [np.array([atm_var, 2.0, atm_var, 0.6, -0.7])]
        for _ in range(n_random):
            starts.append(np.array([rng.uniform(0.3, 2) * atm_var, rng.uniform(0.5, 6),
                                    rng.uniform(0.3, 2) * atm_var, rng.uniform(0.2, 1.5),
                                    rng.uniform(-0.95, -0.2)]))
    results = []
    for x0 in starts:
        x_full = np.clip(np.array(x0, float), LOWER, UPPER)
        for name, value in fixed.items():
            x_full[NAMES.index(name)] = value
        res = least_squares(_residuals, x_full[free], bounds=(LOWER[free], UPPER[free]),
                            args=(free, x_full, surface, mkt, vega), x_scale="jac",
                            method="trf", xtol=1e-10, ftol=1e-10, max_nfev=400)
        x_full[free] = res.x
        results.append((x0, x_full, 2 * res.cost))
    best = min(results, key=lambda t: t[2])
    p = HestonParams(*best[1])

    s = surface.copy()
    s["model_price"] = model_prices(p, s)
    S_eff, r = s["D"] * s["F"], -np.log(s["D"]) / s["T"]
    s["model_iv"] = bs.implied_vol_vec(s["model_price"].to_numpy(), S_eff.to_numpy(),
                                       s["K"].to_numpy(), s["T"].to_numpy(), r.to_numpy(), 0.0,
                                       s["is_call"].to_numpy())
    s["iv_error"] = s["model_iv"] - s["iv"]
    rmse = float(np.sqrt(np.nanmean(s["iv_error"] ** 2)))
    return HestonFit(params=p, rmse_iv=rmse, surface=s, starts=results)
