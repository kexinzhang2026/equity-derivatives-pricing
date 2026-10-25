"""SABR model and Hagan et al. (2002) implied volatility approximation.

Dynamics of the forward (one expiry at a time):
    dF = alpha F^beta dW1,   d alpha = nu alpha dW2,   d<W1, W2> = rho dt

SABR is a per-expiry smile model: each expiry gets its own (alpha, rho, nu)
with beta fixed. It does not describe the term structure jointly, which is the
key contrast with Heston in the calibration report.

Hagan's formula is a small-time asymptotic expansion, so it is accurate for
short expiries and moderate nu*sqrt(T); for long expiries and low strikes it
can imply a negative density (butterfly arbitrage). See
docs/derivations/07_sabr.md and the arbitrage check in the tests.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares


def hagan_iv(K, F, T, alpha, beta, rho, nu):
    """Black (lognormal) implied vol from Hagan's expansion, vectorised in K.

    sigma = alpha / [(FK)^((1-b)/2) (1 + (1-b)^2/24 L^2 + (1-b)^4/1920 L^4)]
            * z / x(z)
            * [1 + ((1-b)^2/24 alpha^2/(FK)^(1-b) + rho b nu alpha / (4 (FK)^((1-b)/2))
                    + (2 - 3 rho^2)/24 nu^2) T]
    with L = ln(F/K), z = nu/alpha (FK)^((1-b)/2) L, x(z) = ln[(sqrt(1-2 rho z+z^2)+z-rho)/(1-rho)].
    """
    K = np.asarray(K, dtype=float)
    L = np.log(F / K)
    fk = (F * K) ** ((1 - beta) / 2)
    omb = 1 - beta
    denom = fk * (1 + omb**2 / 24 * L**2 + omb**4 / 1920 * L**4)
    z = nu / alpha * fk * L
    x = np.log((np.sqrt(1 - 2 * rho * z + z**2) + z - rho) / (1 - rho))
    # z/x(z) -> 1 as z -> 0 (ATM); use the series 1 - rho z / 2 to avoid 0/0.
    small = np.abs(z) < 1e-7
    zx = np.where(small, 1 - 0.5 * rho * z, z / np.where(small, 1.0, x))
    corr = 1 + (omb**2 / 24 * alpha**2 / fk**2 + rho * beta * nu * alpha / (4 * fk)
                + (2 - 3 * rho**2) / 24 * nu**2) * T
    return alpha / denom * zx * corr


def calibrate_slice(K, F, T, iv, beta=1.0, weights=None):
    """Fit (alpha, rho, nu) to one expiry's implied vols, beta fixed.

    Residuals are in implied-vol units directly (the formula gives IVs in closed form).
    beta and rho both control the skew, so they cannot be fitted together from a
    single smile; fixing beta is standard (beta = 1: lognormal backbone).
    Returns dict(alpha, rho, nu, rmse_iv).
    """
    K, iv = np.asarray(K, float), np.asarray(iv, float)
    w = np.ones_like(iv) if weights is None else np.asarray(weights, float)
    atm_iv = float(np.interp(0.0, np.log(K / F), iv))
    alpha0 = atm_iv * F ** (1 - beta)

    def resid(x):
        return w * (hagan_iv(K, F, T, x[0], beta, x[1], x[2]) - iv)

    best = None
    for rho0, nu0 in [(-0.6, 0.5), (-0.3, 1.5), (-0.9, 3.0)]:
        res = least_squares(resid, [alpha0, rho0, nu0],
                            bounds=([1e-6, -0.9999, 1e-4], [np.inf, 0.9999, 10.0]),
                            x_scale="jac", xtol=1e-12, ftol=1e-12)
        if best is None or res.cost < best.cost:
            best = res
    a, r, n = best.x
    err = hagan_iv(K, F, T, a, beta, r, n) - iv
    return {"alpha": a, "rho": r, "nu": n, "beta": beta, "rmse_iv": float(np.sqrt(np.mean(err**2)))}
