"""SVI parameterisation of one implied-variance smile (Gatheral, 2004).

Raw SVI, total implied variance w = sigma_imp^2 T as a function of k = ln(K/F):
    w(k) = a + b [ rho (k - m) + sqrt((k - m)^2 + s^2) ]

Five parameters, linear wings (Lee's moment formula says total variance can grow
at most linearly in |k|), smooth everywhere, and its derivatives are analytic,
which is exactly what Dupire's formula needs.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares


def w(k, a, b, rho, m, s):
    d = k - m
    return a + b * (rho * d + np.sqrt(d * d + s * s))


def dw(k, a, b, rho, m, s):
    d = k - m
    return b * (rho + d / np.sqrt(d * d + s * s))


def d2w(k, a, b, rho, m, s):
    d = k - m
    return b * s * s / (d * d + s * s) ** 1.5


def density_g(k, params):
    """Gatheral's g(k); the risk-neutral density is n(d2) g / sqrt(w), so g < 0 means
    butterfly arbitrage. g is also the denominator of Dupire's formula in (k, w)."""
    W, W1, W2 = w(k, *params), dw(k, *params), d2w(k, *params)
    return (1 - k * W1 / (2 * W)) ** 2 - W1**2 / 4 * (1 / W + 0.25) + W2 / 2


def fit(k, total_var, weights=None):
    """Least-squares fit of raw SVI to one slice of total implied variance.

    Constraint a + b s sqrt(1 - rho^2) >= 0 (minimum of w non-negative) is imposed by
    reparameterising a = a_min_excess - b s sqrt(1-rho^2), with a_min_excess >= 0.
    """
    k, tv = np.asarray(k, float), np.asarray(total_var, float)
    wts = np.ones_like(tv) if weights is None else np.asarray(weights, float)

    def unpack(x):
        e, b, rho, m, s = x
        return (e - b * s * np.sqrt(1 - rho**2), b, rho, m, s)

    def resid(x):
        return wts * (w(k, *unpack(x)) - tv)

    lo = [0.0, 1e-6, -0.999, k.min() - 0.5, 1e-4]
    hi = [tv.max(), 10.0, 0.999, k.max() + 0.5, 5.0]
    best = None
    for rho0, s0 in [(-0.7, 0.1), (-0.3, 0.3), (-0.9, 0.05), (0.0, 0.5)]:
        x0 = np.clip([tv.min(), 0.1, rho0, 0.0, s0], lo, hi)
        res = least_squares(resid, x0, bounds=(lo, hi), x_scale="jac", xtol=1e-14, ftol=1e-14)
        if best is None or res.cost < best.cost:
            best = res
    return unpack(best.x)
