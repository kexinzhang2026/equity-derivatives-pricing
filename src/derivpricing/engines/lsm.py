"""Longstaff-Schwartz (2001) least-squares Monte Carlo for Bermudan/American options.

Backward induction on simulated paths. At each exercise date t_n:
  1. among in-the-money paths, regress the realised discounted future cash flow
     on basis functions of S_{t_n}; the fit approximates the continuation value
     C(S) = E[discounted future cash flow | S_{t_n} = S]
  2. exercise where the immediate payoff exceeds the fitted C(S)
Only ITM paths are used because exercise is never optimal out of the money, and
fitting there wastes the basis on an irrelevant region.

Bias: the exercise rule is suboptimal, so pricing *new* paths with it gives a
lower bound. Reusing the regression paths for pricing adds a small upward
foresight bias; pass `fit_paths` to separate the two. Derivation:
docs/derivations/05_longstaff_schwartz.md.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from derivpricing.engines.monte_carlo import MCResult, plain


def _basis(x, degree):
    """Laguerre polynomials L_0..L_degree weighted by e^{-x/2}, as in Longstaff-Schwartz.
    x is moneyness S/K, so the regressors are O(1) and the design matrix is well conditioned."""
    return np.polynomial.laguerre.lagvander(x, degree) * np.exp(-x / 2)[:, None]


def _fit_rule(paths, payoff, r, dt, scale, degree):
    """Backward pass: return regression coefficients for each exercise date 1..n_steps-1."""
    n_paths, n_cols = paths.shape
    cash = payoff(paths[:, -1])
    coefs = [None] * n_cols
    for n in range(n_cols - 2, 0, -1):
        cash *= np.exp(-r * dt)
        ex_val = payoff(paths[:, n])
        itm = ex_val > 0
        if itm.sum() <= degree + 1:
            continue
        A = _basis(paths[itm, n] / scale, degree)
        beta, *_ = np.linalg.lstsq(A, cash[itm], rcond=None)
        coefs[n] = beta
        exercise = np.zeros(n_paths, bool)
        exercise[itm] = ex_val[itm] > A @ beta
        cash[exercise] = ex_val[exercise]
    return coefs


def price(paths, payoff: Callable, r, T, scale, degree=3, fit_paths=None) -> MCResult:
    """American price with exercise allowed at every grid date after t=0.

    paths: (n_paths, n_steps + 1) including S0;  scale: normalisation for the
    regressor (the strike).  fit_paths: independent paths to fit the rule on.
    """
    n_cols = paths.shape[1]
    dt = T / (n_cols - 1)
    coefs = _fit_rule(paths if fit_paths is None else fit_paths, payoff, r, dt, scale, degree)

    # Forward pass: apply the rule and record each path's discounted cash flow.
    n_paths = paths.shape[0]
    value = np.zeros(n_paths)
    alive = np.ones(n_paths, bool)
    for n in range(1, n_cols - 1):
        if coefs[n] is None:
            continue
        ex_val = payoff(paths[:, n])
        cand = alive & (ex_val > 0)
        cont = _basis(paths[cand, n] / scale, degree) @ coefs[n]
        stop = np.zeros(n_paths, bool)
        stop[cand] = ex_val[cand] > cont
        value[stop] = ex_val[stop] * np.exp(-r * n * dt)
        alive &= ~stop
    value[alive] = payoff(paths[alive, -1]) * np.exp(-r * T)
    return plain(value)
