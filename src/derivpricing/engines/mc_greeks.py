"""Monte Carlo Greeks under GBM: pathwise and likelihood-ratio estimators.

Both differentiate E[e^{-rT} f(S_T)] with respect to S0 without bumping and re-simulating:

  Pathwise (PW): move the derivative inside the expectation, along each path.
      S_T = S0 exp(...)  =>  dS_T/dS0 = S_T / S0,   delta = E[e^{-rT} f'(S_T) S_T / S0]
      Needs f differentiable almost everywhere and Lipschitz: fine for a call,
      useless for a digital (f' = 0 a.e., the delta mass sits on the kink).

  Likelihood ratio (LR): move the derivative onto the density instead.
      delta = E[e^{-rT} f(S_T) * d ln p(S_T; S0)/dS0] = E[e^{-rT} f(S_T) Z / (S0 sigma sqrt(T))]
      Works for any payoff, including digitals, but has higher variance.

Derivation: docs/derivations/10_greeks_hedging.md.
"""
from __future__ import annotations

import numpy as np

from derivpricing.engines.monte_carlo import MCResult, plain


def _terminal(S0, T, r, q, sigma, n, rng):
    Z = rng.standard_normal(n)
    ST = S0 * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    return Z, ST


def call_delta_pathwise(S0, K, T, r, q, sigma, n, rng) -> MCResult:
    _, ST = _terminal(S0, T, r, q, sigma, n, rng)
    return plain(np.exp(-r * T) * (ST > K) * ST / S0)


def delta_likelihood_ratio(payoff, S0, T, r, q, sigma, n, rng) -> MCResult:
    """Any payoff f(S_T). The score of the lognormal density w.r.t. S0 is Z / (S0 sigma sqrt T)."""
    Z, ST = _terminal(S0, T, r, q, sigma, n, rng)
    return plain(np.exp(-r * T) * payoff(ST) * Z / (S0 * sigma * np.sqrt(T)))


def gamma_lr_pathwise(K, S0, T, r, q, sigma, n, rng) -> MCResult:
    """Call gamma by the mixed estimator: LR applied to the pathwise delta.
    Pathwise twice fails (the indicator 1{S_T > K} has zero derivative a.e.); mixing works."""
    Z, ST = _terminal(S0, T, r, q, sigma, n, rng)
    h = (ST > K) * ST / S0                        # pathwise delta integrand h(S_T, S0)
    # Treat S_T as the random variable with density p(S_T; S0):
    # d/dS0 E[h] = E[h * score] + E[dh/dS0 at fixed S_T] = E[h (Z/(S0 sigma sqrt T) - 1/S0)]
    return plain(np.exp(-r * T) * h * (Z / (S0 * sigma * np.sqrt(T)) - 1.0 / S0))
