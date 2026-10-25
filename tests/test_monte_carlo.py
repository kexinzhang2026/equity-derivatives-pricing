"""Monte Carlo engines, checked against the analytic, COS and PDE engines.

All tests use fixed seeds, so they are deterministic; tolerances are stated in
standard errors so that a failure means bias, not bad luck.
"""
from functools import partial

import numpy as np
import pytest

from derivpricing.engines import cos, lsm, pde
from derivpricing.engines import monte_carlo as mc
from derivpricing.models import black_scholes as bs
from derivpricing.models import heston

S, K, T, r, SIGMA, Q = 100.0, 95.0, 0.75, 0.04, 0.25, 0.015


def _call_samples(n, seed, antithetic=False):
    paths = mc.gbm_paths(S, T, r, Q, SIGMA, 1, n, np.random.default_rng(seed), antithetic)
    ST = paths[:, -1]
    return np.exp(-r * T) * np.maximum(ST - K, 0.0), np.exp(-r * T) * ST


def test_gbm_european_within_three_se():
    y, _ = _call_samples(200_000, seed=1)
    assert mc.plain(y).contains(bs.price(S, K, T, r, SIGMA, Q))


def test_stderr_scales_like_inverse_sqrt_n():
    se = [mc.plain(_call_samples(n, seed=2)[0]).stderr for n in (10_000, 160_000)]
    assert se[0] / se[1] == pytest.approx(4.0, rel=0.05)      # sqrt(16) = 4


def test_variance_reduction_is_real_and_unbiased():
    exact = bs.price(S, K, T, r, SIGMA, Q)
    n = 200_000
    plain = mc.plain(_call_samples(n, seed=3)[0])
    anti = mc.antithetic(_call_samples(n, seed=3, antithetic=True)[0])
    y, x = _call_samples(n, seed=3)
    cv = mc.control_variate(y, x, S * np.exp(-Q * T))          # E[e^{-rT} S_T] = S e^{-qT}
    for est in (anti, cv):
        assert est.contains(exact)
        assert est.stderr < plain.stderr
    # For an ITM call the discounted stock is highly correlated with the payoff.
    assert cv.stderr < plain.stderr / 2.5


def test_heston_qe_matches_cos():
    # Fang-Oosterlee parameters violate the Feller condition (ratio ~0.38),
    # the regime where the QE scheme matters.
    p = heston.HestonParams(v0=0.0175, kappa=1.5768, theta=0.0398, xi=0.5751, rho=-0.5711)
    assert p.feller_ratio < 1
    ref = cos.price(partial(heston.char_func, T=1.0, r=0.0, q=0.0, p=p), 100.0, 100.0, 1.0, 0.0)
    ST, v = mc.heston_paths_qe(100.0, 1.0, 0.0, 0.0, p, 32, 200_000, np.random.default_rng(4))
    assert (v >= 0).all()
    est = mc.control_variate(np.maximum(ST[:, -1] - 100.0, 0), ST[:, -1], 100.0)
    assert est.contains(ref)


def test_lsm_american_put_against_pde():
    # Longstaff & Schwartz (2001) Table 1 case: S=36, K=40, r=6%, sigma=20%, T=1, 50 dates.
    put = lambda x: np.maximum(40.0 - x, 0.0)
    rng = np.random.default_rng(5)
    paths = mc.gbm_paths(36.0, 1.0, 0.06, 0.0, 0.2, 50, 100_000, rng)
    fit = mc.gbm_paths(36.0, 1.0, 0.06, 0.0, 0.2, 50, 100_000, rng)
    est = lsm.price(paths, put, 0.06, 1.0, scale=40.0, fit_paths=fit)

    american = pde.price_vanilla(36.0, 40.0, 1.0, 0.06, 0.2, is_call=False, american=True,
                                 M=800, N=800).price
    european = bs.price(36.0, 40.0, 1.0, 0.06, 0.2, is_call=False)
    # Out-of-sample LSM is a lower bound (suboptimal rule, 50 dates instead of continuous)
    # but should recover most of the early-exercise premium.
    assert est.price < american + 3 * est.stderr
    assert est.price > european + 0.9 * (american - european)
