"""COS engine and Heston characteristic function.

Three independent checks:
  1. COS with the Black-Scholes CF reproduces the closed form to machine precision,
     with the error falling exponentially in N.
  2. Heston-COS agrees with a completely different Fourier method (Lewis 2001
     single integral evaluated by adaptive quadrature) and with the reference
     value published by Fang & Oosterlee (2008).
  3. Heston collapses to Black-Scholes as the vol-of-vol goes to zero.
"""
from functools import partial

import numpy as np
import pytest
from scipy.integrate import quad

from derivpricing.engines import cos
from derivpricing.models import black_scholes as bs
from derivpricing.models import heston

FO_PARAMS = heston.HestonParams(v0=0.0175, kappa=1.5768, theta=0.0398, xi=0.5751, rho=-0.5711)


def lewis_call(cf, S, K, T, r):
    """Lewis (2001) call price, written for r = q = 0 (all this test needs):
    C = S - sqrt(S K)/pi * ∫_0^∞ Re[e^{i u ln(S/K)} φ(u - i/2)] / (u^2 + 1/4) du.
    """
    assert r == 0.0
    f = lambda u: (np.exp(1j * u * np.log(S / K)) * cf(u - 0.5j)).real / (u * u + 0.25)
    integral, _ = quad(f, 0, np.inf, limit=1000, epsabs=1e-13, epsrel=1e-13)
    return S - np.sqrt(S * K) / np.pi * integral


@pytest.mark.parametrize("T", [0.05, 1.0, 5.0])
def test_cos_matches_black_scholes(T):
    r, q, sigma, S = 0.03, 0.01, 0.2, 100.0
    K = np.linspace(50, 200, 16)
    cf = partial(bs.char_func, T=T, r=r, q=q, sigma=sigma)
    for is_call in (True, False):
        err = np.abs(cos.price(cf, S, K, T, r, q, is_call, N=128)
                     - bs.price(S, K, T, r, sigma, q, is_call))
        assert err.max() < 1e-12


def test_cos_converges_exponentially():
    # Error should drop by orders of magnitude each time N doubles (until round-off),
    # not by a constant factor as with an algebraic method.
    cf = partial(bs.char_func, T=1.0, r=0.03, q=0.01, sigma=0.2)
    exact = bs.price(100.0, 90.0, 1.0, 0.03, 0.2, 0.01)
    errs = [abs(cos.price(cf, 100.0, 90.0, 1.0, 0.03, 0.01, N=N) - exact) for N in (8, 16, 32)]
    assert errs[1] < 1e-2 * errs[0] and errs[2] < 1e-3 * errs[1]


def test_heston_matches_lewis_integral_and_paper():
    cf = partial(heston.char_func, T=1.0, r=0.0, q=0.0, p=FO_PARAMS)
    c_cos = cos.price(cf, 100.0, 100.0, 1.0, 0.0, N=512, L=12)
    assert c_cos == pytest.approx(lewis_call(cf, 100.0, 100.0, 1.0, 0.0), abs=1e-9)
    # Fang & Oosterlee (2008), Sec. 5.2, quote 5.785155450; we agree to ~2e-8.
    assert c_cos == pytest.approx(5.785155450, abs=1e-7)


def test_heston_tends_to_black_scholes_as_vol_of_vol_vanishes():
    # With rho = 0 the price correction is O(xi^2); with v0 = theta it vanishes in the limit.
    K = np.array([70.0, 100.0, 130.0])
    bs_price = bs.price(100.0, K, 1.0, 0.03, 0.2, 0.01)
    errs = []
    for xi in (1e-2, 1e-3):
        p = heston.HestonParams(v0=0.04, kappa=1.0, theta=0.04, xi=xi, rho=0.0)
        cf = partial(heston.char_func, T=1.0, r=0.03, q=0.01, p=p)
        errs.append(np.abs(cos.price(cf, 100.0, K, 1.0, 0.03, 0.01) - bs_price).max())
    assert errs[1] < 1e-5               # ~4e-6 at xi = 1e-3
    assert errs[1] < errs[0] / 50          # quadratic: xi down 10x -> error down ~100x


def test_heston_prices_are_arbitrage_free_in_strike():
    # Call prices must be decreasing and convex in K (Breeden-Litzenberger: density >= 0).
    K = np.linspace(60, 160, 101)
    cf = partial(heston.char_func, T=0.5, r=0.02, q=0.0, p=FO_PARAMS)
    c = cos.price(cf, 100.0, K, 0.5, 0.02)
    assert np.all(np.diff(c) < 0)
    assert np.all(np.diff(c, 2) > -1e-10)


def test_heston_negative_rho_gives_downward_skew():
    # Implied vol of a low strike should exceed that of a high strike when rho < 0.
    cf = partial(heston.char_func, T=1.0, r=0.0, q=0.0, p=FO_PARAMS)
    S, T = 100.0, 1.0
    ivs = [bs.implied_vol(cos.price(cf, S, K, T, 0.0, is_call=K >= S), S, K, T, 0.0, is_call=K >= S)
           for K in (80.0, 120.0)]
    assert ivs[0] > ivs[1]
