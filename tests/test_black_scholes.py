"""Sanity tests for the Black-Scholes reference engine.

Every later engine (COS, Crank-Nicolson, Monte Carlo) is tested against this
module, so it has to be right first. Each test checks a property that holds
independently of the implementation, not a number copied from the code.
"""
import numpy as np
import pytest

from derivpricing.models import black_scholes as bs

PARAMS = dict(S=100.0, K=95.0, T=0.75, r=0.04, sigma=0.25, q=0.015)


def test_textbook_value():
    # Hull, Options Futures and Other Derivatives, Example 15.6:
    # S=42, K=40, r=10%, sigma=20%, T=0.5 -> c = 4.76, p = 0.81
    assert bs.price(42, 40, 0.5, 0.10, 0.20, is_call=True) == pytest.approx(4.76, abs=5e-3)
    assert bs.price(42, 40, 0.5, 0.10, 0.20, is_call=False) == pytest.approx(0.81, abs=5e-3)


@pytest.mark.parametrize("K", [60.0, 95.0, 100.0, 140.0])
def test_put_call_parity(K):
    # Model-free: C - P = S e^{-qT} - K e^{-rT}
    p = {**PARAMS, "K": K}
    c = bs.price(**p, is_call=True)
    put = bs.price(**p, is_call=False)
    S, T, r, q = p["S"], p["T"], p["r"], p["q"]
    assert c - put == pytest.approx(S * np.exp(-q * T) - K * np.exp(-r * T), abs=1e-12)


def test_no_arbitrage_bounds():
    S, K, T, r, q = 100.0, 100.0, 1.0, 0.03, 0.01
    for sigma in [0.01, 0.2, 1.0, 3.0]:
        c = bs.price(S, K, T, r, sigma, q)
        assert max(S * np.exp(-q * T) - K * np.exp(-r * T), 0) <= c <= S * np.exp(-q * T)


def test_zero_vol_limit_is_discounted_forward_intrinsic():
    S, K, T, r, q = 100.0, 90.0, 1.0, 0.05, 0.0
    c = bs.price(S, K, T, r, 1e-8, q)
    assert c == pytest.approx(S - K * np.exp(-r * T), abs=1e-8)


@pytest.mark.parametrize("is_call", [True, False])
def test_greeks_match_finite_differences(is_call):
    # Central differences have O(h^2) error, so h=1e-4 gives ~1e-8 accuracy.
    g = bs.greeks(**PARAMS, is_call=is_call)
    h = 1e-4

    def V(**bump):
        return bs.price(**{**PARAMS, **bump}, is_call=is_call)

    S, T, r, sig = PARAMS["S"], PARAMS["T"], PARAMS["r"], PARAMS["sigma"]
    fd = {
        "delta": (V(S=S + h) - V(S=S - h)) / (2 * h),
        "gamma": (V(S=S + h) - 2 * V() + V(S=S - h)) / h**2,
        "vega": (V(sigma=sig + h) - V(sigma=sig - h)) / (2 * h),
        "theta": -(V(T=T + h) - V(T=T - h)) / (2 * h),
        "rho": (V(r=r + h) - V(r=r - h)) / (2 * h),
    }
    for name, value in fd.items():
        # gamma uses a second difference: rounding error ~ eps/h^2, so looser tolerance
        tol = 1e-4 if name == "gamma" else 1e-6
        assert g[name] == pytest.approx(value, rel=tol, abs=tol), name


@pytest.mark.parametrize("sigma", [0.05, 0.2, 0.8])
@pytest.mark.parametrize("K", [70.0, 100.0, 130.0])
def test_implied_vol_roundtrip(sigma, K):
    # Invert on the out-of-the-money side, as with market data. A deep ITM option
    # with low vol has vega ~ 1e-9: its price barely moves with sigma, so the
    # inverse problem is ill-conditioned (K=70, sigma=5% fails on the call side).
    F = PARAMS["S"] * np.exp((PARAMS["r"] - PARAMS["q"]) * PARAMS["T"])
    is_call = K >= F
    p = {**PARAMS, "K": K, "sigma": sigma}
    target = bs.price(**p, is_call=is_call)
    iv = bs.implied_vol(target, p["S"], K, p["T"], p["r"], p["q"], is_call=is_call)
    assert iv == pytest.approx(sigma, abs=1e-8)


def test_implied_vol_rejects_arbitrage_price():
    # A call worth more than the (dividend-discounted) stock cannot be priced by any sigma.
    assert np.isnan(bs.implied_vol(150.0, 100.0, 100.0, 1.0, 0.0))


@pytest.mark.parametrize("is_call", [True, False])
def test_greeks_satisfy_bs_pde(is_call):
    # Theta + (r-q) S Delta + 1/2 sigma^2 S^2 Gamma = r V  (see derivation doc)
    g = bs.greeks(**PARAMS, is_call=is_call)
    V = bs.price(**PARAMS, is_call=is_call)
    S, r, q, sig = PARAMS["S"], PARAMS["r"], PARAMS["q"], PARAMS["sigma"]
    lhs = g["theta"] + (r - q) * S * g["delta"] + 0.5 * sig**2 * S**2 * g["gamma"]
    assert lhs == pytest.approx(r * V, abs=1e-12)
