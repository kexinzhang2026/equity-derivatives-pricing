"""Crank-Nicolson engine: accuracy, convergence order, Rannacher smoothing, American exercise."""
import numpy as np
import pytest

from derivpricing.engines import pde
from derivpricing.models import black_scholes as bs

P = dict(S0=100.0, K=95.0, T=0.75, r=0.04, sigma=0.25, q=0.015)


@pytest.mark.parametrize("is_call", [True, False])
def test_european_matches_closed_form(is_call):
    res = pde.price_vanilla(**P, is_call=is_call, M=400, N=400)
    exact = bs.price(P["S0"], P["K"], P["T"], P["r"], P["sigma"], P["q"], is_call)
    g = bs.greeks(P["S0"], P["K"], P["T"], P["r"], P["sigma"], P["q"], is_call)
    assert res.price == pytest.approx(exact, abs=2e-4)
    assert res.delta == pytest.approx(g["delta"], abs=5e-5)
    assert res.gamma == pytest.approx(g["gamma"], abs=5e-7)


def test_second_order_convergence():
    # Refining space and time together (dt ∝ h), the error should fall 4x per halving.
    exact = bs.price(P["S0"], P["K"], P["T"], P["r"], P["sigma"], P["q"])
    Ms = [50, 100, 200, 400]
    errs = [abs(pde.price_vanilla(**P, M=M, N=M).price - exact) for M in Ms]
    order = np.polyfit(np.log(Ms), np.log(errs), 1)[0]
    assert order == pytest.approx(-2.0, abs=0.1)


def test_rannacher_removes_gamma_oscillations():
    # ATM, short maturity, few large time steps: plain CN leaves the payoff kink's
    # high-frequency error undamped and gamma at the strike is badly wrong.
    args = dict(S0=100.0, K=100.0, T=0.25, r=0.04, sigma=0.25, q=0.015, M=400, N=25)
    exact = bs.greeks(100.0, 100.0, 0.25, 0.04, 0.25, 0.015)["gamma"]
    err_cn = abs(pde.price_vanilla(**args, rannacher=False).gamma - exact)
    err_rn = abs(pde.price_vanilla(**args, rannacher=True).gamma - exact)
    assert err_rn < 1e-4 < 0.05 < err_cn


def crr_american_put(S0, K, T, r, sigma, n):
    """Independent reference: Cox-Ross-Rubinstein binomial tree with early exercise."""
    dt = T / n
    u = np.exp(sigma * np.sqrt(dt))
    p = (np.exp(r * dt) - 1 / u) / (u - 1 / u)
    disc = np.exp(-r * dt)
    V = np.maximum(K - S0 * u ** (n - 2.0 * np.arange(n + 1)), 0)
    for i in range(n - 1, -1, -1):
        V = np.maximum(disc * (p * V[:-1] + (1 - p) * V[1:]), K - S0 * u ** (i - 2.0 * np.arange(i + 1)))
    return V[0]


@pytest.mark.parametrize("S0", [36.0, 40.0, 44.0])
def test_american_put_matches_binomial(S0):
    # Accurate values ~4.4867 / 2.3196 / 1.1130. (Longstaff-Schwartz's Table 1 quotes
    # 4.478 / 2.314 / 1.110 from a coarser finite-difference grid.)
    res = pde.price_vanilla(S0, 40.0, 1.0, 0.06, 0.2, is_call=False, american=True, M=800, N=800)
    assert res.price == pytest.approx(crr_american_put(S0, 40.0, 1.0, 0.06, 0.2, 5000), abs=2e-3)
    assert res.price >= max(40.0 - S0, 0.0)
    assert res.price > bs.price(S0, 40.0, 1.0, 0.06, 0.2, is_call=False)


def test_american_call_without_dividends_equals_european():
    # Merton: never optimal to exercise a call early on a non-dividend-paying stock.
    am = pde.price_vanilla(100.0, 100.0, 1.0, 0.05, 0.2, q=0.0, american=True).price
    eu = bs.price(100.0, 100.0, 1.0, 0.05, 0.2)
    assert am == pytest.approx(eu, abs=1e-3)
