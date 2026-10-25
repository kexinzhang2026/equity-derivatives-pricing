"""Monte Carlo Greeks and delta-hedging simulations against known theory."""
import numpy as np
import pytest
from scipy.stats import norm

from derivpricing.engines import mc_greeks
from derivpricing.engines import monte_carlo as mc
from derivpricing.hedging.backtest import attribute_day, simulate_short_call_hedge
from derivpricing.models import black_scholes as bs

S, K, T, R, Q, SIG = 100.0, 105.0, 0.5, 0.04, 0.01, 0.25
N = 400_000


def test_pathwise_and_lr_delta_agree_with_closed_form():
    exact = bs.greeks(S, K, T, R, SIG, Q)["delta"]
    pw = mc_greeks.call_delta_pathwise(S, K, T, R, Q, SIG, N, np.random.default_rng(0))
    lr = mc_greeks.delta_likelihood_ratio(lambda x: np.maximum(x - K, 0), S, T, R, Q, SIG, N,
                                          np.random.default_rng(0))
    assert pw.contains(exact) and lr.contains(exact)
    assert pw.stderr < lr.stderr          # pathwise has lower variance when it applies


def test_lr_handles_digital_delta():
    # Pathwise gives exactly 0 for a digital; LR recovers e^{-rT} n(d2) / (S sigma sqrt T).
    d2 = (np.log(S / K) + (R - Q - 0.5 * SIG**2) * T) / (SIG * np.sqrt(T))
    exact = np.exp(-R * T) * norm.pdf(d2) / (S * SIG * np.sqrt(T))
    est = mc_greeks.delta_likelihood_ratio(lambda x: (x > K) * 1.0, S, T, R, Q, SIG, N,
                                           np.random.default_rng(1))
    assert est.contains(exact)


def test_mixed_gamma():
    exact = bs.greeks(S, K, T, R, SIG, Q)["gamma"]
    assert mc_greeks.gamma_lr_pathwise(K, S, T, R, Q, SIG, N, np.random.default_rng(2)).contains(exact)


def test_hedging_error_matches_kamal_derman():
    # Std of discrete delta-hedging P&L ≈ sqrt(pi/4) * vega * sigma / sqrt(N_rebalances)
    S0, K0, T0, sig = 100.0, 100.0, 0.25, 0.2
    vega = bs.greeks(S0, K0, T0, R, sig, Q)["vega"]
    for n in (40, 320):
        paths = mc.gbm_paths(S0, T0, R, Q, sig, n, 20_000, np.random.default_rng(n))
        pnl = simulate_short_call_hedge(paths, K0, T0, R, Q, sig)
        assert abs(pnl.mean()) < 4 * pnl.std() / np.sqrt(pnl.size)       # unbiased
        assert pnl.std() == pytest.approx(np.sqrt(np.pi / 4) * vega * sig / np.sqrt(n), rel=0.06)


def test_selling_rich_vol_earns_the_price_difference():
    # Sell at 20% implied, realised 15%: expected P&L = BS(20%) - BS(15%) (hedging at implied vol)
    S0, K0, T0 = 100.0, 100.0, 0.25
    paths = mc.gbm_paths(S0, T0, R, Q, 0.15, 252, 20_000, np.random.default_rng(3))
    pnl = simulate_short_call_hedge(paths, K0, T0, R, Q, 0.2)
    expected = bs.price(S0, K0, T0, R, 0.2, Q) - bs.price(S0, K0, T0, R, 0.15, Q)
    assert pnl.mean() == pytest.approx(expected, abs=4 * pnl.std() / np.sqrt(pnl.size))


def test_attribution_explains_a_small_move():
    # One day, small move and vol change: gamma/theta + vega should explain nearly all P&L.
    S0, S1, r, q, T0, dt = 4000.0, 4012.0, 0.045, 0.015, 30 / 365, 1 / 365
    F0, D0 = S0 * np.exp((r - q) * T0), np.exp(-r * T0)
    K0, iv0, iv1 = 4000.0, 0.18, 0.181
    V0 = bs.price(S0, K0, T0, r, iv0, q)
    V1 = bs.price(S1, K0, T0 - dt, r, iv1, q)
    a = attribute_day(S0, S1, F0, D0, K0, T0, dt, iv0, iv1, V0, V1)
    assert abs(a["residual"]) < 0.05 * (abs(a["gamma_theta"]) + abs(a["vega"]))


def test_full_revaluation_adds_up_and_vanishes_without_market_moves():
    """The three full-revaluation pieces sum to the total P&L exactly; with no vol move and a
    forward that follows the index at constant carry, the basis and vol pieces are zero."""
    from derivpricing.hedging.backtest import full_revaluation_day, _black
    S0, S1, K, r, q, T0, T1 = 4000.0, 4050.0, 4000.0, 0.04, 0.015, 30 / 365, 29 / 365
    F0, D0 = S0 * np.exp((r - q) * T0), np.exp(-r * T0)
    F1, D1 = S1 * np.exp((r - q) * T1), np.exp(-r * T1)
    V0, V1 = _black(F0, D0, K, T0, 0.18), _black(F1, D1, K, T1, 0.18)
    out = full_revaluation_day(S0, S1, F0, D0, F1, D1, K, T0, T1, 0.18, 0.18, V0, V1)
    assert abs(out["fr_check"]) < 1e-9
    assert abs(out["fr_basis"]) < 1e-9 and abs(out["fr_vol"]) < 1e-12
    # with a vol move and a basis jump the pieces still add up
    V1b = _black(F1 * 1.0005, D1, K, T1, 0.20)
    out = full_revaluation_day(S0, S1, F0, D0, F1 * 1.0005, D1, K, T0, T1, 0.18, 0.20, V0, V1b)
    assert abs(out["fr_check"]) < 1e-9 and out["fr_vol"] < 0 and out["fr_basis"] < 0
