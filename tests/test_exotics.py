"""Barrier options and the autocallable: closed form vs PDE vs Monte Carlo."""
import numpy as np
import pytest

from derivpricing.engines import monte_carlo as mc
from derivpricing.models import black_scholes as bs
from derivpricing.products import barrier
from derivpricing.products.autocallable import Autocallable, price_mc, price_pde

S, K, H, T, R, SIG, Q = 100.0, 100.0, 80.0, 1.0, 0.04, 0.25, 0.015


def test_barrier_limits():
    # Barrier far below: knock-out put = vanilla. In-out parity holds by construction.
    vanilla = bs.price(S, K, T, R, SIG, Q, is_call=False)
    assert barrier.down_and_out_put_bs(S, K, 1e-3, T, R, SIG, Q) == pytest.approx(vanilla, abs=1e-10)
    # Barrier just below spot: knock-out is almost surely triggered.
    assert barrier.down_and_out_put_bs(S, K, 99.99, T, R, SIG, Q) < 1e-2


def test_barrier_pde_second_order_with_aligned_grid():
    exact = barrier.down_and_out_put_bs(S, K, H, T, R, SIG, Q)
    Ms = [100, 200, 400]
    errs = [abs(barrier.down_and_out_put_pde(S, K, H, T, R, SIG, Q, M=M, N=M).price - exact) for M in Ms]
    assert errs[-1] < 1e-4
    assert np.polyfit(np.log(Ms), np.log(errs), 1)[0] == pytest.approx(-2.0, abs=0.15)


def test_discrete_barrier_mc_matches_bgk_corrected_closed_form():
    paths = mc.gbm_paths(S, T, R, Q, SIG, 252, 100_000, np.random.default_rng(1))
    est = barrier.down_and_in_put_mc(paths, K, H, R, T)
    bgk = barrier.down_and_in_put_bs(S, K, barrier.bgk_shift(H, SIG, 1 / 252), T, R, SIG, Q)
    continuous = barrier.down_and_in_put_bs(S, K, H, T, R, SIG, Q)
    assert est.contains(bgk)
    assert not est.contains(continuous)        # daily monitoring is measurably cheaper


@pytest.fixture(scope="module")
def spec():
    return Autocallable(T=3.0, obs_per_year=4, autocall_level=1.0, coupon=0.08, ki_level=0.6)


def test_autocallable_pde_converges_and_matches_mc(spec):
    S0, r, q, sig = 100.0, 0.04, 0.015, 0.2
    pde_prices = [price_pde(spec, S0, r, q, sig, M=M).price for M in (150, 300)]
    assert abs(pde_prices[1] - pde_prices[0]) < 0.02            # smoothed digitals: fast convergence
    paths = mc.gbm_paths(S0, spec.T, r, q, sig, spec.n_steps, 60_000, np.random.default_rng(3))
    est, diag = price_mc(spec, paths, r)
    assert est.contains(pde_prices[1])
    assert diag["p_call"].sum() + diag["p_ki_loss"] + diag["p_protected"] == pytest.approx(1.0)


def test_autocallable_limits(spec):
    # Zero vol, positive carry: the index drifts up and the note is called at the first date.
    S0, r, q = 100.0, 0.04, 0.0
    t1 = spec.obs_times[0]
    expected = 100 * (1 + spec.coupon * t1) * np.exp(-r * t1)
    # (Monte Carlo here: at vanishing vol the PDE is convection-dominated, a regime where
    # central differences oscillate and an upwind scheme would be needed.)
    paths = mc.gbm_paths(S0, spec.T, r, q, 1e-6, spec.n_steps, 10, np.random.default_rng(0))
    assert price_mc(spec, paths, r)[0].price == pytest.approx(expected, rel=1e-6)
    # Higher coupon must be worth more; lower knock-in barrier must be worth more to the holder.
    base = price_pde(spec, S0, 0.04, 0.015, 0.2, M=200).price
    richer = price_pde(Autocallable(coupon=0.10), S0, 0.04, 0.015, 0.2, M=200).price
    safer = price_pde(Autocallable(ki_level=0.5), S0, 0.04, 0.015, 0.2, M=200).price
    assert richer > base and safer > base
