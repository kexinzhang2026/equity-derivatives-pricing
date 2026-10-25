"""Stage 1 report: three methods, one price. Writes the cross-validation table and figures.

    python scripts/stage1_crossval.py
    -> reports/stage1_crossvalidation.md, reports/figures/*.png
"""
from __future__ import annotations

from functools import partial
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from derivpricing.engines import cos, lsm, pde
from derivpricing.engines import monte_carlo as mc
from derivpricing.models import black_scholes as bs
from derivpricing.models import heston

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, AQUA, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e"
plt.rcParams.update({
    "figure.dpi": 150, "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": "#e6e5e1",
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2, "lines.markersize": 5, "legend.frameon": False,
})

S, K, T, r, SIG, Q = 100.0, 95.0, 0.75, 0.04, 0.25, 0.015
FO = heston.HestonParams(v0=0.0175, kappa=1.5768, theta=0.0398, xi=0.5751, rho=-0.5711)
rng = np.random.default_rng(2026)


def fmt_mc(res):
    return f"{res.price:.4f} ± {res.stderr:.4f}"


def cross_validation_table():
    rows = []
    # 1. European call under Black-Scholes
    exact = bs.price(S, K, T, r, SIG, Q)
    c_cos = cos.price(partial(bs.char_func, T=T, r=r, q=Q, sigma=SIG), S, K, T, r, Q)
    c_pde = pde.price_vanilla(S, K, T, r, SIG, Q, M=800, N=800).price
    paths = mc.gbm_paths(S, T, r, Q, SIG, 1, 1_000_000, rng)
    ST = paths[:, -1]
    c_mc = mc.control_variate(np.exp(-r * T) * np.maximum(ST - K, 0), np.exp(-r * T) * ST,
                              S * np.exp(-Q * T))
    rows.append(("European call, Black-Scholes", f"{exact:.4f}", f"{c_cos:.4f}",
                 f"{c_pde:.4f}", fmt_mc(c_mc)))

    # 2. European call under Heston (Fang-Oosterlee parameters, Feller violated)
    h_cos = cos.price(partial(heston.char_func, T=1.0, r=0.0, q=0.0, p=FO), 100.0, 100.0, 1.0, 0.0,
                      N=512, L=12)
    Sh, _ = mc.heston_paths_qe(100.0, 1.0, 0.0, 0.0, FO, 64, 500_000, rng)
    h_mc = mc.control_variate(np.maximum(Sh[:, -1] - 100.0, 0), Sh[:, -1], 100.0)
    rows.append(("European call, Heston (S=K=100, T=1, r=q=0)", "n/a", f"{h_cos:.4f}", "(2D ADI: not built)", fmt_mc(h_mc)))

    # 3. American put (Longstaff-Schwartz test case)
    a_pde = pde.price_vanilla(36.0, 40.0, 1.0, 0.06, 0.2, is_call=False, american=True,
                              M=1600, N=1600).price
    put = lambda x: np.maximum(40.0 - x, 0.0)
    p_eval = mc.gbm_paths(36.0, 1.0, 0.06, 0.0, 0.2, 50, 200_000, rng)
    p_fit = mc.gbm_paths(36.0, 1.0, 0.06, 0.0, 0.2, 50, 200_000, rng)
    a_lsm = lsm.price(p_eval, put, 0.06, 1.0, scale=40.0, fit_paths=p_fit)
    rows.append(("American put (S=36, K=40)", "none exists", "n/a", f"{a_pde:.4f}",
                 fmt_mc(a_lsm) + " (50 dates)"))

    lines = ["| Product | Closed form | COS (Fourier) | Crank–Nicolson PDE | Monte Carlo (± 1 s.e.) |",
             "|---|---|---|---|---|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def fig_pde_convergence():
    exact = bs.price(S, K, T, r, SIG, Q)
    Ms = np.array([25, 50, 100, 200, 400, 800, 1600])
    err_rn = [abs(pde.price_vanilla(S, K, T, r, SIG, Q, M=M, N=M).price - exact) for M in Ms]
    slope = np.polyfit(np.log(Ms), np.log(err_rn), 1)[0]
    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.loglog(Ms, err_rn, "o-", color=BLUE, label=f"Crank–Nicolson + Rannacher (slope {slope:.2f})")
    ax.loglog(Ms, err_rn[0] * (Ms / Ms[0]) ** -2.0, "--", color=MUTED, lw=1, label="reference slope −2")
    ax.set_xlabel("grid points M (time steps N = M)")
    ax.set_ylabel("|price error|")
    ax.set_title("PDE converges at second order", loc="left", color=INK)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "pde_convergence.png")
    plt.close(fig)
    return slope


def fig_rannacher_gamma():
    args = dict(S0=100.0, K=100.0, T=0.25, r=0.04, sigma=0.25, q=0.015, M=400, N=25)
    cn = pde.price_vanilla(**args, rannacher=False)
    rn = pde.price_vanilla(**args, rannacher=True)

    def grid_gamma(res):
        x = np.log(res.S)
        h = x[1] - x[0]
        Vx = (res.V[2:] - res.V[:-2]) / (2 * h)
        Vxx = (res.V[2:] - 2 * res.V[1:-1] + res.V[:-2]) / h**2
        return res.S[1:-1], (Vxx - Vx) / res.S[1:-1] ** 2

    fig, ax = plt.subplots(figsize=(5, 3.4))
    for res, color, label in [(cn, ORANGE, "plain Crank–Nicolson"), (rn, BLUE, "with Rannacher start")]:
        Sg, g = grid_gamma(res)
        m = (Sg > 85) & (Sg < 115)
        ax.plot(Sg[m], g[m], color=color, label=label, lw=1.5)
    Sg = np.linspace(85, 115, 300)
    ax.plot(Sg, bs.greeks(Sg, 100.0, 0.25, 0.04, 0.25, 0.015)["gamma"], "--", color=INK, lw=1,
            label="exact")
    ax.set_xlabel("spot S")
    ax.set_ylabel("gamma")
    ax.set_title("Rannacher start removes gamma oscillations at the strike", loc="left", color=INK)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "rannacher_gamma.png")
    plt.close(fig)


def fig_cos_convergence():
    Ns = np.array([8, 16, 24, 32, 48, 64, 96, 128, 192, 256])
    bs_cf = partial(bs.char_func, T=T, r=r, q=Q, sigma=SIG)
    bs_err = [abs(cos.price(bs_cf, S, K, T, r, Q, N=N) - bs.price(S, K, T, r, SIG, Q)) for N in Ns]
    h_cf = partial(heston.char_func, T=1.0, r=0.0, q=0.0, p=FO)
    h_ref = cos.price(h_cf, 100.0, 100.0, 1.0, 0.0, N=2048, L=12)
    h_err = [abs(cos.price(h_cf, 100.0, 100.0, 1.0, 0.0, N=N, L=12) - h_ref) for N in Ns]
    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.semilogy(Ns, np.maximum(bs_err, 1e-16), "o-", color=BLUE, label="Black-Scholes")
    ax.semilogy(Ns, np.maximum(h_err, 1e-16), "o-", color=ORANGE, label="Heston (Feller violated)")
    ax.set_xlabel("number of cosine terms N")
    ax.set_ylabel("|price error|  (log scale)")
    ax.set_title("COS error falls exponentially in N", loc="left", color=INK)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "cos_convergence.png")
    plt.close(fig)


def fig_mc_variance_reduction():
    Ns = np.array([1_000, 4_000, 16_000, 64_000, 256_000])
    out = {"plain": [], "antithetic": [], "control variate": []}
    D = np.exp(-r * T)
    for n in Ns:
        ST = mc.gbm_paths(S, T, r, Q, SIG, 1, n, rng)[:, -1]
        out["plain"].append(mc.plain(D * np.maximum(ST - K, 0)).stderr)
        STa = mc.gbm_paths(S, T, r, Q, SIG, 1, n, rng, antithetic=True)[:, -1]
        out["antithetic"].append(mc.antithetic(D * np.maximum(STa - K, 0)).stderr)
        out["control variate"].append(
            mc.control_variate(D * np.maximum(ST - K, 0), D * ST, S * np.exp(-Q * T)).stderr)
    fig, ax = plt.subplots(figsize=(5, 3.4))
    for (name, se), color in zip(out.items(), [MUTED, ORANGE, BLUE]):
        ax.loglog(Ns, se, "o-", color=color, label=name)
    ax.set_xlabel("number of paths")
    ax.set_ylabel("standard error")
    ax.set_title("All estimators converge like 1/√n; variance reduction shifts the line down",
                 loc="left", color=INK, fontsize=8.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "mc_variance_reduction.png")
    plt.close(fig)
    return {k: v[-1] for k, v in out.items()}


if __name__ == "__main__":
    table = cross_validation_table()
    slope = fig_pde_convergence()
    fig_rannacher_gamma()
    fig_cos_convergence()
    se = fig_mc_variance_reduction()
    vr_cv = (se["plain"] / se["control variate"]) ** 2
    vr_av = (se["plain"] / se["antithetic"]) ** 2
    report = f"""# Stage 1 · Three methods, one price

{table}

- Black-Scholes call: S={S}, K={K}, T={T}, r={r}, σ={SIG}, q={Q}.
- Heston call: the Fang & Oosterlee (2008) test case, a different contract: S=100, K=100, T=1, r=0, q=0;
  v0=0.0175, κ=1.5768, θ=0.0398, ξ=0.5751, ρ=−0.5711. Published reference value 5.785155450.
- American put: S=36, K=40, T=1, r=0.06, σ=0.2 (Longstaff & Schwartz 2001).
- American put: the PDE value is continuous exercise; LSM uses 50 exercise dates and an
  out-of-sample rule, so it is a lower bound for the 50-date Bermudan, itself below the American.

## Convergence
- **PDE**: fitted log-log slope {slope:.2f} (theory −2). `figures/pde_convergence.png`
- **Rannacher**: with 25 time steps, plain CN gamma oscillates at the strike. `figures/rannacher_gamma.png`
- **COS**: error reaches machine precision by N≈64 for Black-Scholes. `figures/cos_convergence.png`
- **Monte Carlo** at 256k paths: variance reduction factor {vr_av:.1f}x (antithetic),
  {vr_cv:.1f}x (control variate e^(-rT)S_T). `figures/mc_variance_reduction.png`
"""
    (ROOT / "reports" / "stage1_crossvalidation.md").write_text(report)
    print(report)
