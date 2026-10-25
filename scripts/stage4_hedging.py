"""Stage 4: where does delta-hedging error come from?

Part A (simulation, Black-Scholes world): discrete rebalancing, wrong volatility,
transaction costs.
Part B (real SPX data, optionsDX): every month of 2020 Q1 and 2023, sell a ~1-month
ATM SPX call at the market mid, delta-hedge daily with the index, re-mark daily at
market mids, and attribute each day's P&L to gamma/theta, vega and a residual.

    python scripts/stage4_hedging.py   ->  reports/stage4_hedging.md, figures
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from derivpricing.calibration.surface import implied_forward
from derivpricing.data import loaders
from derivpricing.engines import monte_carlo as mc
from derivpricing.hedging.backtest import attribute_day, full_revaluation_day, simulate_short_call_hedge
from derivpricing.models import black_scholes as bs

ROOT = Path(__file__).resolve().parents[1]
FIG, REP, DATA = ROOT / "reports" / "figures", ROOT / "reports", ROOT / "data" / "optionsdx"
BLUE, ORANGE, AQUA, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e"
plt.rcParams.update({"figure.dpi": 150, "font.size": 8.5, "axes.grid": True, "grid.color": "#ecebe7",
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "axes.edgecolor": MUTED, "lines.linewidth": 2})


# ---------------------------------------------------------------- Part A: simulation

def part_a():
    S0, K, T, r, q, sig = 100.0, 100.0, 0.25, 0.04, 0.01, 0.2
    vega = bs.greeks(S0, K, T, r, sig, q)["vega"]
    Ns = np.array([4, 8, 16, 32, 63, 126, 252, 504])
    std, cost_mean, cost_std = [], [], []
    for n in Ns:
        paths = mc.gbm_paths(S0, T, r, q, sig, int(n), 20_000, np.random.default_rng(int(n)))
        std.append(simulate_short_call_hedge(paths, K, T, r, q, sig).std())
        pc = simulate_short_call_hedge(paths, K, T, r, q, sig, cost=5e-4)      # 5 bp per trade
        cost_mean.append(-pc.mean())
        cost_std.append(pc.std())
    std, cost_mean, cost_std = map(np.array, (std, cost_mean, cost_std))
    rmse_cost = np.sqrt(cost_mean**2 + cost_std**2)

    # wrong vol: implied 20%, realised 15%, hedge at implied vs at realised
    paths = mc.gbm_paths(S0, T, r, q, 0.15, 252, 20_000, np.random.default_rng(7))
    pnl_imp = simulate_short_call_hedge(paths, K, T, r, q, 0.20)
    # hedging at realised vol: sell at the 20% price, hedge with 15% deltas
    pnl_real = simulate_short_call_hedge(paths, K, T, r, q, 0.15) + \
        (bs.price(S0, K, T, r, 0.20, q) - bs.price(S0, K, T, r, 0.15, q))
    expected = bs.price(S0, K, T, r, 0.20, q) - bs.price(S0, K, T, r, 0.15, q)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    ax = axes[0]
    ax.loglog(Ns, std, "o-", color=BLUE, label="simulated std of P&L")
    ax.loglog(Ns, np.sqrt(np.pi / 4) * vega * sig / np.sqrt(Ns), "--", color=MUTED, lw=1,
              label="√(π/4)·vega·σ/√N")
    ax.set_xlabel("rebalances over 3 months")
    ax.set_ylabel("P&L std (per 100 notional)")
    ax.set_title("Discrete hedging error ∝ 1/√N", loc="left")
    ax.legend()
    ax = axes[1]
    ax.semilogx(Ns, std, "o-", color=BLUE, label="no costs: std")
    ax.semilogx(Ns, cost_mean, "o-", color=ORANGE, label="5 bp costs: mean cost")
    ax.semilogx(Ns, rmse_cost, "o-", color=INK, label="5 bp costs: RMS P&L")
    best = Ns[np.argmin(rmse_cost)]
    ax.axvline(best, color=MUTED, lw=1, ls=":")
    ax.set_xlabel("rebalances over 3 months")
    ax.set_title(f"With costs there is an optimal frequency (~{best})", loc="left")
    ax.legend()
    ax = axes[2]
    bins = np.linspace(min(pnl_imp.min(), pnl_real.min()), max(pnl_imp.max(), pnl_real.max()), 60)
    ax.hist(pnl_imp, bins=bins, color=BLUE, alpha=0.7, label="hedge at implied vol (20%)")
    ax.hist(pnl_real, bins=bins, color=ORANGE, alpha=0.7, label="hedge at realised vol (15%)")
    ax.axvline(expected, color=INK, lw=1, ls="--")
    ax.set_xlabel("P&L of selling at 20%, realised 15%")
    ax.set_title("Same mean, very different paths", loc="left")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "hedging_simulation.png")
    plt.close(fig)
    return {"std_63": float(std[Ns == 63][0]), "kd_63": float(np.sqrt(np.pi / 4) * vega * sig / np.sqrt(63)),
            "best_n": int(best), "imp_mean": pnl_imp.mean(), "imp_std": pnl_imp.std(),
            "real_mean": pnl_real.mean(), "real_std": pnl_real.std(), "expected": expected}


# ---------------------------------------------------------------- Part B: real SPX backtest

def load_period(months):
    frames = []
    for ym in months:
        df = loaders.load_optionsdx(DATA / f"spx_eod_{ym}.txt.gz")
        dte = (df["expiry"] - df["quote_date"]).dt.days
        df = df[(dte >= 0) & (dte <= 50) & ((df["strike"] / df["spot"] - 1).abs() < 0.15)]
        df = df[(df["bid"] > 0) & (df["ask"] > df["bid"])]
        frames.append(df.assign(mid=0.5 * (df["bid"] + df["ask"])))
    return pd.concat(frames, ignore_index=True)


def run_trade(data, entry_date, max_month_end):
    """Sell the ~30-day ATM call listed on entry_date and hedge it daily until the last
    quote date before expiry (closing out at mid avoids the AM-settlement fixing)."""
    day0 = data[data["quote_date"] == entry_date]
    dte = (day0["expiry"] - day0["quote_date"]).dt.days
    cand = day0[(dte >= 21) & (dte <= 45)]
    if cand.empty:
        return None
    expiry = cand.loc[(dte[cand.index] - 30).abs().idxmin(), "expiry"]
    S_entry = day0["spot"].iloc[0]
    F0, _, _ = implied_forward(day0[day0["expiry"] == expiry], S_entry)
    calls0 = day0[(day0["expiry"] == expiry) & day0["is_call"]]
    K = calls0.loc[(calls0["strike"] - F0).abs().idxmin(), "strike"]

    opt = data[(data["expiry"] == expiry) & (data["strike"] == K) & data["is_call"]].set_index("quote_date")
    days = sorted(d for d in opt.index.unique() if entry_date <= d < expiry)
    rows = []
    for d0, d1 in zip(days[:-1], days[1:]):
        chain0 = data[(data["quote_date"] == d0) & (data["expiry"] == expiry)]
        chain1 = data[(data["quote_date"] == d1) & (data["expiry"] == expiry)]
        S0, S1 = chain0["spot"].iloc[0], chain1["spot"].iloc[0]
        F0, D0, _ = implied_forward(chain0, S0, band=0.05)
        F1, D1, _ = implied_forward(chain1, S1, band=0.05)
        if not (np.isfinite(F0) and np.isfinite(F1)):
            continue
        T0 = (expiry - d0).days / 365
        T1 = (expiry - d1).days / 365
        V0, V1 = opt.loc[d0, "mid"], opt.loc[d1, "mid"]
        iv0 = bs.implied_vol(V0, D0 * F0, K, T0, -np.log(D0) / T0)
        iv1 = bs.implied_vol(V1, D1 * F1, K, T1, -np.log(D1) / T1)
        if not (np.isfinite(iv0) and np.isfinite(iv1)):
            continue
        a = attribute_day(S0, S1, F0, D0, K, T0, (d1 - d0).days / 365, iv0, iv1, V0, V1)
        fr = full_revaluation_day(S0, S1, F0, D0, F1, D1, K, T0, T1, iv0, iv1, V0, V1)
        rows.append({"date": d1, "S": S1, "iv": iv1, **a, **fr, "F0": F0, "F1": F1, "D0": D0, "D1": D1,
                     "weekday_gap": (d1 - d0).days,
                     "half_spread": 0.5 * (opt.loc[d0, "ask"] - opt.loc[d0, "bid"])})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df["entry"], df["expiry"], df["K"] = entry_date, expiry, K
    df["premium"] = opt.loc[entry_date, "mid"]
    df["iv_entry"] = bs.implied_vol(opt.loc[entry_date, "mid"], S_entry, K,
                                    (expiry - entry_date).days / 365, 0.0)
    return df


def part_b():
    periods = {"2020 Q1": ["202001", "202002", "202003"],
               "2023": [f"2023{m:02d}" for m in range(1, 13)]}
    trades = []
    for name, months in periods.items():
        data = load_period(months)
        dates = pd.Series(sorted(data["quote_date"].unique()))
        entries = dates.groupby(dates.dt.to_period("M")).first()
        for entry in entries:
            t = run_trade(data, entry, None)
            if t is not None:
                t["period"] = name
                trades.append(t)
    daily = pd.concat(trades, ignore_index=True)
    daily.to_csv(REP / "stage4_daily_attribution.csv", index=False)

    per_trade = daily.groupby(["period", "entry"]).agg(
        expiry=("expiry", "first"), K=("K", "first"), premium=("premium", "first"),
        iv_entry=("iv_entry", "first"), days=("total", "size"), total=("total", "sum"),
        gamma_theta=("gamma_theta", "sum"), vega=("vega", "sum"), vanna_volga=("vanna_volga", "sum"),
        residual=("residual", "sum"), fr_spot_time=("fr_spot_time", "sum"), fr_vol=("fr_vol", "sum"),
        fr_basis=("fr_basis", "sum"),
        realised_vol=("S", lambda s: np.std(np.diff(np.log(s.to_numpy()))) * np.sqrt(252)))
    per_trade.to_csv(REP / "stage4_trades.csv")

    # Figure: per-trade decomposition by full revaluation (the three pieces add up to the total)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1, 3.2]})
    for ax, (name, grp) in zip(axes, per_trade.groupby(level=0, sort=False)):
        x = np.arange(len(grp))
        labels = [pd.Timestamp(e).strftime("%b") for e in grp.index.get_level_values(1)]
        w = 0.27
        ax.bar(x - w, grp["fr_spot_time"], w, color=BLUE, label="spot + time at fixed vol (gamma/theta)")
        ax.bar(x, grp["fr_vol"], w, color=ORANGE, label="implied vol re-marks")
        ax.bar(x + w, grp["fr_basis"], w, color=MUTED, label="forward-index basis")
        ax.plot(x, grp["total"], "D", color=INK, ms=5, label="total hedged P&L")
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xticks(x, labels)
        ax.set_title(name, loc="left")
        ax.set_ylabel("index points per short call")
    axes[1].legend(fontsize=7, ncol=2, loc="lower left")
    fig.suptitle("Short 1-month ATM SPX call, delta-hedged daily: full-revaluation P&L attribution per trade",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "hedging_spx_attribution.png")
    plt.close(fig)
    return daily, per_trade


if __name__ == "__main__":
    a = part_a()
    daily, trades = part_b()
    agg = trades.groupby(level=0, sort=False)[["premium", "total", "gamma_theta", "vega", "vanna_volga",
                                               "residual", "fr_spot_time", "fr_vol", "fr_basis"]].sum()
    ho = daily.groupby("period", sort=False)[["higher_order_spot", "higher_order_vol", "fr_basis", "residual"]].sum()
    daily["days_left"] = (daily["expiry"] - daily["date"]).dt.days
    last3 = daily[daily["days_left"] <= 3].groupby("period", sort=False)["higher_order_vol"].sum()
    hit = trades.groupby(level=0, sort=False)["total"].apply(lambda s: (s > 0).mean())
    iv_rv = trades.groupby(level=0, sort=False)[["iv_entry", "realised_vol"]].mean()
    r2 = 1 - (daily["residual"].var() / daily["total"].var())
    r2_first = 1 - ((daily["residual"] + daily["vanna_volga"]).var() / daily["total"].var())
    spread_cost = trades.reset_index().merge(
        daily.groupby("entry")["half_spread"].first().rename("hs"), left_on="entry", right_index=True)
    lines = ["# Stage 4 · Greeks and hedging: where the hedging error comes from", "",
             "## A. Simulation (Black-Scholes world, 3-month ATM call, σ = 20%)",
             f"- Discrete hedging, 63 daily rebalances: P&L std {a['std_63']:.3f} vs Kamal–Derman "
             f"√(π/4)·vega·σ/√N = {a['kd_63']:.3f} (per 100 notional). Halving the error needs 4× the trades.",
             f"- 5 bp transaction costs: cost grows like √N while hedging error falls like 1/√N; "
             f"RMS P&L is minimised around N ≈ {a['best_n']} rebalances.",
             f"- Selling at 20% implied when realised is 15%: expected profit {a['expected']:.3f}. "
             f"Hedging with implied-vol deltas: mean {a['imp_mean']:.3f}, std {a['imp_std']:.3f}. "
             f"Hedging with realised-vol deltas: mean {a['real_mean']:.3f}, std {a['real_std']:.3f}.",
             "  Same expected profit; with realised-vol deltas the P&L is locked in (but you must know realised vol), "
             "with implied-vol deltas it accrues as ½Γ S²(σ_imp² − σ_real²) and depends on the path of gamma.",
             "", "![](figures/hedging_simulation.png)", "",
             "## B. Real SPX: short 1-month ATM call each month, hedged daily",
             "Entry on the first trading day of each month at mid; strike nearest the implied forward; "
             "expiry nearest 30 days; daily re-marks at mid; closed out on the last quote date before expiry. "
             "Index points per one call (SPX multiplier not applied).", "",
             "| Period | Trades | Premium collected | Hedged P&L | gamma/theta | vega | vanna/volga | residual | Winning trades | Mean entry IV | Mean realised vol |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for p in agg.index:
        n = int((trades.index.get_level_values(0) == p).sum())
        lines.append(f"| {p} | {n} | {agg.loc[p, 'premium']:.1f} | {agg.loc[p, 'total']:.1f} | "
                     f"{agg.loc[p, 'gamma_theta']:.1f} | {agg.loc[p, 'vega']:.1f} | {agg.loc[p, 'vanna_volga']:.1f} | "
                     f"{agg.loc[p, 'residual']:.1f} | "
                     f"{hit[p]:.0%} | {iv_rv.loc[p, 'iv_entry']:.1%} | {iv_rv.loc[p, 'realised_vol']:.1%} |")
    r2_p = {p: (1 - ((g["residual"] + g["vanna_volga"]).var() / g["total"].var()),
                1 - g["residual"].var() / g["total"].var(), g["fr_basis"].var() / g["total"].var())
            for p, g in daily.groupby("period", sort=False)}
    lines += ["", f"Daily attribution quality, both periods pooled: gamma/theta + vega explain {r2_first:.0%} of the "
              f"variance of daily hedged P&L; adding vanna/volga, {r2:.0%}. The pooled figure is dominated by the "
              "large 2020 days. By period:", "",
              "| Period | gamma/theta + vega | + vanna/volga | variance share of forward-index basis |", "|---|---|---|---|"]
    lines += [f"| {p} | {a1:.0%} | {a2:.0%} | {b:.0%} |" for p, (a1, a2, b) in r2_p.items()]
    lines += ["",
              "### Cumulative attribution by full revaluation",
              "Each day the option is repriced in three steps: (a) time passes and the index moves, at fixed implied vol, "
              "with the forward rolling at constant carry; (b) the forward and discount factor jump to their own day-1 "
              "values (forward-index basis); (c) implied vol moves to its day-1 value. The pieces add up to the total exactly.", "",
              "| Period | Hedged P&L | spot + time (fixed vol) | implied vol re-marks | forward-index basis |",
              "|---|---|---|---|---|"]
    for p in agg.index:
        lines.append(f"| {p} | {agg.loc[p, 'total']:.1f} | {agg.loc[p, 'fr_spot_time']:.1f} | "
                     f"{agg.loc[p, 'fr_vol']:.1f} | {agg.loc[p, 'fr_basis']:.1f} |")
    lines += ["", "Where the Taylor residual comes from (index points, sums over all days):", "",
              "| Period | Taylor residual | higher-order spot (full reval − gamma/theta) | higher-order vol (full reval − vega − vanna/volga) | of which last 3 days to expiry | forward-index basis |",
              "|---|---|---|---|---|---|"]
    for p in ho.index:
        lines.append(f"| {p} | {ho.loc[p, 'residual']:.1f} | {ho.loc[p, 'higher_order_spot']:.1f} | "
                     f"{ho.loc[p, 'higher_order_vol']:.1f} | {last3.get(p, 0.0):.1f} | {ho.loc[p, 'fr_basis']:.1f} |")
    lines += ["", "![](figures/hedging_spx_attribution.png)", ""]
    (REP / "stage4_hedging.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(trades.round(2).to_string())
