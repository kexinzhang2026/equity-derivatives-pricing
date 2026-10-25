# Stage 4 · Greeks and hedging: where the hedging error comes from

## A. Simulation (Black-Scholes world, 3-month ATM call, σ = 20%)
- Discrete hedging, 63 daily rebalances: P&L std 0.434 vs Kamal–Derman √(π/4)·vega·σ/√N = 0.441 (per 100 notional). Halving the error needs 4× the trades.
- 5 bp transaction costs: cost grows like √N while hedging error falls like 1/√N; RMS P&L is minimised around N ≈ 252 rebalances.
- Selling at 20% implied when realised is 15%: expected profit 0.986. Hedging with implied-vol deltas: mean 0.987, std 0.354. Hedging with realised-vol deltas: mean 0.986, std 0.163.
  Same expected profit; with realised-vol deltas the P&L is locked in (but you must know realised vol), with implied-vol deltas it accrues as ½Γ S²(σ_imp² − σ_real²) and depends on the path of gamma.

![](figures/hedging_simulation.png)

## B. Real SPX: short 1-month ATM call each month, hedged daily
Entry on the first trading day of each month at mid; strike nearest the implied forward; expiry nearest 30 days; daily re-marks at mid; closed out on the last quote date before expiry. Index points per one call (SPX multiplier not applied).

| Period | Trades | Premium collected | Hedged P&L | gamma/theta | vega | vanna/volga | residual | Winning trades | Mean entry IV | Mean realised vol |
|---|---|---|---|---|---|---|---|---|---|---|
| 2020 Q1 | 3 | 189.8 | -127.8 | -96.4 | -65.5 | 10.6 | 23.5 | 33% | 17.2% | 36.3% |
| 2023 | 12 | 826.7 | 31.8 | 73.6 | -75.6 | 2.1 | 31.7 | 58% | 15.6% | 12.1% |

Daily attribution quality: gamma/theta + vega explain 86% of the variance of daily hedged P&L; adding vanna/volga, 91%.

![](figures/hedging_spx_attribution.png)

## Reading the results
- **2023 is the variance risk premium at work.** Entry implied vol averaged 15.6% against 12.1% realised, so the gamma/theta term earned +73.6 points over 12 trades. Most of it was given back through vega (−75.6): the position is a fixed strike, and in a rising market (2023) that strike slides down the skew (K/S falls), so its implied vol rises even when the smile itself does not move. This is the sticky-strike vs sticky-moneyness question; a smile-adjusted delta (adding vega × dσ/dS from the skew) would move part of this from the vega line into the hedge. Net hedged P&L: +31.8 points on 826.7 points of premium, 7 of 12 trades profitable.
- **2020 Q1 is the tail.** Realised vol of 36% against 17% implied: gamma/theta −96.4 points and vega −65.5 in three trades. The March trade is cut at 2020-03-31 (the data loaded ends there), after 8 days, so its loss is a lower bound on the full-month loss.
- **What the hedge cannot remove.** A daily delta hedge neutralises the first-order spot exposure only. What remains is exactly the second-order exposure the Black-Scholes PDE prices in: gamma (realised vs implied variance) and vega (implied vol re-marks). These two explain 86% of daily P&L variance; adding the vanna (spot × vol) and volga (vol × vol) terms raises it to 91%.
- **Open item.** The remaining residual is small day by day but positive on average (+0.13 points per day in 2023), concentrated on large moves (the two biggest are 2020-03-09 and 2020-02-27). Candidates: third-order spot terms (speed) on fat-tailed daily returns, and the bid–ask noise of daily mids. A full-revaluation P&L explain (reprice at day-1 spot and vol, holding other inputs fixed) would separate them.
- **Caveats.** No transaction costs: selling at the bid instead of mid would cost roughly half the quoted spread per trade, and index futures hedging adds a few basis points per rebalance. Index points are per one call, before the SPX ×100 multiplier.
