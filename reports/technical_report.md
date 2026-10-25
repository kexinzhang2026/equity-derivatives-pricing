# Cross-validated equity derivatives pricing on SPX: methods, calibration, model risk and hedging

*Technical report. Code, tests and figures: this repository. Every formula used in the code is derived in `docs/derivations/01`–`10`.*

## Summary
1. **Three independent pricing methods agree.** A Fourier (COS) engine, a Crank–Nicolson PDE engine and a Monte Carlo engine price the same European, Heston and American contracts to within discretisation or statistical error. Convergence rates are measured, not assumed: PDE order 2.00, exponential COS convergence, Monte Carlo error ∝ 1/√n.
2. **Calibration to SPX (optionsDX end-of-day, 2020 Q1 and 2023).** One Heston parameter set fits a day's surface to about 0.8 implied-vol points but cannot reproduce the 1-month skew; SABR, fitted per expiry, reaches 0.43. On the 2023-01-18 surface κ is only weakly identified (any κ in [1.2, 3.0] fits within 0.1 vol pt of the optimum, with ξ compensating). Across 52 weekly fits in 2023, ρ is stable (CV 3%) while κ and ξ move together (correlation of weekly changes 0.96). In March 2020 the fit error rises to 2.7 vol points.
3. **Model risk on an autocallable.** With BS, Heston and Dupire local vol all fitted to the same SPX surface, the fair coupon of a 3-year autocallable with a 60% knock-in is 14.0%, 15.2% and 16.0%. The models do not reprice vanillas equally well (implied-vol RMSE 8.9, 0.83 and 0.61 vol pt), so a control model is used: local vol built from Heston's own smiles (15.94%). It attributes +0.77 of the 0.78-point Heston–local vol gap to dynamics and +0.01 to calibration (MC s.e. about 0.19), consistent with their different forward smiles. One date, one product.
4. **Hedging.** Discrete-hedging error matches the Kamal–Derman formula. On real SPX data, short 1-month ATM calls hedged daily earned +31.8 points in 2023 and lost 127.8 in 2020 Q1. An exact full-revaluation split gives, for 2023, +74.2 from realised vs implied variance, −34.2 from implied-vol re-marks and −8.2 from forward–index basis. The Taylor (Greek) terms explain 97% (2020 Q1) and 84% (2023) of daily P&L variance, but their 2023 residual (+31.7) holds almost the whole cumulative profit; most of it is the failure of the vega expansion in the last three days before expiry.

## 1. Pricing engines and cross-validation
| Engine | Method | Validation |
|---|---|---|
| Closed form | Black-Scholes, Greeks, implied vol | Hull textbook values, put-call parity, no-arbitrage bounds, Greeks vs finite differences, Greeks satisfy the BS PDE |
| Fourier | Heston characteristic function ("little Heston trap" form), COS method | Agrees with the Lewis (2001) integral by adaptive quadrature to 1e-9; reduces to BS at rate O(ξ²) as ξ → 0 |
| PDE | Crank–Nicolson in log-price, Rannacher start, cell-averaged payoff, penalty method for American exercise, local-vol coefficients | Order 2.00 on Europeans; American put equals a 20,000-step binomial tree |
| Monte Carlo | Exact GBM, antithetic and control variates, Andersen QE for Heston, Longstaff–Schwartz, local-vol Euler | Within 0.3 s.e. of the reference on the European test contracts; LSM is below the PDE American value, as a lower bound should be |

| Product | Reference | COS | Crank–Nicolson | Monte Carlo (± 1 s.e.) |
|---|---|---|---|---|
| European call, BS (S=100, K=95, T=0.75, r=4%, q=1.5%, σ=25%) | 12.0056 (closed form) | 12.0056 | 12.0057 | 12.0042 ± 0.0058 |
| European call, Heston (Fang–Oosterlee case: S=K=100, T=1, r=q=0; Feller violated) | 5.7852 (published) | 5.7852 | – | 5.7838 ± 0.0071 |
| American put (S=36, K=40, T=1, r=6%, σ=20%) | 4.4867 (binomial tree) | – | 4.4867 | 4.4758 ± 0.0065 (50 dates) |

**Numerical findings.**
- **Rannacher start-up:** without it, Crank–Nicolson leaves the payoff kink's high-frequency error undamped, because its amplification factor tends to −1 for stiff modes. Gamma at the strike is then off by 0.2; with the start-up the error is 1e-5.
- **Barrier on the grid:** placing the knock-out barrier exactly on a node turns first-order convergence into second order.
- **Autocallable discontinuities:** averaging the node at the autocall level and the node at the knock-in barrier (both are jumps in value) turns first-order convergence into near-exact convergence by M = 300.
- **Published benchmarks have errors too:** the Longstaff–Schwartz paper's finite-difference benchmark (4.478) is off by 0.009; the accurate value is 4.4867.

## 2. Market data and calibration
**Cleaning.** Two-sided quotes only, spread ≤ 50% of mid, T ≥ 7 days, OTM options only. The forward and discount factor per expiry come from a regression of C − P on K (put-call parity), so neither a stale spot nor an assumed dividend yield enters.

**Objective.** Vega-weighted price errors, Σ((C_model − C_mkt)/vega)². These are implied-vol errors to first order, with no root-finding inside the optimiser. Bounded trust-region least squares runs from several starting points.

**Validation first.** On synthetic Heston surfaces, the calibrator recovers the parameters to 1e-4. With 0.2 vol points of noise added, the fit stops at the noise level (RMSE 0.16 vol points). A κ profile shows that on short-dated surfaces κ is weakly identified and ξ compensates for it.

**SPX results.**
- **Heston fit by expiry:** error falls with maturity, from 1.48 vol points at 1 month to about 0.5 at 2 years. The residual map is structured, not noise: a diffusive stochastic-vol model cannot produce the steep short-dated skew, which motivates jumps or rough volatility.
- **SABR:** ν decays with maturity (2.25 at 1 month to 0.58 at 2 years), so a per-expiry fit absorbs the term structure into its parameters.
- **Local vol skew:** Dupire local vol from SVI slices has a skew 1.95 times the implied skew at the money, close to the rule of thumb of 2.
- **Identifiability on one surface (2023-01-18):** holding κ fixed anywhere in [1.2, 3.0] and refitting raises the RMSE by at most 0.1 vol pt (about one half spread) while ξ moves from 0.84 to 1.18; κ is weakly, not completely, unidentified.
- **Stability (2023, 52 weekly fits, one per calendar week):**
  - ρ: CV 3%, the stable parameter.
  - v0: tracks short-dated vol (√v0 vs 1-month ATM vol: correlation 0.99), as a state variable should.
  - κ and ξ: correlation of weekly changes 0.96. The surface identifies a combination of the two, so daily recalibration injects noise into κ-sensitive risk.
- **Crisis (2020-02-19 to 2020-03-18):** √v0 rises from 12% to 81%, ρ falls to −0.93, and the fit RMSE rises to 2.7 vol points.

## 3. Exotics and model risk
**Products.**
- **Down barrier puts:** closed form (Reiner–Rubinstein), PDE, and daily-monitored Monte Carlo checked against the Broadie–Glasserman–Kou continuity correction.
- **3-year autocallable on SPX:** quarterly autocall at 100% with an 8% p.a. accruing coupon, and a 60% knock-in monitored daily. It is priced by Monte Carlo and by two coupled PDEs (knocked / not knocked) stepped once per business day. PDE 97.45 vs Monte Carlo 97.43 ± 0.04.

**Model risk (SPX surface of 2023-01-18).**

| Model | Fair coupon | P(called at first date) | P(knock-in loss) | 60% knock-in put, % of notional |
|---|---|---|---|---|
| Black-Scholes (ATM vol 20.65%) | 14.0% | 50.3% | 8.2% | 4.9 |
| Heston | 15.2% | 63.8% | 8.3% | 9.3 |
| Local vol | 16.0% | 58.8% | 8.9% | 8.4 |

**Interpretation.**
- **Black-Scholes:** with no skew, it values the downside put at about half what the skew-consistent models do.
- **Vanilla repricing on the real surface:** implied-vol RMSE 8.91 (BS), 0.83 (Heston), 0.61 (local vol, by Crank–Nicolson) vol pt; none is inside the spread. Local vol reproduces a synthetic surface to 0.03 vol pt; on SPX it inherits the SVI slice errors (0.3–0.4 vol pt).
- **Calibration vs dynamics:** local vol built from Heston's own smiles reprices Heston's vanillas to 0.07 vol pt and gives a 15.94% coupon. Dynamics effect +0.77, calibration effect +0.01 coupon points (MC s.e. about 0.19 each). This is consistent with the forward smiles: local vol's forward skew flattens over time, Heston's persists (implied smile of a 3-month option starting in one year, from each model's paths).
- **3-year product:** stage 3 recalibrates on 7 expiries up to 2.92 years (the longest listed); beyond that, local vol holds each strike's implied vol constant and Heston extrapolates by its dynamics.
- **Scope:** one date and one product. Industry practice for such notes is stochastic local vol, whose mixing weight then becomes the model-risk parameter.

## 4. Greeks and hedging
- **Monte Carlo Greeks:** pathwise and likelihood-ratio deltas match the closed form. Likelihood ratio handles the digital option, where pathwise is identically zero. A mixed estimator gives gamma.
- **Hedged P&L identity:** for a short, delta-hedged option, the BS PDE reduces daily P&L to ½ΓS²(σ_imp²dt − (dS/S)²) − vega·dσ_imp, plus second-order terms.
- **Simulation:**
  - Discrete-hedging error 0.434 vs Kamal–Derman 0.441 at 63 rebalances, scaling as 1/√N.
  - With 5 bp costs there is an optimal rebalancing frequency.
  - Hedging at realised rather than implied vol locks in the same expected profit with less than half the dispersion.
- **Real SPX backtest:** monthly 1-month ATM calls sold at mid, hedged daily, re-marked daily at market mids.

| Period | Trades | Hedged P&L | gamma/theta | vega | vanna/volga | residual | Mean entry IV | Mean realised vol |
|---|---|---|---|---|---|---|---|---|
| 2020 Q1 | 3 | −127.8 | −96.4 | −65.5 | 10.6 | 23.5 | 17.2% | 36.3% |
| 2023 | 12 | +31.8 | +73.6 | −75.6 | 2.1 | 31.7 | 15.6% | 12.1% |

(Index points per one call; the March 2020 trade ends at the data cut-off on 2020-03-31 and is not fully observed.)

**Reading the backtest.**
- **2023:** the variance risk premium shows up in the variance term (+74.2); fixed-strike implied-vol re-marks during a rally give back about half of it (−34.2, full revaluation).
- **2020 Q1:** the tail of selling volatility.
- **Full revaluation (exact):** 2023: +74.2 spot/time at fixed vol, −34.2 implied-vol re-marks, −8.2 forward–index basis. 2020 Q1: −84.7, −41.3, −1.8.
- **Taylor attribution:** explains 97% (2020 Q1) and 84% (2023) of daily P&L variance (91% pooled). Its 2023 residual of +31.7 comes from the vol expansion (+39.4, of which +25.0 in the last three days before expiry) and the forward–index basis (−8.2, asynchronous index print). Daily fit quality and cumulative attribution are different questions.

## 5. Limitations and extensions
- **Data:** end-of-day mids from a single vendor; no intraday synchronicity; constant r and q per date when building local vol.
- **Heston PDE:** no 2D (ADI) solver. Heston prices are cross-checked by COS and Monte Carlo only.
- **Models:** no jumps and no rough volatility, which the short-dated residuals call for.
- **Next steps:** stochastic local vol for the autocallable; Andersen–Broadie upper bounds for LSM; smile-adjusted (minimum-variance) delta in the backtest; transaction costs in the real backtest.

## Reproducing
```bash
pip install -e ".[dev]" && pytest                    # 70 tests, all synthetic, no data needed
python scripts/stage1_crossval.py                    # stage 1 table and convergence figures
python scripts/stage2_calibrate_spx.py showcase      # needs optionsDX files in data/optionsdx/
python scripts/stage2_calibrate_spx.py timeseries
python scripts/stage3_model_risk.py
python scripts/stage4_hedging.py
```
