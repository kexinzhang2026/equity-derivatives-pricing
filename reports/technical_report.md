# Cross-validated equity derivatives pricing on SPX: methods, calibration, model risk and hedging

*Technical report. Code, tests and figures: this repository. Every formula used in the code is derived in `docs/derivations/01`–`10`.*

## Summary
1. **Three independent pricing methods agree.** A Fourier (COS) engine, a Crank–Nicolson PDE engine and a Monte Carlo engine price the same European, Heston and American contracts to within discretisation or statistical error. Convergence rates are measured, not assumed: PDE order 2.00, exponential COS convergence, Monte Carlo error ∝ 1/√n.
2. **Calibration to SPX (optionsDX end-of-day, 2020 Q1 and 2023).** One Heston parameter set fits a day's surface to about 0.8 implied-vol points but cannot reproduce the 1-month skew; SABR, fitted per expiry, reaches 0.43. Across 59 weekly fits in 2023, ρ is stable (CV 3%) while κ and ξ move together (correlation of weekly changes 0.96), the identifiability ridge predicted beforehand on synthetic data. In March 2020 the fit error rises to 2.7 vol points.
3. **Model risk on an autocallable.** With BS, Heston and Dupire local vol all fitted to the same SPX surface, the fair coupon of a 3-year autocallable with a 60% knock-in is 14.0%, 15.2% and 16.0%. BS misses the skew. Heston and local vol agree on vanillas but imply different forward smiles, so they disagree on path-dependent payoffs.
4. **Hedging.** Discrete-hedging error matches the Kamal–Derman formula. On real SPX data, short 1-month ATM calls hedged daily earned the variance risk premium in 2023 and lost heavily in 2020 Q1; gamma/theta, vega and vanna/volga explain 91% of daily P&L variance.

## 1. Pricing engines and cross-validation
| Engine | Method | Validation |
|---|---|---|
| Closed form | Black-Scholes, Greeks, implied vol | Hull textbook values, put-call parity, no-arbitrage bounds, Greeks vs finite differences, Greeks satisfy the BS PDE |
| Fourier | Heston characteristic function ("little Heston trap" form), COS method | Agrees with the Lewis (2001) integral by adaptive quadrature to 1e-9; reduces to BS at rate O(ξ²) as ξ → 0 |
| PDE | Crank–Nicolson in log-price, Rannacher start, cell-averaged payoff, penalty method for American exercise, local-vol coefficients | Order 2.00 on Europeans; American put equals a 20,000-step binomial tree |
| Monte Carlo | Exact GBM, antithetic and control variates, Andersen QE for Heston, Longstaff–Schwartz, local-vol Euler | Within 3 s.e. of the other engines; LSM is below the PDE American value, as a lower bound should be |

| Product | Closed form | COS | Crank–Nicolson | Monte Carlo (± 1 s.e.) |
|---|---|---|---|---|
| European call, BS | 12.0056 | 12.0056 | 12.0057 | 12.0042 ± 0.0058 |
| European call, Heston (Feller violated) | – | 5.7852 | – | 5.7838 ± 0.0071 |
| American put (S=36, K=40) | – | – | 4.4867 | 4.4758 ± 0.0065 (50 dates) |

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
- **Stability (2023, 59 weekly fits):**
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
- **Heston vs local vol:** both reproduce the vanilla market (local vol exactly by Dupire's theorem, which was verified on synthetic data to 0.03 vol points). They differ because a path-dependent payoff depends on the forward smile. Local vol's forward skew flattens over time; Heston's persists. This is shown with the implied smile of a 3-month option starting in one year, computed from each model's paths.
- **Industry practice:** stochastic local vol, which matches vanillas exactly and takes its forward-smile dynamics from the stochastic part. Its mixing weight then becomes the model-risk parameter.

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

(Index points per one call; the March 2020 trade ends at the data cut-off on 2020-03-31.)

**Reading the backtest.**
- **2023:** the variance risk premium shows up in gamma/theta, and most of it is given back through fixed-strike vol re-marks during a rally.
- **2020 Q1:** the tail of selling volatility.
- **Attribution:** gamma/theta plus vega explain 86% of daily P&L variance, and vanna/volga raise this to 91%. The small positive residual is concentrated on large down days.

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
