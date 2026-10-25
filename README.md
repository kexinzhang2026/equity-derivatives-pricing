# Derivatives Pricing Engine

One product, three independent pricing methods, checked against each other — then calibrated to real SPX options and used to measure model risk on an autocallable.

| Module | Status |
|---|---|
| Black-Scholes closed form, Greeks, implied vol | ✅ tested |
| Heston via COS (Fourier) | ✅ tested against an independent Fourier method |
| Crank–Nicolson PDE, Rannacher start, American via penalty | ✅ measured order 2.00 |
| Monte Carlo: antithetic, control variates, Heston QE, Longstaff–Schwartz | ✅ tested |
| Market data: implied forward from parity, OTM IV surface | ✅ tested |
| Heston (whole surface) and SABR (per expiry) calibration to SPX, 2020 Q1 + 2023 | ✅ [stage 2 report](reports/stage2_calibration.md) |
| Dupire local volatility from SVI, local-vol PDE | ✅ reprices Heston vanillas within 0.03 vol pt |
| Barriers (closed form / PDE / MC) and a 3y autocallable (PDE / MC) | ✅ tested |
| Model risk: same SPX surface, BS vs Heston vs local vol | ✅ [stage 3 report](reports/stage3_model_risk.md) |
| MC Greeks (pathwise, likelihood ratio), hedging simulation and real SPX hedging backtest with P&L attribution | ✅ [stage 4 report](reports/stage4_hedging.md) |

## Stage 1 result: three methods, one price
See [reports/stage1_crossvalidation.md](reports/stage1_crossvalidation.md).

![PDE convergence](reports/figures/pde_convergence.png)

## Headline results
- **Three methods agree**: COS, Crank–Nicolson (measured order 2.00) and Monte Carlo on European, Heston and American prices.
- **SPX calibration**: Heston fits 2023 surfaces to ~0.8 vol pt but misses the 1-month skew; ρ is stable (CV 3%) while κ and ξ move together (weekly-change correlation 0.96), the ridge predicted by the synthetic identifiability study. In March 2020 the fit error triples.
- **Hedging**: discrete-hedging error matches Kamal–Derman (0.434 vs 0.441). On real SPX data, short 1-month ATM calls hedged daily earned the variance risk premium in 2023 (+73.6 index points of gamma/theta, mostly given back via vega) and lost heavily in 2020 Q1; gamma/theta + vega + vanna/volga explain 91% of daily P&L variance.
- **Model risk**: on the same market, the fair coupon of a 3y SPX autocallable is 14.0% (BS), 15.2% (Heston), 16.0% (local vol). BS misses the skew; Heston and local vol match vanillas but imply different forward smiles.

![Model risk](reports/figures/autocall_model_risk.png)

## Data
SPX end-of-day option quotes from [optionsDX](https://www.optionsdx.com) (free registration). Put the monthly files in `data/optionsdx/` (`.txt` or `.txt.gz`); they are not redistributed here.

## Layout
```
src/derivpricing/   models/ engines/ products/ calibration/ hedging/ data/
tests/              property tests: parity, limits, convergence order
notebooks/          analysis, one per stage
reports/            figures and the technical report
docs/derivations/   every formula used in code, derived step by step
```

## Run
```bash
pip install -e ".[dev]"
pytest                              # 70 tests
python scripts/stage1_crossval.py   # regenerate the stage 1 table and figures
python scripts/stage2_calibrate_spx.py showcase     # one-date SPX calibration
python scripts/stage2_calibrate_spx.py timeseries   # weekly fits (~10 min)
python scripts/stage3_model_risk.py                 # autocallable model risk
python scripts/stage4_hedging.py                    # hedging simulation + SPX backtest
python scripts/fetch_spx_chain.py   # daily live SPX snapshot from Yahoo (run in market hours)
```
