# Derivatives Pricing Engine

One product, three independent pricing methods, checked against each other — then calibrated to real SPX options and used to measure model risk on an autocallable.

| Module | Status |
|---|---|
| Black-Scholes closed form, Greeks, implied vol | ✅ tested |
| Heston via COS (Fourier) | ✅ tested against an independent Fourier method |
| Crank–Nicolson PDE, Rannacher start, American via penalty | ✅ measured order 2.00 |
| Monte Carlo: antithetic, control variates, Heston QE, Longstaff–Schwartz | ✅ tested |
| Market data: implied forward from parity, OTM IV surface | ✅ tested |
| Heston (whole surface) and SABR (per expiry) calibration | ✅ recovers known parameters on synthetic surfaces; SPX data pending |
| Dupire local volatility from SVI, local-vol PDE | ✅ reprices Heston vanillas within 0.03 vol pt |
| Barrier and autocallable: model risk across BS / Heston / local vol | planned |
| Greeks and hedging backtest with P&L attribution | planned |

## Stage 1 result: three methods, one price
See [reports/stage1_crossvalidation.md](reports/stage1_crossvalidation.md).

![PDE convergence](reports/figures/pde_convergence.png)

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
pytest                              # 59 tests, ~10 s
python scripts/stage1_crossval.py   # regenerate the stage 1 table and figures
python scripts/fetch_spx_chain.py   # daily SPX option chain snapshot (needs internet)
```
