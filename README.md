# Derivatives Pricing Engine

One product, three independent pricing methods, checked against each other — then calibrated to real SPX options and used to measure model risk on an autocallable.

| Module | Status |
|---|---|
| Black-Scholes closed form, Greeks, implied vol | ✅ tested |
| Heston via COS (Fourier) | ✅ tested against an independent Fourier method |
| Crank–Nicolson PDE, Rannacher start, American via penalty | ✅ measured order 2.00 |
| Monte Carlo: antithetic, control variates, Heston QE, Longstaff–Schwartz | ✅ tested |
| Heston / SABR calibration to SPX implied vol surface | planned |
| Dupire local volatility | planned |
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
pytest                              # 44 tests, ~5 s
python scripts/stage1_crossval.py   # regenerate the stage 1 table and figures
python scripts/fetch_spx_chain.py   # daily SPX option chain snapshot (needs internet)
```
