# Derivatives Pricing Engine

One product, three independent pricing methods, checked against each other — then calibrated to real SPX options and used to measure model risk on an autocallable.

| Module | Status |
|---|---|
| Black-Scholes closed form, Greeks, implied vol | ✅ tested |
| Heston via COS (Fourier) | planned |
| Crank–Nicolson PDE (European / American / barrier) | planned |
| Monte Carlo with variance reduction, Longstaff–Schwartz | planned |
| Heston / SABR calibration to SPX implied vol surface | planned |
| Barrier and autocallable: model risk across BS / Heston / local vol | planned |
| Greeks and hedging backtest with P&L attribution | planned |

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
pytest
```
