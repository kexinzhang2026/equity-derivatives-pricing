# 06 · From quotes to a calibrated Heston model

Code: `src/derivpricing/calibration/surface.py`, `heston_calib.py`, `synthetic.py`

## 1. Implied forward and discount from parity
For European options on the same expiry, $C(K) - P(K) = D\,(F - K)$ holds without any model. Regressing $C-P$ on $K$ over near-the-money strikes (where both sides are liquid) gives slope $-D$ and intercept $DF$.
Why not use the spot and a rate curve? (i) Yahoo's spot is not synchronous with the option quotes; (ii) SPX dividends are discrete and their yield is not observed; (iii) the rate the options imply (box-spread rate) differs from Treasury rates. The options' own forward is the only consistent choice. Unit test: on synthetic BS prices the regression recovers $F$ and $D$ to $10^{-10}$.

Pricing with $(F, D)$: Black-76, $C = D[F N(d_1) - K N(d_2)]$. In code we reuse the BS functions with $S_{\text{eff}} = DF$, $r = -\ln D / T$, $q = 0$, which gives exactly the same forward.

## 2. Cleaning
Two-sided markets only, mid ≥ 0.05, relative spread ≤ 50%, $T \ge 7$ days, standardised moneyness $\ln(K/F)/(0.2\sqrt T) \in [-4, 2.5]$, and only OTM options (puts below $F$, calls above). The bid and ask are also inverted to IVs: the IV bid–ask band is the natural yardstick for "good fit".

## 3. Objective
$$\min_{p}\ \sum_i \left(\frac{C^{\text{Heston}}_i(p) - C^{\text{mkt}}_i}{\mathcal V_i}\right)^2$$
Since $C(\sigma+\delta\sigma) \approx C(\sigma) + \mathcal V\,\delta\sigma$, each term is approximately the squared implied-vol error, but needs no IV inversion inside the optimiser. Price errors without weights would be dominated by expensive long-dated options; plain IV errors are correct but cost one root-find per option per iteration.
Wing vegas are floored at 1% of the median vega so one illiquid quote cannot dominate.

## 4. Optimiser
Bounded trust-region least squares (`scipy.optimize.least_squares`, TRF) with a finite-difference Jacobian; each residual evaluation is one vectorised COS call per expiry, so a full calibration takes about one second. The surface is non-convex, so we start from a "typical equity" point plus random points and keep all solutions.
Analytic Heston gradients exist (Cui et al. 2017) and would be the next speed-up.

## 5. Validation on synthetic data
1. Noise-free surface generated from known parameters → recovered to relative $10^{-4}$, RMSE $<10^{-5}$.
2. 0.2 vol-pt Gaussian noise → RMSE ≈ 0.16 vol pt: the fit stops at the noise level (no overfitting, no failed optimisation). $v_0$ and $\rho$ stay within a few percent; $\kappa$ moves more.

## 6. Identifiability: the κ profile
Fix $\kappa$ on a grid and refit the other four (`fixed={"kappa": k}`). Figure `heston_kappa_profile.png`:
- With expiries out to 2 years the error has a clear minimum at the true $\kappa$.
- With expiries only out to 6 months, the valley is much flatter: $\kappa$ governs how fast variance reverts, which shows up mostly in the term structure across long maturities.
- Along the valley $\xi$ rises with $\kappa$: the smile curvature at maturity $T$ depends roughly on $\xi$ relative to how much mean reversion has averaged the variance, so a faster reversion needs a larger vol-of-vol to produce the same smile.
This is why calibrated $\kappa$ jumps from day to day on real data while $\rho$ and $v_0$ stay stable. Stability across dates is measured in the real-data stage.
