# 04 · Monte Carlo: estimators, variance reduction, Heston QE

Code: `src/derivpricing/engines/monte_carlo.py`

## 1. Plain estimator
$\hat V_n = \frac1n\sum_i Y_i$, $Y_i = e^{-rT}\text{payoff}(S_T^{(i)})$. By the CLT, $\hat V_n \approx N(V, s^2/n)$, so the standard error is $s/\sqrt n$: halving the error costs 4× the paths (test `test_stderr_scales_like_inverse_sqrt_n`). Variance reduction attacks $s$, not the rate.

GBM is simulated **exactly**: $\ln S_{t+\Delta} = \ln S_t + (r-q-\tfrac12\sigma^2)\Delta + \sigma\sqrt\Delta Z$. No time-discretisation bias, so any MC error vs the closed form is pure statistics.

## 2. Antithetic variates
Pair each $Z$ with $-Z$. $\mathrm{Var}\big(\tfrac{Y(Z)+Y(-Z)}2\big) = \tfrac12\mathrm{Var}(Y)(1+\mathrm{corr}(Y(Z),Y(-Z)))$. Helps when the payoff is monotone in $Z$ (correlation negative). Measured: ~2× variance reduction for an ITM call.
**Standard error pitfall:** the $2n$ samples are not independent. Average each pair first and compute the stderr over the $n$ pair-means (`antithetic()`); treating all $2n$ as i.i.d. understates the error.

## 3. Control variates
For a variable $X$ with known mean, $Y_{cv} = Y - \beta(X - \mathbb E X)$ is unbiased for any $\beta$, with variance $\mathrm{Var}Y - 2\beta\mathrm{Cov}(Y,X) + \beta^2\mathrm{Var}X$, minimised at $\beta^* = \mathrm{Cov}(Y,X)/\mathrm{Var}X$, giving $\mathrm{Var}Y\,(1-\rho_{XY}^2)$.
Control used: $X = e^{-rT}S_T$, $\mathbb E^Q X = S_0e^{-qT}$ (a model-free fact: the discounted, dividend-adjusted stock is a martingale). Works under Heston too. Measured: ~8× variance reduction for the ITM call. $\hat\beta$ estimated from the same sample introduces an $O(1/n)$ bias, negligible here.

## 4. Heston: why not Euler
Euler for the variance: $v_{t+\Delta} = v_t + \kappa(\theta-v_t)\Delta + \xi\sqrt{v_t\Delta}Z$ goes negative with positive probability, then $\sqrt{v}$ is undefined. Fixes like truncation ($v^+$) or reflection ($|v|$) are biased and converge slowly, especially when the Feller condition $2\kappa\theta\ge\xi^2$ fails, which is the usual case for calibrated equity parameters (Fang–Oosterlee set: ratio 0.38).

## 5. Andersen's QE scheme (2008)
The CIR conditional law $v_{t+\Delta}\mid v_t$ is a scaled non-central $\chi^2$ with known moments:
$m = \theta + (v_t-\theta)e^{-\kappa\Delta}$,
$s^2 = \frac{v_t\xi^2e^{-\kappa\Delta}}{\kappa}(1-e^{-\kappa\Delta}) + \frac{\theta\xi^2}{2\kappa}(1-e^{-\kappa\Delta})^2$, $\psi = s^2/m^2$.
- **$\psi\le\psi_c$ (=1.5), variance away from 0:** $v' = a(b+Z)^2$. Matching $\mathbb E v' = a(1+b^2)=m$ and $\mathrm{Var}\,v' = 2a^2(1+2b^2) = s^2$ gives
  $b^2 = 2\psi^{-1}-1+\sqrt{2\psi^{-1}}\sqrt{2\psi^{-1}-1}$, $a = m/(1+b^2)$ (requires $\psi\le2$).
- **$\psi>\psi_c$, variance near 0:** mixture of a point mass at 0 (probability $p$) and an exponential with rate $\beta$:
  $p = \frac{\psi-1}{\psi+1}$, $\beta = \frac{1-p}m$; sample by inverting the CDF: $v' = 0$ if $U\le p$, else $\beta^{-1}\ln\frac{1-p}{1-U}$.
Both branches are non-negative by construction and match the first two moments exactly.

**Log-spot step.** Write $W^S = \rho W^v + \sqrt{1-\rho^2}W^\perp$ and use the variance SDE to replace $\int\sqrt v\,dW^v = \frac1\xi\big(v' - v - \kappa\theta\Delta + \kappa\int v\,ds\big)$. Then approximate $\int_t^{t+\Delta}v\,ds \approx \Delta(\gamma_1 v + \gamma_2 v')$, $\gamma_1=\gamma_2=\tfrac12$:
$$\ln S' = \ln S + (r-q)\Delta + K_0 + K_1v + K_2v' + \sqrt{K_3v+K_4v'}\,Z^\perp,$$
$K_0=-\frac{\rho\kappa\theta}\xi\Delta$, $K_{1}=\gamma_1\Delta(\frac{\kappa\rho}\xi-\frac12)-\frac\rho\xi$, $K_{2}=\gamma_2\Delta(\frac{\kappa\rho}\xi-\frac12)+\frac\rho\xi$, $K_{3}=\gamma_1\Delta(1-\rho^2)$, $K_4=\gamma_2\Delta(1-\rho^2)$.

**Check.** 64 steps, 500k paths, control variate: 5.7838 ± 0.0071 vs COS 5.7852 (0.2 s.e.).
