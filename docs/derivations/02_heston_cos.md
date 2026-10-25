# 02 · Heston characteristic function and the COS method

Code: `src/derivpricing/models/heston.py`, `src/derivpricing/engines/cos.py`

## 1. Heston characteristic function
Let $X_t = \ln(S_t/S_0)$. By Itô,
$dX = (r-q-\tfrac12 v)\,dt + \sqrt v\,dW^S$, $\ dv = \kappa(\theta-v)\,dt + \xi\sqrt v\,dW^v$, $\ d\langle W^S,W^v\rangle=\rho\,dt$.

Guess an exponential-affine form (the model is affine in $(X, v)$):
$$\phi(u;\tau) = \mathbb E\big[e^{iuX_T}\mid X_t=x, v_t=v\big] = \exp\big(iux + C(\tau) + D(\tau)\,v\big),\quad \tau=T-t.$$
$f(t,x,v)=\phi$ is a martingale, so it satisfies the Kolmogorov backward equation
$$-f_\tau + (r-q-\tfrac12 v)f_x + \kappa(\theta-v)f_v + \tfrac12 v f_{xx} + \rho\xi v f_{xv} + \tfrac12\xi^2 v f_{vv} = 0.$$
Substituting and collecting the terms that multiply $v$ and those that don't gives two ODEs with $C(0)=D(0)=0$:
$$D' = \tfrac12\xi^2 D^2 - (\kappa - \rho\xi iu)D - \tfrac12(iu + u^2),\qquad C' = iu(r-q) + \kappa\theta D.$$
The first is a Riccati equation with constant coefficients. With $b=\kappa-\rho\xi iu$, $d=\sqrt{b^2+\xi^2(iu+u^2)}$ (roots of the quadratic are $(b\pm d)/\xi^2$) and $g=(b-d)/(b+d)$:
$$D(\tau) = \frac{b-d}{\xi^2}\,\frac{1-e^{-d\tau}}{1-ge^{-d\tau}},\qquad C(\tau) = iu(r-q)\tau + \frac{\kappa\theta}{\xi^2}\Big[(b-d)\tau - 2\ln\frac{1-ge^{-d\tau}}{1-g}\Big].$$

**Why this form ("little Heston trap", Albrecher et al. 2007).** Heston's original paper writes the same thing with $e^{+d\tau}$ and $g' = 1/g$. Both are algebraically equal, but the complex logarithm $\ln\frac{1-g'e^{d\tau}}{1-g'}$ crosses the negative real axis as $u$ grows for long maturities, and the principal branch then jumps by $2\pi i$, producing a discontinuous, wrong CF. With $e^{-d\tau}$ (Re $d>0$) the argument stays away from the branch cut.

**Checks in the tests.** $\xi\to0$ with $v_0=\theta=\sigma^2$: the CF tends to Black-Scholes. With $\rho=0$ the price error is $O(\xi^2)$ (first-order terms in $\xi$ come only with $\rho$), and we observe exactly that: $\xi$ ÷10 ⇒ error ÷100.

## 2. COS method (Fang & Oosterlee 2008)
Price as an integral against the density of $y=\ln(S_T/K)=x+X_T$, $x=\ln(S_0/K)$:
$$V = e^{-rT}\int_{\mathbb R} v(y)\,f(y)\,dy \approx e^{-rT}\int_a^b v(y)\,f(y)\,dy.$$
**Step 1 (cosine expansion of the density on $[a,b]$).**
$f(y) \approx \sum_{k=0}^{N-1}{}' A_k\cos\big(u_k(y-a)\big)$, $u_k = \frac{k\pi}{b-a}$, $A_k=\frac2{b-a}\int_a^b f(y)\cos(u_k(y-a))dy$.
Because $f$ is negligible outside $[a,b]$, extend the integral to $\mathbb R$:
$A_k \approx \frac2{b-a}\,\mathrm{Re}\big[\phi_y(u_k)e^{-iu_ka}\big]$ and $\phi_y(u) = e^{iux}\phi(u)$, so
$A_k \approx \frac2{b-a}\mathrm{Re}\big[\phi(u_k)\,e^{iu_k(x-a)}\big]$.
**Step 2 (payoff coefficients).** $V_k = \frac2{b-a}\int_a^b v(y)\cos(u_k(y-a))dy$. Substituting:
$$V \approx e^{-rT}\sum_{k=0}^{N-1}{}'\,\mathrm{Re}\big[\phi(u_k)e^{iu_k(x-a)}\big]\,V_k.$$
**Put coefficients.** $v(y)=K(1-e^y)^+$ is non-zero on $[a,0]$:
$V_k = \frac{2K}{b-a}\big(\psi_k(a,0)-\chi_k(a,0)\big)$, with
$\chi_k(c,d)=\int_c^d e^y\cos(u_k(y-a))dy$ (two integrations by parts) and $\psi_k(c,d)=\int_c^d\cos(u_k(y-a))dy$; closed forms in `_chi_psi`.

**Why price puts and get calls by parity.** A call payoff grows like $e^y$, so the part of the payoff beyond $b$ that the truncation drops is large and the result depends on $b$. The put payoff is bounded by $K$.

**Truncation interval.** $[a,b]=x+c_1 \pm L\sqrt{c_2+\sqrt{|c_4|}}$, $L=10$, with $c_n$ the cumulants of $X_T$. We get them generically from the cumulant generating function $\mathcal K(s)=\ln\phi(-is)$ by central differences at $s=\pm h,\pm2h$, so the same code works for any model with a CF.

**Error.** Two sources: truncation (controlled by $L$, decays like the tails of $f$) and series truncation (decays exponentially in $N$ for a smooth density). Black-Scholes reaches $10^{-14}$ by $N\approx 48$. Heston's CF decays more slowly in $u$ than the Gaussian one (its density is less smooth, with heavier tails) and $L=12$ widens the interval, so more terms are needed (figure `cos_convergence.png`).

**Independent check.** The tests compare with the Lewis (2001) single-integral formula evaluated by adaptive quadrature: agreement to $10^{-9}$. Both give 5.7851554344 for the Fang–Oosterlee case; the paper prints 5.785155450.
