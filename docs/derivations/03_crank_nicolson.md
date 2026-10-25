# 03 · Crank–Nicolson for the Black-Scholes PDE

Code: `src/derivpricing/engines/pde.py`

## 1. Transformation
BS PDE: $V_t + (r-q)SV_S + \tfrac12\sigma^2S^2V_{SS} - rV = 0$, $V(T,S)=\text{payoff}(S)$.
With $x=\ln S$ ($SV_S = V_x$, $S^2V_{SS}=V_{xx}-V_x$) and $\tau=T-t$:
$$V_\tau = aV_{xx} + bV_x - rV,\qquad a=\tfrac12\sigma^2,\ b=r-q-\tfrac12\sigma^2.$$
Constant coefficients, a forward-in-$\tau$ initial-value problem with $V(0,x)=\text{payoff}(e^x)$. A uniform $x$-grid is naturally denser at low $S$, where relative moves matter.

## 2. Spatial discretisation
Central differences on $x_j = x_0 + jh$:
$(LV)_j = a\frac{V_{j+1}-2V_j+V_{j-1}}{h^2} + b\frac{V_{j+1}-V_{j-1}}{2h} - rV_j = \ell V_{j-1} + dV_j + uV_{j+1}$,
$\ell = \frac a{h^2}-\frac b{2h}$, $d=-\frac{2a}{h^2}-r$, $u=\frac a{h^2}+\frac b{2h}$. Truncation error $O(h^2)$ (Taylor: the odd terms cancel).
If $|b|h/(2a) > 1$ then $\ell<0$ or $u<0$ and the scheme can produce oscillations (cell Péclet number). With $\sigma = 25\%$ that needs $h > 2a/|b|$, far coarser than any grid we use.

## 3. Time stepping: the θ-scheme
$$(I-\theta\Delta\tau L)V^{n+1} = (I+(1-\theta)\Delta\tau L)V^n$$
θ=0 explicit (stable only if $\Delta\tau \lesssim h^2/(2a)$), θ=1 implicit Euler (first order), θ=½ Crank–Nicolson (second order, unconditionally stable). Each step is one tridiagonal solve, $O(M)$.

**Von Neumann analysis.** For a Fourier mode with symbol $\lambda\le0$ of $L$ (eigenvalue of the discrete operator, $\lambda \approx -4a\sin^2(\omega h/2)/h^2$), the amplification factor is
$$G(\lambda\Delta\tau) = \frac{1+(1-\theta)\lambda\Delta\tau}{1-\theta\lambda\Delta\tau}.$$
CN: $|G|\le 1$ for all $\lambda\Delta\tau\le0$ (A-stable), **but** $G\to-1$ as $\lambda\Delta\tau\to-\infty$. The highest-frequency modes are not damped, they just flip sign each step.
Implicit Euler: $G = 1/(1-\lambda\Delta\tau)\to0$ (L-stable): high frequencies are wiped out.

## 4. Why the payoff kink matters, and the Rannacher fix
A call/put payoff has a kink at $K$, so its Fourier content has large high-frequency components. Under CN they persist as a sawtooth that is small in $V$ but large in $V_{xx}$: gamma near the strike oscillates (figure `rannacher_gamma.png`, error 0.2 vs exact gamma 0.03).
**Rannacher (1984), as analysed by Giles & Carter (2006):** replace the first two CN steps by four implicit Euler half-steps. Those damp high frequencies by $\approx\prod(1-\lambda\Delta\tau/2)^{-1}$; after that the solution is smooth and CN's second-order accuracy is recovered. Four steps (not one) are needed so that the first-order error of implicit Euler, confined to $O(\Delta\tau)$ total time, does not spoil the global $O(\Delta\tau^2)$.

**Cell averaging (Pooley, Vetzal & Forsyth 2003).** Set $V^0_j = \frac1h\int_{x_j-h/2}^{x_j+h/2}\text{payoff}(e^y)dy$ instead of the point value. Then the error no longer depends on where $K$ falls between nodes, and log-log convergence plots are a clean straight line (measured slope −2.00). Without it the error oscillates between refinements.

## 5. Boundary conditions
Grid $x_0\pm 6\sigma\sqrt T$. Dirichlet values from the asymptotics: call $V(\tau,S_{\max})\approx Se^{-q\tau}-Ke^{-r\tau}$, $V(\tau,S_{\min})=0$; put the reverse. At 6 standard deviations the boundary error is far below the discretisation error.

## 6. Greeks from the grid
$\Delta = V_x/S$, $\Gamma = (V_{xx}-V_x)/S^2$ with central differences at the node holding $S_0$. Free byproduct of the solve, second-order accurate.

## 7. American exercise: penalty method (Forsyth & Vetzal 2002)
The American price solves the linear complementarity problem
$$\min\big(-V_t - \mathcal L V,\ V - g\big) = 0,\quad g = \text{payoff}.$$
Discretely: $\min(AV - \text{rhs},\ V-g) = 0$. Penalty form: $(A + P)V = \text{rhs} + Pg$ with $P_{jj} = \rho$ (large) where $V_j<g_j$, else 0. Iterate: solve, update the set of exercised nodes, solve again; stops when the set is unchanged, typically 2–3 passes. Each pass is one tridiagonal solve. Penalty error is $O(1/\rho)$.
The exercise boundary creates a second non-smoothness (in $V_{SS}$), so convergence of American prices is between first and second order in practice.

**Check.** S=36, K=40, r=6%, σ=20%, T=1: PDE 4.4867, CRR binomial with 20,000 steps 4.4867. (Longstaff–Schwartz's Table 1 lists 4.478 from a coarser finite-difference grid, a good reminder to check "reference values" too.)
