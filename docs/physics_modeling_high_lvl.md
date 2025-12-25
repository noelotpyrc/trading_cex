# Physics → Trading Framework Mapping (Stochastic Damped Oscillator to Momentum + Mean Reversion)

This note formalizes how the **stochastic damped oscillator (underdamped Langevin)** in physics maps into our **discrete-time "momentum + mean reversion"** price-action framework, using unified symbols and consistent indexing.

---

## 1) Unified Symbols

### Market variables
- Price: $P_t > 0$
- Log price: $y_t := \ln P_t$
- Mean / equilibrium proxy: $m_t$ (e.g., EMA of $y_t$; simplest case is constant)

### State variables
- Displacement from equilibrium:
  $$
  x_t := y_t - m_t
  $$
- Velocity (latent momentum state): $v_t$

### Observables
- One-step log return:
  $$
  r_{t+1} := y_{t+1} - y_t
  $$

---

## 2) Physics Model: Stochastic Damped Oscillator (Continuous Time)

Let $x(t)$ be displacement from equilibrium.

Second-order form:
$$
\ddot{x}(t) + 2\beta\,\dot{x}(t) + \omega_0^2\,x(t) = \xi(t)
$$

Equivalent first-order system (define $v(t):=\dot{x}(t)$):
$$
\begin{aligned}
\dot{x}(t) &= v(t) \\
\dot{v}(t) &= -2\beta\,v(t) - \omega_0^2\,x(t) + \xi(t)
\end{aligned}
$$

Interpretation:
- $x$: displacement (stretch from equilibrium)
- $v$: velocity (momentum/inertia)
- $2\beta$: damping (friction; kills momentum)
- $\omega_0^2$: spring strength (mean reversion pull)
- $\xi(t)$: random forcing (shocks / order flow / news)

---

## 3) Discretization (Per-Bar Dynamics)

Assume uniform timestep $\Delta t$ (often set $\Delta t=1$ per bar). A simple Euler discretization gives:

$$
\boxed{
\begin{aligned}
x_{t+1} &= x_t + \Delta t\, v_t \\
v_{t+1} &= (1-2\beta\Delta t)\,v_t - (\omega_0^2\Delta t)\,x_t + \eta_{t+1}
\end{aligned}}
$$

where $\eta_{t+1}$ is discrete noise.

Define shorthand:
$$
a := 1-2\beta\Delta t,
\qquad
b := \omega_0^2\Delta t
$$
Then:
$$
x_{t+1} = x_t + \Delta t\,v_t,
\qquad
v_{t+1} = a\,v_t - b\,x_t + \eta_{t+1}
$$

---

## 4) Bridge to Returns: Where $r\approx \Delta t\,v$ Comes From

By definition:
$$
x_t = y_t - m_t
$$
so:
$$
x_{t+1}-x_t = (y_{t+1}-y_t) - (m_{t+1}-m_t) = r_{t+1} - \Delta m_{t+1}
$$
where $\Delta m_{t+1} := m_{t+1}-m_t$.

Thus:
$$
\boxed{r_{t+1} = (x_{t+1}-x_t) + \Delta m_{t+1}}
$$

Using the discrete physics position update $x_{t+1}-x_t=\Delta t\,v_t$:
$$
\boxed{r_{t+1} = \Delta t\,v_t + \Delta m_{t+1}}
$$

Two common regimes:
- **Constant mean proxy** $m_t \equiv m$: $\Delta m_{t+1}=0$, so
  $$
  \boxed{r_{t+1} = \Delta t\,v_t}
  $$
- **Slowly varying mean** (EMA/VWAP): treat $\Delta m_{t+1}$ as small drift or absorb into noise.

This is the precise place where the "$r\approx \Delta t\,v$" bridge is used: it connects the latent velocity state to observable returns.

---

## 5) Return-Only Form: Momentum Persistence + Mean Reversion Pull

From the bridge (constant-mean case for clarity):
$$
v_t = \frac{r_{t+1}}{\Delta t}
$$

Use the velocity update:
$$
v_{t+1} = a\,v_t - b\,x_t + \eta_{t+1}
$$

Multiply by $\Delta t$ to express in returns:
$$
\Delta t\,v_{t+1} = a(\Delta t\,v_t) - (\Delta t\,b)\,x_t + \Delta t\,\eta_{t+1}
$$

Replace $\Delta t\,v_{t+1}\approx r_{t+2}$ and $\Delta t\,v_t\approx r_{t+1}$:
$$
\boxed{r_{t+2} \approx a\,r_{t+1} - \gamma\,x_t + \varepsilon_{t+2}}
$$
where:
$$
\gamma := \Delta t\,b = \omega_0^2\,\Delta t^2,
\qquad
\varepsilon_{t+2} := \Delta t\,\eta_{t+1}
$$

Interpretation:
- **Momentum persistence:** $a\,r_{t+1}$ (returns persist when damping is weak)
- **Mean reversion:** $-\gamma x_t$ (pull proportional to displacement from equilibrium)

> Index note: the appearance of $t+2$ is just a one-step shift from using the velocity update. You can reindex time to write an equivalent "next-step" form if preferred.

---

## 6) Our Framework (Feature-Driven, Regime-Varying Coefficients)

We use the same structure but allow coefficients to vary with regime:
$$
\boxed{r_{t+1} \approx \rho_t\,r_t - \gamma_t\,x_{t-1} + \varepsilon_{t+1}}
$$

Where:
- $x_t = y_t - m_t$ is displacement
- $\rho_t$ is an **effective persistence** (discrete analogue of damping)
- $\gamma_t$ is an **effective restoring strength** (discrete analogue of spring pull)
- both can be modeled as functions of observable features (volatility, autocorr, PV, efficiency, etc.)

Conceptual mapping (constant coefficients):
$$
\rho \leftrightarrow a = 1-2\beta\Delta t,
\qquad
\gamma \leftrightarrow \omega_0^2\Delta t^2
$$

---

## 7) Why This Unifies Momentum and Mean Reversion

In the physics system, both behaviors arise from the same dynamics:
- Momentum = persistence of $v$ when damping is low ($\beta$ small → $\rho$ high)
- Mean reversion = spring pull toward equilibrium when displacement is large ($\omega_0$ large → $\gamma$ high)

In markets, regime features (autocorr/run-structure/VR, PullSlope, Parkinson vol, efficiency ratio) are used to estimate the **effective** $\rho_t$ and $\gamma_t$ from history.

---

## 8) Practical Interpretation for Feature Modeling

- $x_t = y_t - m_t$: "how stretched are we?"
- $r_t$: "what is current motion?"
- $\rho_t$: "does motion persist in this regime?"
- $\gamma_t$: "does stretch get corrected in this regime?"

Parkinson volatility + efficiency ratio can indicate whether high range is:
- directional progress (trend regime) or
- two-way exploration (mean-reversion regime)

---

## 9) Summary

Physics model:
$$
\ddot{x} + 2\beta\dot{x} + \omega_0^2 x = \xi
$$

Discrete state model:
$$
x_{t+1} = x_t + \Delta t v_t,\quad v_{t+1} = a v_t - b x_t + \eta
$$

Bridge to returns (constant mean):
$$
r_{t+1} = \Delta t v_t
$$

Return-only form:
$$
r_{t+2} \approx a r_{t+1} - \gamma x_t + \varepsilon
$$

Our trading framework:
$$
r_{t+1} \approx \rho_t r_t - \gamma_t x_{t-1} + \varepsilon_{t+1}
$$

Same structure, but $\rho_t,\gamma_t$ are learned from features (regime-dependent "effective physics constants").

---

## 10) Approximate "Closed-Form" for $\rho$ and $\gamma$

This section gives two useful approximations:

1) a **physics ↔ trading** parameter mapping (interpretation), and  
2) a **data-driven closed-form estimator** from price history (rolling OLS).

### 10.1 Physics ↔ trading mapping (constant-coefficient approximation)

From the discretized oscillator
$$
v_{t+1} = (1-2\beta\Delta t)\,v_t - (\omega_0^2\Delta t)\,x_t + \eta_{t+1}
$$
the "effective" discrete coefficients are approximately:
$$
\rho \approx 1-2\beta\Delta t,
\qquad
\gamma \approx \omega_0^2\,\Delta t^2
$$
(using the bridge $r_{t+1}\approx \Delta t\,v_t$ when $m_t$ is constant or slow).

So you can invert for physical intuition:
$$
\beta \approx \frac{1-\rho}{2\Delta t},
\qquad
\omega_0^2 \approx \frac{\gamma}{\Delta t^2}.
$$

### 10.2 Data-driven closed-form estimator (rolling window)

Use the return-only trading form with consistent symbols:
- $y_t=\ln P_t$
- $m_t$ is your mean proxy
- $x_t=y_t-m_t$
- $r_t=y_t-y_{t-1}$

Fit over a window of $W$ samples:
$$
\boxed{r_{u+1} \approx \rho\,r_u - \gamma\,x_u}
\quad\text{for } u=t-W,\dots,t-1.
$$

This is a 2-variable least-squares fit with a closed-form solution.

Compute these sums (all over $u=t-W,\dots,t-1$):

$$
\sum r_u^2,\quad
\sum x_u^2,\quad
\sum r_u x_u,\quad
\sum r_u r_{u+1},\quad
\sum x_u r_{u+1}.
$$

Define:
$$
\Delta :=
\left(\sum r_u^2\right)\left(\sum x_u^2\right) - \left(\sum r_u x_u\right)^2.
$$

Then the least-squares estimates are:
$$
\boxed{
\rho
=
\frac{
\left(\sum x_u^2\right)\left(\sum r_u r_{u+1}\right)
-
\left(\sum r_u x_u\right)\left(\sum x_u r_{u+1}\right)
}{
\Delta
}}
$$

$$
\boxed{
\gamma
=
\frac{
\left(\sum r_u x_u\right)\left(\sum r_u r_{u+1}\right)
-
\left(\sum r_u^2\right)\left(\sum x_u r_{u+1}\right)
}{
\Delta
}}
$$

Notes:
- This is the same idea as the oscillator: $\rho$ captures **persistence** of motion; $\gamma$ captures **pullback** proportional to displacement.
- If $\Delta$ is tiny (collinearity), add a small ridge by replacing $\sum r_u^2 \to \sum r_u^2+\lambda$ and $\sum x_u^2 \to \sum x_u^2+\lambda$.
- If you want to constrain: $\rho \leftarrow \min(1,\max(0,\rho))$, $\gamma \leftarrow \max(0,\gamma)$ (optional, depending on your use).

### 10.3 Optional: incorporate moving mean drift explicitly

If $m_t$ changes materially at your bar size, use:
$$
r_{u+1} = (x_{u+1}-x_u) + (m_{u+1}-m_u)
$$
and either:
- include $(m_{u+1}-m_u)$ as an additional regressor, or
- treat it as part of the residual term when $m$ is "slow".

# OHLC-Only Approximate Closed-Form Estimates for $\rho_t$ and $\gamma_t$

This note defines **practical, approximate “closed-form”** estimates of:
- $\rho_t$: **momentum persistence**
- $\gamma_t$: **mean-reversion (pullback) strength**

using only the last $W$ bars of **OHLC** data.

---

## 1) Inputs and Windowing

At time $t$, assume you have OHLC bars:
$$
(O_u, H_u, L_u, C_u),\quad u=t-W,\dots,t
$$
with $H_u \ge \max(O_u,C_u)$ and $L_u \le \min(O_u,C_u)$, and all prices $>0$.

Use a small $\epsilon > 0$ in denominators for numerical stability.

---

## 2) Construct the Two Core Series (from OHLC)

### 2.1 One-step log return (observable “speed”)
Define close-to-close log return:
$$
r_u := \ln C_u - \ln C_{u-1}
$$

This is the observable increment used to measure persistence.

### 2.2 Level and displacement (equilibrium deviation)

Define a log “typical price” level (OHLC-only):
$$
z_u := \ln\!\left(\frac{O_u+H_u+L_u+C_u}{4}\right)
$$

Define a simple equilibrium over the last $W$ bars:
$$
\bar z_t := \frac{1}{W}\sum_{j=t-W+1}^{t} z_j
$$

Define displacement from equilibrium:
$$
x_u := z_u - \bar z_t
$$

Notes:
- Using $z_u$ instead of $\ln C_u$ injects intrabar information from OHLC.
- $\bar z_t$ is intentionally simple: it is a locally constant “fair value” proxy.

---

## 3) Approximate Closed-Form $\rho_t$: Momentum Persistence

Estimate persistence as an AR(1) coefficient on returns over the window:

Model (no intercept):
$$
r_u \approx \rho_t\, r_{u-1}
$$

Closed-form OLS estimate:
$$
\boxed{
\rho_t \approx
\frac{\sum_{u=t-W+2}^{t} r_u\,r_{u-1}}
     {\sum_{u=t-W+2}^{t} r_{u-1}^2 + \epsilon}
}
$$

Interpretation:
- $\rho_t > 0$: returns tend to persist (momentum)
- $\rho_t \approx 0$: little persistence
- $\rho_t < 0$: sign-flipping tendency (mean-reverting returns)

Optional constraint (if desired):
$$
\rho_t \leftarrow \min(1,\max(0,\rho_t))
$$
(Only do this if you conceptually require $\rho_t\in[0,1]$.)

---

## 4) Approximate Closed-Form $\gamma_t$: Mean-Reversion (Pullback) Strength

Estimate pullback strength as the slope of “next return vs negative displacement”.

Model (no intercept):
$$
r_{u+1} \approx \gamma_t(-x_u)
$$

Closed-form OLS estimate:
$$
\boxed{
\gamma_t \approx
-\frac{\sum_{u=t-W}^{t-1} x_u\, r_{u+1}}
       {\sum_{u=t-W}^{t-1} x_u^2 + \epsilon}
}
$$

Interpretation:
- $\gamma_t > 0$: when price is above equilibrium ($x_u>0$), the next return tends to be negative ⇒ **mean reversion**
- $\gamma_t \approx 0$: weak pullback
- $\gamma_t < 0$: displacement tends to extend ⇒ **trend-away**

Optional constraint (if desired):
$$
\gamma_t \leftarrow \max(0,\gamma_t)
$$

---

## 5) Optional OHLC Regime Features (Useful in Practice)

These are not required for the closed-form estimates above, but are commonly useful for:
- gating regimes (trend vs chop),
- explaining why $\gamma$ and $\rho$ behave differently when ranges expand.

### 5.1 Parkinson volatility (range-based scale)
Per-bar log range:
$$
g_u := \ln\left(\frac{H_u}{L_u}\right)
$$

Windowed Parkinson volatility:
$$
\text{PV}_t := \sqrt{\frac{1}{4W\ln 2}\sum_{u=t-W+1}^{t} g_u^2}
$$

### 5.2 Efficiency ratio (range vs progress)
Per-bar efficiency:
$$
\text{Eff}_u := \frac{|\ln C_u - \ln O_u|}{|\ln H_u - \ln L_u|+\epsilon}
$$

Windowed efficiency:
$$
\text{EffAvg}_t :=
\frac{\sum_{u=t-W+1}^{t} |\ln C_u - \ln O_u|}
     {\sum_{u=t-W+1}^{t} |\ln H_u - \ln L_u|+\epsilon}
$$

Interpretation:
- high $\text{EffAvg}$: big range mostly becomes net progress (trend-like)
- low $\text{EffAvg}$: big range but little progress (two-way chop / reversion-like)

---

## 6) Summary

Given OHLC bars, define:
- returns $r_u = \ln C_u - \ln C_{u-1}$
- log-typical level $z_u = \ln((O_u+H_u+L_u+C_u)/4)$
- displacement $x_u = z_u - \bar z_t$ with $\bar z_t = \frac{1}{W}\sum z$

Then estimate:
$$
\rho_t \approx
\frac{\sum r_u r_{u-1}}{\sum r_{u-1}^2 + \epsilon},
\qquad
\gamma_t \approx
-\frac{\sum x_u r_{u+1}}{\sum x_u^2 + \epsilon}
$$

Optionally compute PV and efficiency to interpret or gate regimes:
- $\text{PV}_t$ from $\ln(H/L)$
- $\text{EffAvg}_t$ from net move vs range

---

