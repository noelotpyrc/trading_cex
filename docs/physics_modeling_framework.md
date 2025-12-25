# Unified Momentum + Mean-Reversion Framework (Discrete-Time, Feature-Driven)

This document formalizes a unified “physics-style” model of price action that combines **momentum** and **mean reversion** in a single high-level equation, and translates each model element into a **minimal, sufficient-ish feature set** suitable for a **boosted tree** predicting **N-step log return**.

It also incorporates **range-based volatility (Parkinson volatility)** and a **range-vs-progress (efficiency) ratio** as key regime indicators—especially useful when you observe Parkinson volatility behaving like a mean-reversion signal.

---

## 1) Notation and Inputs

**Input data:** a price series $P_t > 0$ sampled at uniform intervals.  
If available, also use **OHLC** per bar: $O_t, H_t, L_t, C_t$.

Define **log price**:
$$
y_t := \ln P_t
$$

Define the **N-step log return** (prediction target):
$$
R_{t,N} := y_{t+N} - y_t
$$

All features below are computed using only historical data up to time $t$.

---

## 2) High-Level One-Step Model (Unified Dynamics)

### 2.1 Mean proxy (equilibrium)
Choose a mean proxy $m_t$. Minimal choice:
- $m_t = \text{EMA}_L(y)_t$  (EMA of log price over lookback $L$)

### 2.2 State variables (physics mapping)
**Displacement (distance from equilibrium):**
$$
d_t := y_t - m_t
$$

**Speed (1-step log return):**
$$
v_t := y_t - y_{t-1}
$$

**One-step return:**
$$
r_{t+1} := y_{t+1} - y_t
$$

### 2.3 Unified one-step equation
$$
\boxed{r_{t+1} \approx \rho_t\, v_t \;-\; \gamma_t\, d_t \;+\; \varepsilon_{t+1}}
$$

Interpretation:
- $\rho_t$ = **momentum persistence** (how much current speed carries forward)
- $\gamma_t$ = **mean-reversion strength** (how strongly displacement pulls back)
- $\varepsilon$ = residual/noise

This is one framework: momentum is the $\rho_t v_t$ term; mean reversion is the $\gamma_t d_t$ term.

---

## 3) From One-Step Model to N-Step Target

The N-step log return decomposes as a sum of one-step returns:
$$
R_{t,N} = \sum_{j=0}^{N-1} r_{t+1+j}
$$

A boosted tree can predict $R_{t,N}$ directly using features that represent:
1) the **current state** $(v_t, d_t)$, and
2) the **regime** (whether $\rho_t$ and $\gamma_t$ are effectively “high” or “low”).

---

## 4) Feature Mapping: Model Element → Related Features

This section describes what features provide information content for each element of:
$$
r_{t+1} \approx \rho_t v_t - \gamma_t d_t
$$

### 4.1 Mean proxy $m_t$
You typically compute $m_t$ only to form $d_t$. (The tree doesn’t need $m_t$ explicitly.)

- $m_t = \text{EMA}_L(\ln P)_t$  (or SMA)

### 4.2 Displacement $d_t = y_t - m_t$ (mean-reversion “force” input)
Core features:
- **Distance from mean:** $d_t$
- **Absolute distance:** $|d_t|$
- **Normalized distance (z-score):**
  $$
  z_t := \frac{d_t}{\sigma_t}
  \quad\text{where}\quad
  \sigma_t := \text{stdev}_W(\Delta y)_t,\;\Delta y_t=y_t-y_{t-1}
  $$
- **Distance change:** $d_t - d_{t-k}$
- **Mean-crossing / time-since-cross:** number of bars since $d_t$ last changed sign

Why these matter:
- Mean reversion is often nonlinear in $|d|$ (pull strengthens when stretched).
- Crossing behavior strongly differentiates chop vs trend regimes.

### 4.3 Speed $v_t = y_t - y_{t-1}$ (momentum “impulse” input)

Core idea: speed should be represented both as a **raw move** and as a **signal-to-noise (SNR)** quantity so the model can separate “real shove” from noise.

Core features:
- **Raw last return (speed):**  
  $v_t = y_t - y_{t-1}$

- **Normalized speed (SNR / z-speed):**  
  $\tilde v_t = \dfrac{y_t - y_{t-1}}{\sigma_t^*}$  
  where $\sigma_t$ is a backward-looking volatility estimate on $\Delta y$ (or Parkinson volatility if using OHLC), and $\sigma_t^*=\max(\sigma_t,\sigma_{\min})$ to avoid blow-ups in quiet regimes.

- **Multi-lag raw returns:**  
  $y_t - y_{t-k}$ for small $k$ (e.g., $k \in \{2,3,5\}$)

- **Multi-lag normalized returns (horizon-consistent SNR):**  
  $\tilde v_{t,k} = \dfrac{y_t - y_{t-k}}{\sqrt{k}\,\sigma_t^*}$  
  (the $\sqrt{k}$ keeps scaling comparable across horizons under diffusive noise)

- **Volatility level as its own feature (regime scale):**  
  include $\sigma_t^*$ (or preferably $\log \sigma_t^*$) alongside normalized speed so the model can distinguish “big ratio because vol is tiny” vs “big ratio because move is real.”

- **Trend slope (multi-step velocity proxy):**  
  slope of a linear fit of $y$ vs time over last $M$

- **Trend strength (SNR of slope):**  
  $\text{trend\_strength}_t = \dfrac{\text{slope}_t(M)}{\sigma_t^*}$ (optional but often more stable than 1-bar speed)

Why these matter:
- Raw $v_t$ captures the immediate impulse, but it is noisy; $\tilde v_t$ expresses “how many sigmas of move” occurred.
- Including $\sigma_t^*$ (or $\log \sigma_t^*$) prevents the model from overreacting to extreme ratios caused by very low volatility.
- Multi-lag returns and slope/strength capture persistent drift over multiple bars, complementing the single-step impulse.



### 4.4 Persistence regime $\rho_t$ (how sticky momentum is)

Let $y_t=\ln P_t$ and $r_t := y_t-y_{t-1}$ be the 1-step log return.

Features that proxy “momentum persistence”:

- **Lag-1 return autocorrelation (over a window of $W$ returns):**  
  Compute the sample correlation between successive returns inside the lookback window:
  $$
  \text{AC1}_t(W)
  := \text{corr}\big(\{r_{t-W+2},\dots,r_t\},\{r_{t-W+1},\dots,r_{t-1}\}\big)
  $$
  (High positive values suggest persistent momentum; negative suggests mean reversion.)

- **Run structure (sign persistence, window $M$):**  
  A simple “same-direction fraction” relative to the most recent return’s sign:
  $$
  \text{RunFrac}_t(M)
  := \frac{1}{M}\sum_{i=0}^{M-1}\mathbf{1}\Big(\text{sign}(r_{t-i})=\text{sign}(r_t)\Big)
  $$
  (Near 1 = strong same-direction streak; near 0.5 = choppy; near 0 = frequent opposite moves.)

  Optional companion feature (streak length):
  $$
  \text{RunLen}_t := \max\{k\ge 1:\ \text{sign}(r_t)=\text{sign}(r_{t-1})=\dots=\text{sign}(r_{t-k+1})\}
  $$

- **Variance ratio (trendiness proxy, horizon $q$, window $W$):**  
  Define 1-step returns $r_u=y_u-y_{u-1}$ and $q$-step returns
  $$
  r^{(q)}_u := y_u - y_{u-q}
  $$
  computed for all $u$ in the lookback window where $u-q$ exists. Then:
  $$
  \text{VR}_t(q,W)
  := \frac{\mathrm{Var}\big(\{r^{(q)}_u:\ u=t-W+1+q,\dots,t\}\big)}
          {q\cdot \mathrm{Var}\big(\{r_u:\ u=t-W+1,\dots,t\}\big)}
  $$
  Interpretation:
  - $\text{VR}\approx 1$: approximately random-walk scaling  
  - $\text{VR}>1$: positive serial correlation / trending persistence  
  - $\text{VR}<1$: negative serial correlation / mean reversion


### 4.5 Mean-reversion regime $\gamma_t$ (how strong the pullback is)

Let $y_t=\ln P_t$ and define a mean proxy $m_t$ (e.g., $m_t=\text{EMA}_L(y)_t$). Define displacement
$$
d_t := y_t - m_t
$$
and 1-step return
$$
r_t := y_t - y_{t-1}.
$$

Features that proxy “pull toward mean”:

- **Pullback slope (preferred; window $W$):**  
  This measures how strongly “being above/below the mean” predicts the *next* return back toward the mean.
  
  Build paired samples over the last $W$ bars:
  - predictor: $x_u := -d_{u-1}$ (negative displacement at time $u-1$)
  - response:  $y_u := r_u$ (the return from $u-1$ to $u$)

  for $u=t-W+1,\dots,t$.

  Then compute:
  $$
  \text{PullSlope}_t(W)
  := \frac{\sum_{u=t-W+1}^{t} (x_u-\bar x)(y_u-\bar y)}
           {\sum_{u=t-W+1}^{t} (x_u-\bar x)^2}
  $$
  where
  $$
  \bar x = \frac{1}{W}\sum_{u=t-W+1}^{t} x_u,
  \qquad
  \bar y = \frac{1}{W}\sum_{u=t-W+1}^{t} y_u.
  $$
  Interpretation:
  - large positive PullSlope: strong tendency to revert toward $m$ (higher effective $\gamma$)
  - near 0: weak pullback
  - negative: “anti-reversion” (distance tends to extend)

- **Pullback correlation (window $W$):**  
  Same samples $(x_u,y_u)$ as above, but scale-free:
  $$
  \text{PullCorr}_t(W)
  := \frac{\sum_{u=t-W+1}^{t} (x_u-\bar x)(y_u-\bar y)}
          {\sqrt{\sum_{u=t-W+1}^{t} (x_u-\bar x)^2}\;
           \sqrt{\sum_{u=t-W+1}^{t} (y_u-\bar y)^2}}
  $$

- **Mean-crossing rate (chop indicator; window $W$):**  
  Count how often the displacement changes sign:
  $$
  \text{CrossRate}_t(W)
  := \frac{1}{W-1}\sum_{u=t-W+2}^{t}\mathbf{1}\big(\text{sign}(d_u)\neq \text{sign}(d_{u-1})\big)
  $$
  Higher values indicate frequent flips around the mean (often consistent with stronger mean-reverting regimes).


### 4.6 Range-based volatility (Parkinson) as scale + regime indicator
When you observe Parkinson volatility correlating strongly with MFE/MAE and behaving like a mean-reversion indicator, treat it as BOTH:
- **scale** (how large excursions can be), and
- **regime marker** (two-way exploration / “chop” vs directional progress).

**Per-bar Parkinson range term**:
$$
g_t := \ln\left(\frac{H_t}{L_t}\right)
$$

**Rolling Parkinson volatility over window $W$** (one common form):
$$
\text{PV}_t := \sqrt{\frac{1}{4W\ln 2}\sum_{i=t-W+1}^{t} g_i^2}
$$

Notes:
- PV is directly tied to the **high–low range**, which naturally links to **extremes** (MFE/MAE).
- If your MAE is stored as a negative number, PV will often show **positive corr with MFE** and **negative corr with MAE** purely by sign convention (larger adverse excursion → more negative MAE).

### 4.7 Range vs progress: “Efficiency ratio” to separate breakout-range from chop-range (OHLC)

High Parkinson volatility (large high–low range) can come from either:
- **directional expansion** (breakout/trend): big range + meaningful net progress, or
- **two-way exploration** (chop/mean-reversion): big range + little net progress.

The efficiency ratio measures **net progress relative to intrabar range**.

#### 4.7.1 Per-bar efficiency (no window)
Using OHLC for bar $t$:
- $O_t$ = open, $H_t$ = high, $L_t$ = low, $C_t$ = close

Define log-range and log-net-move:
$$
\text{Range}_t := \left|\ln H_t - \ln L_t\right|
$$
$$
\text{Net}_t := \left|\ln C_t - \ln O_t\right|
$$

Then the per-bar efficiency ratio is:
$$
\text{Eff}_t := \frac{\text{Net}_t}{\text{Range}_t + \epsilon}
= \frac{\left|\ln C_t - \ln O_t\right|}{\left|\ln H_t - \ln L_t\right| + \epsilon}
$$

Interpretation:
- $\text{Eff}_t \approx 1$: most of the range was directional progress (trend-like bar)
- $\text{Eff}_t \approx 0$: lots of range but little net change (two-way chop / mean reversion)

#### 4.7.2 Windowed efficiency (recommended for regime stability)
Single bars can be noisy. Build a windowed version over the last $W$ bars:

Compute for each bar $u$ in the window:
$$
\text{Range}_u := \left|\ln H_u - \ln L_u\right|,\quad
\text{Net}_u := \left|\ln C_u - \ln O_u\right|
$$

Then aggregate:
$$
\text{EffAvg}_t(W) := \frac{\sum_{u=t-W+1}^{t}\text{Net}_u}{\sum_{u=t-W+1}^{t}\text{Range}_u + \epsilon}
$$

Interpretation:
- high $\text{EffAvg}$: sustained directional progress relative to range (momentum-friendly regime)
- low $\text{EffAvg}$: sustained two-way exploration (mean-reversion-friendly regime)

Practical note:
- Use a small $\epsilon$ to avoid division issues when ranges are tiny.
- Often include both $\text{Eff}_t$ (instant) and $\text{EffAvg}_t(W)$ (regime) as separate features.


### 4.8 Noise / scale (when signals matter)
- **Close-to-close volatility:** $\sigma_t = \text{stdev}_W(\Delta y)$
- Optional: **vol-of-vol** (stdev of rolling $\sigma_t$)

---

## 5) Minimal Feature Set for Predicting $R_{t,N}$ with a Boosted Tree

You want the smallest set that allows the tree to learn:
- state $(d_t, v_t)$,
- whether momentum persists ($\rho$-like),
- whether pullback dominates ($\gamma$-like),
- and whether the regime is “two-way exploration” (PV + low efficiency).

### 5.1 Minimal base features (single scale)
1) **Distance from mean:** $d_t = \ln P_t - \text{EMA}_L(\ln P)_t$  
2) **Speed:** $v_t = \ln P_t - \ln P_{t-1}$  
3) **Persistence proxy:** $\text{AC1}_t = \text{corr}_W(\Delta y_t,\Delta y_{t-1})$  
4) **Pullback proxy:** $\text{PullSlope}_t = \mathrm{Cov}_W(\Delta y_{t+1},-d_t)/\mathrm{Var}_W(d_t)$  
5) **Parkinson volatility:** $\text{PV}_t$ (range-based)  
6) **Efficiency ratio:** $\text{Eff}_t$ (range vs progress)

### 5.2 Minimal interactions (critical)
7) **Stretch × speed interaction:**
$$
d_t \cdot v_t
$$
This lets the model distinguish “moving away from mean” vs “moving back toward mean”.

8) **Range-chop interaction (mean-reversion gate):**
$$
\text{PV}_t \cdot (1-\text{Eff}_t)
$$
High PV with low efficiency is a strong “violent chop / reversion” fingerprint.

9) **Range × stretch nonlinearity (reversion strength increases with stretch):**
$$
\text{PV}_t \cdot |d_t|
$$

If you must keep interactions to just one: keep **$d_t\cdot v_t$** first; next best is **$\text{PV}_t(1-\text{Eff}_t)$**.

### 5.3 Multi-scale extension (recommended, still lightweight)
Compute the same minimal set at **two time scales**:
- “Fast” scale: $W_1 \approx N$
- “Slow” scale: $W_2 \approx 4N$

This helps the tree infer regime persistence without extra modeling.

---

## 6) Diagnostics / Sanity Checks (Recommended)

These checks help confirm whether PV is acting as:
- amplitude-only (excursion scaler), or
- a true mean-reversion regime marker.

1) Compare:
- $\text{corr}(\text{PV}_t,\;|R_{t,N}|)$ vs $\text{corr}(\text{PV}_t,\;R_{t,N})$  
If |return| dominates, PV is mostly amplitude.

2) Bucket by efficiency:
- Low-efficiency ($\text{Eff}$ small): does PV predict stronger reversion?
- High-efficiency ($\text{Eff}$ large): does PV instead line up with continuation?

3) Control for state:
- Does PV still add signal after conditioning on $d_t$ and $v_t$?
If yes, PV is capturing extra regime structure (not just proxying for trend/stretch).

---

## 7) Practical Defaults (Simple Choices)

- Use $y_t=\ln P_t$
- Mean proxy: $m_t=\text{EMA}_{L}(y)_t$, with $L \approx 4N$
- Feature windows:
  - $W_1 \approx N$
  - $W_2 \approx 4N$ (optional second scale)
- If only one window: set $W \approx 4N$

---

## 8) Summary

**Unified model:**  
$$
r_{t+1} \approx \rho_t v_t - \gamma_t d_t
$$

**Tree needs:**
- state: $d_t, v_t$
- regime: persistence proxy (autocorr), pullback proxy (PullSlope)
- range regime: Parkinson volatility + efficiency ratio
- interactions: $d_t v_t$, PV×(1−Eff), PV×|d|

This is a compact feature-based implementation of a single framework that can express both momentum and mean reversion while targeting:
$$
R_{t,N}=\ln(P_{t+N})-\ln(P_t)
$$

---
