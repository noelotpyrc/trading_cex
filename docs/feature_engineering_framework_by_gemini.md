# The Energy-State-Force (ESF) Framework

**Objective:** To model financial price action as a physical system governed by vector mechanics. By separating variables into **Energy State** (Kinetic vs. Potential), **Active Forces** (External Impulse), and **Environmental Regime** (Turbulence), we allow machine learning models to solve for the "Net Vector" rather than memorizing historical price patterns.

---

## I. Theoretical Basis: The Vector Equation

We model the future price path $P_{t+\tau}$ not as a random walk, but as the result of forces acting on a particle within a viscous medium. The "Target" is simply the net result of these competing vectors.

$$P_{t+\tau} \approx P_t + \underbrace{(v_t \cdot \tau)}_{\text{Inertial Drift}} - \underbrace{k(P_t - P_{mean})}_{\text{Structural Tension}} + \underbrace{F_{ext}}_{\text{Flow Impact}} + \underbrace{\mathcal{N}(0, \sigma_{regime})}_{\text{Regime Turbulence}}$$

* **Inertial Drift (Kinetic):** The displacement due strictly to current velocity (Momentum).  
* **Structural Tension (Potential):** The restoring counter-vector proportional to the distance from liquidity equilibrium (Mean Reversion).  
* **Flow Impact (Force):** The injection of new energy (Alpha) by aggressive participants.  
* **Regime Turbulence (Noise):** The stochastic error term that degrades the reliability of the signal vectors.

---

## II. Strategic Philosophy: Measurement > Prediction

In mature markets, "True Alpha" (predicting information before it happens) is the domain of HFT and prop firms. For this framework, we redefine "Alpha" not as a crystal ball, but as a **Wind Gauge**.

* **The Goal:** We do not attempt to predict the *origin* of the Force ($F_{ext}$). We aim to **measure** the presence and intensity of the Force faster and more accurately than standard retail indicators.  
* **The Edge:** Our edge comes from **Risk Premia** (holding risk during structural moves) and **Physics-Based Reaction** (identifying when Inertia has overcome Tension), rather than information asymmetry.  
* **The Mechanism:**  
  * **Prop Firm:** "I predict a buy order *will* arrive." (Predictive Alpha).  
  * **ESF Model:** "I measure that Kinetic Energy has just exceeded Potential Energy in a Low Turbulence regime." (Reactive/Structural Edge).

---

## III. Feature Engineering Pillars

Features are categorized by which term of the physical equation they estimate. All features must be normalized (dimensionless) to be robust across different market regimes.

**Layer 1: The Energy State (System Balance)**

**Purpose:** To define the current mechanical equilibrium. We separate the energy moving the price forward (Kinetic) from the energy pulling it backward (Potential).

**Key Principle:** Base → Derived. First measure **Base Components** independently, then combine them into **Derived Ratios** that are dimensionless and stationary.

---

### Base Components

These are the raw building blocks. Each should be measured independently.

| Component | Abstraction | Examples |
| :---- | :---- | :---- |
| **Velocity ($\vec{v}$)** | Lag-minimized directional speed. The "drift" term. | Kalman Filter Slope, Linear Regression Slope, Hull MA Delta. |
| **Center of Mass ($P_{mean}$)** | Fair value anchor. The gravity well where potential energy is zero. | VWAP, Rolling EMA, Fair Value Model. |
| **Volatility ($\sigma$)** | Dispersion of price. The normalizing denominator for all ratios. | Parkinson HV, ATR, Rolling Std of Returns. |

---

### Derived Ratios (Stationary, Dimensionless)

These combine base components into comparable, regime-robust signals.

| Ratio | Formula | Interpretation |
| :---- | :---- | :---- |
| **Kinetic State ($K$)** | $K = \frac{\vec{v}}{\sigma} \sqrt{\tau}$ | **Forward Vector.** Momentum efficiency — velocity adjusted for noise. High $K$ = strong, clean trend. |
| **Potential State ($U$)** | $U = \frac{P_t - P_{mean}}{\sigma}$ | **Restoring Vector.** Spring tension — how stretched price is from equilibrium. High $|U|$ = mean-reversion pressure. |

---

### Interaction Logic

* A high **Kinetic State ($K$)** is bullish *only if* **Potential State ($U$)** is not critical (e.g., $|U| < 2.0\sigma$).
* If $|U|$ is extreme and $K$ is high → "Blow-off Top" rather than sustainable trend.
* If $|U|$ is extreme and $K$ is near zero → Spring snap (mean reversion).

### ---

**Layer 2: The Active Forces (Impulse & Drag)**

**Purpose:** To quantify the external forces acting to disrupt the current energy state. In ESF, we separate **Impulse** (Aggressive orders pushing price) from **Friction** (Passive orders slowing price).

**Key Principle:** Impulse Normalization. Forces must be Z-Scored to represent "Shock" relative to recent history.

| Component | Abstraction | Examples of Implementation |
| :---- | :---- | :---- |
| **Impulse Injection ($F_{ext}$)** | **The Accelerator.** Measures aggressive capital entering to override the Potential State. | Net Taker Volume Z-Score, CVD (Cumulative Volume Delta) Slope, OI Change Rate. |
| **Friction Density ($F_{drag}$)** | **The Brake.** Measures the "viscosity" of the order book that absorbs Kinetic Energy. | Market Depth (Bid/Ask) imbalance, Limit Order Book density, "Iceberg" detection. |
| **Force Efficiency** | The ratio of Impulse to Price Change. How much fuel is required to move 1 unit? | $\frac{\Delta Price}{\Delta Volume}$ (Kyle's Lambda), Amihud Illiquidity Proxy. |

* **Engineering Note:** This layer answers the "Escape Velocity" question. If **Potential State ($U$)** is high (heavy gravity), the price requires a massive **Impulse Injection ($F_{ext}$)** to continue trending. If $U$ is high and $F_{ext}$ is low, the model predicts Reversion.

### ---

**Layer 3: The Environmental Regime (Medium)**

**Purpose:** To define the properties of the medium (Market) to determine if the physics calculations are reliable.

**Key Principle:** Entropy measurement. High entropy (Turbulence) degrades the reliability of the Energy State vectors.

| Component | Abstraction | Examples of Implementation |
| :---- | :---- | :---- |
| **Turbulence Factor** | **System Entropy.** Is the random noise term overpowering the signal vectors? | Ratio of Short-term Volatility / Long-term Volatility ($\frac{\sigma_{fast}}{\sigma_{slow}}$). |
| **Coupling State** | **External Gravity.** Is the asset being pulled by a larger massive object (Index/Sector)? | Rolling Correlation with SPX/BTC. (High coupling = Internal physics matter less). |
| **Liquidity Phase** | **Structural Integrity.** Is the medium solid (thick) or gaseous (thin)? | Bid-Ask Spread Ratio, Average Tick Size variability. |

* **Engineering Note:** This acts as the "Confidence Weight." If **Turbulence > 1.5**, the ML model should effectively "zero out" the Kinetic and Potential inputs because the vector mechanics are being drowned out by noise.

---

## IV. Machine Learning Logic (The Vector Sum)

By feeding these specific inputs, a Tree-Based Model (e.g., XGBoost) effectively approximates the vector sum logic:

1. **Trend Scenario (Escape Velocity):**  
   * **State:** Inertial Drift ($K$) is **Positive**.  
   * **Force:** Flow Impact ($F_{ext}$) is **Positive**.  
   * **Logic:** Forward Vector > Restoring Vector.  
   * **Result:** High Probability of Trend Continuation.  

2. **Mean Reversion Scenario (Spring Snap):**  
   * **State:** Structural Tension ($U$) is **High** (Price far from VWAP).  
   * **State:** Inertial Drift ($K$) is **Near Zero** (Momentum stalled).  
   * **Logic:** Restoring Vector > Forward Vector.  
   * **Result:** High Probability of Reversal.  

3. **Regime Filter (Turbulence):**  
   * **Regime:** Turbulence Factor is **Extreme**.  
   * **Logic:** The Noise term $\mathcal{N}$ drowns out both vectors.  
   * **Result:** Prediction confidence converges to 0. No Trade.