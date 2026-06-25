# State Representation (Baseline - V1)

This document describes the design and components of the baseline observation space fed to the RL agent.

The observation space is built inside the `_get_observation` method of [trading_env.py](file:///mnt/c/Users/Saket/Desktop/Projects/Finsearch_RL/src/environment/trading_env.py#L142).

---

## 1. State Space Structure

The observation vector is a flat 19-dimensional array containing:

### A. Technical & Market Features (16 dimensions)
These features are engineered using [feature_engineer.py](file:///mnt/c/Users/Saket/Desktop/Projects/Finsearch_RL/src/features/feature_engineer.py) and represent the market's technical state for the current day step:

* **Returns** (2 dimensions): `Daily_Return`, `Log_Return`
* **Trend** (5 dimensions): `EMA_10`, `EMA_30`, `MACD`, `MACD_Signal`, `MACD_Hist`
* **Momentum** (2 dimensions): `RSI`, `ROC`
* **Volatility** (5 dimensions): `ATR`, `BB_Middle`, `BB_Std`, `BB_Upper`, `BB_Lower`
* **Volume** (1 dimension): `Volume_Change`
* **Market Context** (1 dimension): `Market_Return` (returns of the NIFTY 50 benchmark index)

### B. Portfolio State Features (3 dimensions)
These features represent the agent's internal cash and asset positions, normalized relative to the current total portfolio value:

1. **Normalized Cash** ($CashRatio_t$):
   $$CashRatio_t = \frac{\text{Cash}_t}{\text{Portfolio Value}_t}$$
   Represents the proportion of the portfolio currently in cash. Bounded between $[0.0, 1.0]$.
2. **Normalized Position** ($PositionRatio_t$):
   $$PositionRatio_t = \frac{\text{Holdings}_t \times Close_t}{\text{Portfolio Value}_t}$$
   Represents the proportion of the portfolio currently exposed to the stock. Bounded between $[0.0, 1.0]$.
3. **Normalized Cumulative Return** ($CumReturn_t$):
   $$CumReturn_t = \frac{\text{Portfolio Value}_t - \text{Initial Cash}}{\text{Initial Cash}}$$
   Represents the overall percentage return achieved by the agent since the beginning of the episode.

---

## 2. Dimensionless Ratio Scaling (Gradient Safety)

To prevent neural network gradient explosion (which leads to `nan` logits output in PyTorch), all price-denominated indicators are transformed into dimensionless ratios relative to the current Close price:
- **EMAs**: scaled as $(Close - EMA)/Close$
- **Bollinger Bands**: scaled as $(BB\_Limit - Close)/Close$
- **RSI / ROC**: divided by $100.0$ to scale into $[0, 1]$
- **Volume Change**: clipped to $[-3.0, 3.0]$

This scaling ensures that all 19 features are approximately in the range of $[-1.0, 1.0]$, maintaining network training stability.
