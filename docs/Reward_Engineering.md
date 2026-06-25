# Reward Engineering (Baseline - V1)

This document describes the reward formulation used in the baseline PPO agent.

---

## 1. Reward Formulation: Portfolio Return

The default reward function in the baseline environment is the **simple daily return** of the portfolio value:

$$Reward_t = \frac{PV_{t+1} - PV_t}{PV_t}$$

Where:
* $PV_t$ is the portfolio value at step $t$: $PV_t = Cash_t + Holdings_t \times Close_t$
* $PV_{t+1}$ is the portfolio value at step $t+1$: $PV_{t+1} = Cash_{t+1} + Holdings_{t+1} \times Close_{t+1}$

---

## 2. Rationale & Financial Interpretation
This reward directly aligns the RL agent's objectives with net capital growth. Maximizing the expected cumulative sum of daily returns:

$$\max \mathbb{E} \left[ \sum_{t=0}^{T} Reward_t \right]$$

is mathematically equivalent to maximizing the compounding growth rate of the trading account over the investment period.

---

## 3. Potential Drawbacks & Limitations
* **Volatility Blindness**: The simple portfolio return reward does not penalize risk. It values a $1\%$ daily gain achieved with high volatility identically to a $1\%$ daily gain achieved with zero volatility. This can lead to the agent holding highly volatile assets during market drawdowns rather than moving to cash.
* **No Capital Drawdown Penalty**: The agent is not penalized for experiencing deep peak-to-trough drawdowns, which makes it risk-tolerant and susceptible to extreme losses.
* **Whipsaw Sensitiveness**: In choppy horizontal markets, the simple return reward might not provide clear gradients to distinguish between short-term noise and long-term trend changes.
