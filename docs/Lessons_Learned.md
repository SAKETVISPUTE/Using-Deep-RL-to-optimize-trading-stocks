# Lessons Learned

This document summarizes the core challenges, solutions, and engineering takeaways discovered during this project.

---

## 1. Technical Indicators Must Be Stationary
Feedforward neural networks are poor at handling absolute price data because prices trend upward over time (non-stationary). Feeding raw prices into the network causes policy gradient overflow and failure to generalize. 

**Takeaway**: Always convert price-denominated technical indicators (like EMAs, Bollinger Bands, MACD, and standard deviations) into **dimensionless ratios** relative to the current Close price. This bounds inputs to a small range (typically $[-1.0, 1.0]$) and prevents weight overflow (`nan` logits).

---

## 2. Transaction Costs Shift Policies Toward Passive Holding
When transaction fees (0.1%) and slippage (0.05%) are applied to trades, executing a round trade (Buy then Sell) costs approximately $0.3\%$ of the traded volume. 
If an agent trades daily:
- Over a year (252 steps), it would pay a massive fee drag of over $30-50\%$ of its portfolio value.
- In a strong bull market, simple passive holding has zero trading cost after day one and benefits from market trend drift.
- Consequently, under a simple portfolio return reward, PPO agents quickly learn that active trading is negative-sum and converge to a passive "buy-and-hold" strategy.

**Takeaway**: To train an active trading agent, the environment must contain a reward function that penalizes risk/drawdowns or the state space must include features that identify short-term trend reversals, combined with fractional size actions so the agent can scale in and out of positions.

---

## 3. Walk-Forward Validation Prevents Regime Bias
Evaluating the model on a single test period can lead to false conclusions. For example, a model evaluated only on 2023 would look highly successful simply because tech and energy stocks were in a strong rally. Walk-forward validation forces the agent to train and test across different years (including downturns like 2022), providing a robust, realistic, and unbiased measure of its out-of-sample risk-adjusted returns.
