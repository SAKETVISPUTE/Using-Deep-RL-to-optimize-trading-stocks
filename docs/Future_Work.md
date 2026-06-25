# Future Work (V2 Master Goal Roadmap)

This document details the research roadmap for Version 2, structured as controlled campaigns to systematically improve out-of-sample trading performance.

---

## 1. Research Campaign 1 — State Representation (Progressive Tuning)
Our objective is to build richer observation vectors incrementally. We will train and test models by adding one feature category at a time, evaluating performance against the Version 1 baseline:

* **State 0 (Baseline)**: OHLCV + Current portfolio state (V1).
* **State 1 — Price Dynamics**: Add intraday return, gap return, high-low range, rolling mean returns, and rolling standard deviation returns.
* **State 2 — Trend Indicators**: Add SMAs (5, 10, 20, 50), EMAs (10, 20, 50), and MACD lines.
* **State 3 — Momentum Indicators**: Add RSI, ROC, Stochastic (%K, %D), Williams %R, and CCI.
* **State 4 — Volatility Indicators**: Add ATR, Bollinger Width, Historical Volatility, and Rolling Variance.
* **State 5 — Volume Indicators**: Add OBV, Volume Moving Average, Volume Ratio, and Chaikin Money Flow.
* **State 6 — Portfolio State**: Add Average Buy Price, Cash Ratio, Position Exposure, Unrealized/Realized PnL, and Time Since Last Trade.
* **State 7 — Market Context**: Add NIFTY 50 returns, Sector returns, Rolling Market Volatility, and Relative Stock Strength vs NIFTY.
* **State 8 — Full State**: Combine all validated feature groups.

---

## 2. Research Campaign 2 — Reward Engineering
We will evaluate alternative reward formulations using the best state representation identified in Campaign 1:
* **Reward 1 (Baseline)**: Daily Portfolio Return.
* **Reward 2**: Portfolio Return minus Transaction Cost.
* **Reward 3**: Portfolio Return minus Drawdown Penalty.
* **Reward 4**: Portfolio Return minus Volatility Penalty.
* **Reward 5**: Differential Sharpe Reward.
* **Reward 6**: Differential Sortino Reward.
* **Reward 7 (Hybrid)**: Portfolio Return + Transaction Cost + Drawdown + Volatility.

---

## 3. Research Campaign 3 — Action Space
Test progressively richer action spaces:
* **Version A (Baseline)**: Hold (0), Buy (1), Sell (2).
* **Version B**: Fractional trading: Buy 25%, Buy 50%, Buy 100%, Hold, Sell 25%, Sell 50%, Sell 100% (7 discrete actions).

---

## 4. Research Campaign 5 — RL Algorithms
If justified, compare our tuned PPO model against:
* **SAC (Soft Actor-Critic)**: If we implement continuous action spaces.
* **A2C (Advantage Actor-Critic)**: To compare training speed and stability.
