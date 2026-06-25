# Strategy Evaluation & Backtesting

This document details the evaluation methodology, backtesting engine, performance metrics, and initial backtest results for our reinforcement learning agents.

The backtesting module is implemented in [backtester.py](file:///mnt/c/Users/Saket/Desktop/Projects/Finsearch_RL/src/evaluation/backtester.py) and executed via [run_eval.py](file:///mnt/c/Users/Saket/Desktop/Projects/Finsearch_RL/src/evaluation/run_eval.py). The output performance plot is saved in [reliance_ns_backtest.png](file:///mnt/c/Users/Saket/Desktop/Projects/Finsearch_RL/results/reliance_ns_backtest.png).

---

## 1. Objective
To construct a deterministic evaluation pipeline that evaluates trained policies under realistic market constraints (fees and slippage). The module must generate comparative performance curves against baseline strategies, track drawdowns, and output standard quantitative metrics.

---

## 2. Backtester Architecture

The [Backtester](file:///mnt/c/Users/Saket/Desktop/Projects/Finsearch_RL/src/evaluation/backtester.py#L11) class handles deterministic policy rollouts.

### Deterministic Rollout Loop
Unlike training (where the policy samples actions stochastically from a probability distribution to encourage exploration), during evaluation we use a **deterministic policy**.
* **Action Prediction**: `action, _ = model.predict(observation, deterministic=True)`
* This selects the action with the highest logit output (argmax), reflecting the model's optimal learned behavior.
* The loop runs until the environment registers a `terminated` flag.

### Metrics Computed
We calculate standard financial metrics (Cumulative Return, CAGR, Volatility, Sharpe, Sortino, Max Drawdown, Calmar, and Trades). Additionally, we compute:
1. **Trade Win Rate (%)**:
   - A trade round is defined as: An entry trade (e.g. Buy when holding 0 shares) to an exit trade (e.g. Sell to liquidate).
   - **Win Rate** = $\frac{\text{Number of profitable trade rounds}}{\text{Total completed trade rounds}} \times 100$.
2. **Completed Trades**: The total count of completed trade rounds (entry to exit).

---

## 3. Performance Visualization

The plotting engine generates a multi-panel visual analysis:
1. **Upper Panel**: Tracks the daily portfolio value (₹) of the RL agent alongside the passive `Buy & Hold` baseline. It overlays green upward triangles ($\blacktriangle$) for Buy executions and red downward triangles ($\blacktriangledown$) for Sell executions to show exactly where the model placed trades.
2. **Lower Panel**: An **Underwater Plot** tracking the drawdowns of both the agent and the baseline. This shows the depth and duration of peak-to-trough losses, allowing researchers to evaluate the strategy's risk profile during market corrections.

---

## 4. Evaluation Results: Reliance Case Study

Running the backtest on the trained `RELIANCE.NS` PPO model (20,000 steps training) returned:

* **Final Portfolio Value**: ₹810,180.06 (Initial: ₹100,000)
* **Cumulative Return**: 710.18%
* **Annualized Return (CAGR)**: 21.43%
* **Annualized Volatility**: 26.62%
* **Sharpe Ratio**: 0.6372
* **Maximum Drawdown**: -45.09%
* **Trades Executed**: 1 (Buy at step 1, Hold until end)

### Quantitative Research Insight
With only 20,000 training timesteps, the PPO agent converged to a **passive Buy & Hold policy**.
In a strong historical bull market (like Reliance from 2015 to 2025), holding the asset yields high returns. Because active trading incurs transaction fees (0.1%) and slippage (0.05%), the agent's policy network learned that any frequent buying/selling would reduce returns due to trading friction. It discovered that the optimal deterministic strategy is to buy immediately and hold, effectively matching the strongest heuristic baseline.

---

## 5. Limitations & Future Directions
- **In-Sample Bias**: This evaluation was run on the same data used for training. To verify if the agent has learned real trading patterns (rather than just memorizing a bull market), we must implement **Out-of-Sample Validation** (Stage 9).
- **Short-Selling**: The environment currently restricts short-selling. Enabling short-selling would allow the agent to trade during market downtrends, which might lead to active short strategies instead of passive holding.
