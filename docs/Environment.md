# Trading Environment Design

This document details the architecture, design choices, financial realism mechanisms, and state-action definitions of our reinforcement learning trading environment.

The codebase is implemented in [trading_env.py](file:///mnt/c/Users/Saket/Desktop/Projects/Finsearch_RL/src/environment/trading_env.py) and verified via [test_trading_env.py](file:///mnt/c/Users/Saket/Desktop/Projects/Finsearch_RL/tests/test_trading_env.py).

---

## 1. Objective
To construct a Gymnasium-compliant reinforcement learning environment that accurately simulates the mechanics of trading a single asset. The environment must incorporate trading costs (slippage and brokerage fees), enforce capital constraints, track portfolio metrics, and expose standard interfaces (`reset`, `step`, `observation_space`, `action_space`) to interface with off-the-shelf RL libraries (like Stable-Baselines3).

---

## 2. Environment Architecture & Code Reference

Our environment is defined as the [TradingEnv](file:///mnt/c/Users/Saket/Desktop/Projects/Finsearch_RL/src/environment/trading_env.py#L6) class, which inherits from `gymnasium.Env`.

### Action Space
We utilize a discrete action space of size 3:
* **`0` - Hold**: Do nothing. Keep existing position and cash.
* **`1` - Buy**: Deploy cash to purchase shares of the asset.
* **`2` - Sell**: Liquidate the current position to cash.

The fraction of capital allocated for buying/selling is configured via the parameter `trade_volume_fraction` (default is `1.0`, representing full-size allocation where Buy deploys all cash and Sell liquidates all shares).

### Observation Space
The state vector is flat and contains:
1. **Engineered Technical & Market Features** (16 dimensions):
   - Returns: `Daily_Return`, `Log_Return`
   - Trend: `EMA_10`, `EMA_30`, `MACD`, `MACD_Signal`, `MACD_Hist`
   - Momentum: `RSI`, `ROC`
   - Volatility: `ATR`, `BB_Middle`, `BB_Std`, `BB_Upper`, `BB_Lower`
   - Volume: `Volume_Change`
   - Market Benchmark: `Market_Return`
2. **Portfolio Features** (3 dimensions):
   - **Normalized Cash**: $\frac{\text{Cash}_t}{\text{Portfolio Value}_t}$
   - **Normalized Position**: $\frac{\text{Holdings}_t \times Close_t}{\text{Portfolio Value}_t}$
   - **Normalized Cumulative Return**: $\frac{\text{Portfolio Value}_t - \text{Initial Cash}}{\text{Initial Cash}}$

The total observation dimension is $16 + 3 = 19$. Including portfolio state variables is vital; without them, the agent cannot know if it currently holds shares or cash, meaning it could not learn to make sell decisions or manage drawdowns.

---

## 3. Financial Realism Mechanisms

Many academic RL trading models suffer from "unrealistic profitability" because they ignore real-world market constraints. We address this with three mechanisms:

### A. Transaction Fees
Every transaction incurs fees (brokerage, exchange charges, securities transaction tax, GST). We implement a configurable fee rate $\text{fee} = 0.001$ (0.1% of the traded volume).
* **Buy Execution**: Total cost per share purchased is $P_{exec} \times (1 + \text{fee})$.
* **Sell Execution**: Net cash received per share liquidated is $P_{exec} \times (1 - \text{fee})$.

### B. Price Slippage
In real markets, placing a market order incurs slippage—the executed price is worse than the last observed price because of order book depth and latency. We implement a configurable slippage rate $\text{slippage} = 0.0005$ (0.05% price penalty).
* **Execution Buy Price**: $P_{buy} = P_{close} \times (1 + \text{slippage})$ (buying higher than market close).
* **Execution Sell Price**: $P_{sell} = P_{close} \times (1 - \text{slippage})$ (selling lower than market close).

### C. Capital & Portfolio Constraints
* **No Leverage**: The agent cannot borrow funds. Cash must remain $\ge 0$.
* **No Short Selling**: The agent cannot sell shares it does not own. Holdings must remain $\ge 0$.
* **Portfolio Value Equation**: 
  $$PV_t = \text{Cash}_t + (\text{Holdings}_t \times Close_t)$$

---

## 4. Step and Reward Formulation

At each step, the environment processes:
1. Trade execution at the current day's Close price (incorporating slippage and fees).
2. Time transition to the next trading day.
3. Revaluation of the portfolio value at the next day's Close price.
4. Reward calculation.

### Reward Options
To support reward shaping studies, we implement three options:
1. **Simple Portfolio Return**:
   $$R_t = \frac{PV_{t+1} - PV_t}{PV_t}$$
2. **Log Return**:
   $$R_t = \ln\left(\frac{PV_{t+1}}{PV_t}\right)$$
3. **Risk-Adjusted Return**:
   $$R_t = \frac{PV_{t+1} - PV_t}{PV_t} - \gamma \cdot \sigma_t \cdot w_t$$
   Where $\sigma_t$ is the normalized rolling price volatility, $w_t$ is the position weight, and $\gamma$ is a risk penalty weight. This penalizes the agent for holding volatile assets.

---

## 5. Verification Results
We ran a random agent simulation over the entire Reliance history (2,714 trading days):
* **Compliance**: Environment fully conforms to Gymnasium API (returned correct observation shape of 19 and correct types).
* **Transaction Cost Impact**: The random agent's initial capital of ₹100,000 was reduced to ₹49,314.99 (-50.69% return) due to friction from random buying and selling. This demonstrates that transaction fees and slippage are actively functioning to penalize over-trading, setting a realistic baseline.

---

## 6. Limitations & Future Improvements
* **Execution Model**: Trades are executed at the day's close price. A more advanced model could execute at the next day's Open price to prevent same-day execution bias.
* **Discrete Sizing**: The agent currently trades in binary blocks (full size). Adding multi-size allocations (fractional positions) could allow more nuanced accumulation and distribution strategies.
