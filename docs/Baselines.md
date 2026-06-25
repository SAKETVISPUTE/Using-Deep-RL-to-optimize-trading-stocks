# Heuristic & Financial Baselines

This document describes the implementation, performance metrics, and evaluation results of the heuristic trading strategies that serve as the baseline comparison for our reinforcement learning agents.

The codebase is located in [baseline_strategies.py](file:///mnt/c/Users/Saket/Desktop/Projects/Finsearch_RL/src/baselines/baseline_strategies.py) and executed via [run_baselines.py](file:///mnt/c/Users/Saket/Desktop/Projects/Finsearch_RL/src/baselines/run_baselines.py). The results are saved in [baselines_comparison.csv](file:///mnt/c/Users/Saket/Desktop/Projects/Finsearch_RL/results/baselines_comparison.csv).

---

## 1. Objective
To build standard, reproducible trading strategies to establish benchmark metrics. By running the baselines in the same [TradingEnv](file:///mnt/c/Users/Saket/Desktop/Projects/Finsearch_RL/src/environment/trading_env.py#L6) configuration, we ensure that transaction costs (0.1%) and price slippage (0.05%) are applied equally, preventing unfair comparison.

---

## 2. Baseline Strategies Defined

We implement four heuristic strategies:

### A. Buy and Hold (`BuyAndHoldStrategy`)
- **Logic**: Executes a single buy action at $t=0$, investing all available capital into the asset. It holds this position until the end of the 11-year dataset.
- **Financial Rationale**: Represents the passive investing approach. In long-term bull markets, passive holding is notoriously difficult for active strategies to beat because it incurs zero transaction costs after day one.

### B. Random Trading (`RandomStrategy`)
- **Logic**: Selects randomly among Hold (0), Buy (1), and Sell (2) at each day step with equal probability.
- **Financial Rationale**: Establishes a floor for performance. If an RL agent performs worse than random, its learning has failed. Furthermore, it measures the devastating compounding effect of transaction costs and slippage on high-frequency, non-strategic trading.

### C. EMA Crossover (`EMACrossoverStrategy`)
- **Logic**: 
  - Buy when the 10-day Exponential Moving Average (fast) crosses *above* the 30-day Exponential Moving Average (slow).
  - Sell when the 10-day EMA crosses *below* the 30-day EMA.
- **Financial Rationale**: A classic trend-following strategy. Moving average crossovers attempt to capture long-term momentum shifts while exiting positions when the trend reverses.

### D. RSI Mean Reversion (`RSIStrategy`)
- **Logic**:
  - Buy when the Relative Strength Index (RSI-14) falls below 30 (oversold).
  - Sell when the RSI-14 rises above 70 (overbought).
- **Financial Rationale**: A mean-reversion strategy. It assumes that extreme price moves are temporary and that prices will regress to their historical average.

---

## 3. Key Quantitative Metrics Explained

To evaluate these strategies, we compute the following performance metrics:
1. **Cumulative Return**: The total percentage growth of the portfolio: $\frac{PV_{final} - PV_{initial}}{PV_{initial}} \times 100$.
2. **Annualized Return (CAGR)**: The geometric average return per year over the period: $(1 + R_{cum})^{252/N} - 1$.
3. **Annualized Volatility**: The standard deviation of daily returns scaled to annual terms: $\sigma_{daily} \times \sqrt{252}$.
4. **Sharpe Ratio**: A measure of risk-adjusted return: $\frac{\text{Mean}(R_d) - R_{rf}}{\text{Std}(R_d)} \times \sqrt{252}$ (using risk-free rate $R_{rf} = 6.0\%$ annualized).
5. **Maximum Drawdown (Max DD)**: The largest peak-to-trough peak percentage loss in portfolio value: $\min\left(\frac{PV_t - \max(PV_{1..t})}{\max(PV_{1..t})}\right)$.
6. **Calmar Ratio**: Annualized Return divided by the absolute Maximum Drawdown. A higher Calmar ratio indicates better risk-adjusted return relative to extreme losses.

---

## 4. Evaluation Results

Below is the comparative summary of the baseline strategies across the four constituent stocks for the period 2015-2015 to 2025-2025:

| Stock | Strategy | Final Value (₹) | Cumulative Return | Annualized Return | Sharpe Ratio | Max Drawdown | Trades |
|---|---|---|---|---|---|---|---|
| **RELIANCE** | Buy and Hold | 809,207.75 | 709.21% | 21.42% | 0.63 | -45.09% | 1 |
| | Random | 36,644.56 | -63.36% | -8.90% | -0.70 | -69.63% | 894 |
| | EMA Crossover | 319,845.28 | 219.85% | 11.40% | 0.34 | -32.66% | 89 |
| | RSI Strategy | 218,665.49 | 118.67% | 7.53% | 0.16 | -41.21% | 66 |
| **TCS** | Buy and Hold | 321,985.94 | 221.99% | 11.46% | 0.33 | -34.45% | 1 |
| | Random | 27,038.14 | -72.96% | -11.43% | -1.00 | -73.91% | 896 |
| | EMA Crossover | 123,036.90 | 23.04% | 1.94% | -0.16 | -41.32% | 103 |
| | RSI Strategy | 226,739.58 | 126.74% | 7.89% | 0.18 | -29.57% | 74 |
| **HDFCBANK** | Buy and Hold | 455,457.85 | 355.46% | 15.11% | 0.48 | -41.05% | 1 |
| | Random | 105,432.92 | 5.43% | 0.49% | -0.28 | -44.31% | 873 |
| | EMA Crossover | 228,063.85 | 128.06% | 7.95% | 0.18 | -37.14% | 92 |
| | RSI Strategy | 171,059.42 | 71.06% | 5.11% | 0.01 | -38.53% | 60 |
| **INFY** | Buy and Hold | 440,266.15 | 340.27% | 14.75% | 0.43 | -36.55% | 1 |
| | Random | 20,337.51 | -79.66% | -13.74% | -0.99 | -80.20% | 919 |
| | EMA Crossover | 240,215.33 | 140.22% | 8.47% | 0.21 | -39.22% | 83 |
| | RSI Strategy | 189,950.66 | 89.95% | 6.14% | 0.09 | -41.57% | 60 |

---

## 5. Key Research Insights
1. **Passive outperformance**: Passively holding the stock (`Buy_and_Hold`) is the strongest baseline on all four stocks. This is because the Indian equity market was in a strong secular bull market over 2015-2025. 
2. **Transaction Friction is Deadly**: Random trading results in massive capital erosion (e.g. -79.66% for INFY) due to high transaction counts (~900 trades). Each trade loses 0.1% in fees and 0.05% in slippage, compounding to drain the account.
3. **EMA Crossover**: Provides moderate trend capture (especially on Reliance with 219.85% return) and reduces the Maximum Drawdown (e.g., Reliance Max DD dropped from -45.09% to -32.66%). However, it suffers from whipsaws (false signals) in choppy/rangebound markets, leading to over-trading and fee accumulation.
4. **RSI Mean Reversion**: RSI provides lower Max Drawdowns than Buy & Hold (e.g., TCS Max DD dropped from -34.45% to -29.57%), but underperforms in CAGR since it exits positions early when a trend continues to run.
