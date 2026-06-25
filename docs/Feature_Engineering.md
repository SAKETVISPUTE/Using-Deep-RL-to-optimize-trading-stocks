# Feature Engineering

This document outlines the design, mathematical formulation, financial intuition, and reinforcement learning implications of the feature engineering pipeline.

The implementation is located in [feature_engineer.py](file:///mnt/c/Users/Saket/Desktop/Projects/Finsearch_RL/src/features/feature_engineer.py) and executed via [process_all.py](file:///mnt/c/Users/Saket/Desktop/Projects/Finsearch_RL/src/features/process_all.py).

---

## 1. Objective
Raw stock prices (Open, High, Low, Close, Volume) do not directly reveal the underlying market state, momentum, or volatility to a reinforcement learning agent. The objective of this module is to extract informative, stationarized, and normalized financial features to provide the agent with a rich representation of the market.

---

## 2. Engineered Features: Mathematical & Financial Intuition

We engineer features across five key categories:

### A. Price & Return Features
1. **Daily Simple Return (`Daily_Return`)**
   - **Formula**: $R_t = \frac{Close_t - Close_{t-1}}{Close_{t-1}}$
   - **Financial Intuition**: The percentage change in value over one day.
   - **RL Intuition**: Raw price is non-stationary (tends to drift upward over time), which makes learning unstable. Returns are stationary (mean-reverting around a constant mean), allowing the agent to generalize across different price scales.
2. **Log Return (`Log_Return`)**
   - **Formula**: $r_t = \ln\left(\frac{Close_t}{Close_{t-1}}\right)$
   - **Financial Intuition**: Log returns are time-additive: the log return over multiple periods is the sum of log returns of individual periods.
   - **RL Intuition**: Used as a clean, symmetric measure of rate of change.

### B. Trend Features
Trend indicators help the agent identify if the asset is in a sustained upward or downward movement.
1. **Exponential Moving Averages (`EMA_10`, `EMA_30`)**
   - **Formula**: $EMA_t = Close_t \cdot \alpha + EMA_{t-1} \cdot (1 - \alpha)$, where $\alpha = \frac{2}{N+1}$.
   - **Financial Intuition**: EMAs track the average price over a rolling window ($N=10$ and $N=30$ days), giving more weight to recent prices. Crossovers (e.g., short-term EMA crossing above long-term EMA) signify changes in trend direction.
   - **RL Intuition**: Helps the agent recognize the medium-to-long term price trajectory.
2. **Moving Average Convergence Divergence (`MACD`, `MACD_Signal`, `MACD_Hist`)**
   - **Formula**: 
     - $MACD = EMA_{12}(Close) - EMA_{26}(Close)$
     - $Signal = EMA_{9}(MACD)$
     - $Histogram = MACD - Signal$
   - **Financial Intuition**: MACD measures the relationship between two moving averages. When the MACD line crosses above the Signal line, it indicates bullish momentum; when it crosses below, it indicates bearish momentum.
   - **RL Intuition**: The histogram provides a smooth, bounded oscillator that represents trend strength and potential reversals.

### C. Momentum Features
Momentum measures the velocity of price changes to determine if a trend is accelerating or decelerating.
1. **Relative Strength Index (`RSI`)**
   - **Formula**: $RSI = 100 - \frac{100}{1 + RS}$, where $RS = \frac{\text{Average Gain over 14 days}}{\text{Average Loss over 14 days}}$.
   - **Financial Intuition**: Bounded between 0 and 100. Traditionally, values $>70$ indicate "overbought" conditions (potential pullback) and values $<30$ indicate "oversold" conditions (potential bounce).
   - **RL Intuition**: Since the state space must be bounded for stable neural network training, RSI is an excellent feature because it is naturally constrained to $[0, 100]$.
2. **Rate of Change (`ROC`)**
   - **Formula**: $ROC = \frac{Close_t - Close_{t-N}}{Close_{t-N}} \times 100$, for $N=10$.
   - **Financial Intuition**: Measures the percentage change between the current price and the price $N$ periods ago.

### D. Volatility Features
Volatility represents the magnitude of price movements, which correlates directly with risk.
1. **Average True Range (`ATR`)**
   - **Formula**: $ATR_t = \frac{1}{14} \sum_{i=0}^{13} TR_{t-i}$, where $TR = \max(High - Low, |High - Close_{t-1}|, |Low - Close_{t-1}|)$.
   - **Financial Intuition**: Measures the average daily trading range, accounting for overnight gaps.
   - **RL Intuition**: High ATR suggests high volatility. The agent can use this to lower position sizes (risk-management) or avoid trading during turbulent market regimes.
2. **Bollinger Bands (`BB_Upper`, `BB_Middle`, `BB_Lower`)**
   - **Formula**: 
     - $Middle = SMA_{20}(Close)$
     - $Std = \sigma_{20}(Close)$
     - $Upper = Middle + 2 \cdot Std$
     - $Lower = Middle - 2 \cdot Std$
   - **Financial Intuition**: Prices tend to stay within the bands. Touching the upper band suggests overextended prices, while touching the lower band suggests underextended prices.

### E. Volume Features
1. **Volume Change (`Volume_Change`)**
   - **Formula**: $\frac{Volume_t - Volume_{t-1}}{Volume_{t-1}}$
   - **Financial Intuition**: Identifies spikes in trading activity. High volume changes accompanying price moves suggest strong institutional participation and trend confirmation.

### F. Market Benchmark Features
1. **Market Return (`Market_Return`)**
   - **Formula**: $R_{t, \text{NSEI}} = \frac{Index_t - Index_{t-1}}{Index_{t-1}}$
   - **Financial Intuition**: Captures the daily return of the broader NIFTY 50 market index.
   - **RL Intuition**: Helps the agent contextualize individual stock movements. For example, a stock drop during a market-wide sell-off is interpreted differently than a stock drop when the broader market is rising.

---

## 3. Avoidance of Lookahead Bias
A common pitfall in financial ML is **lookahead bias**—using future information to predict the present. We strictly prevent this by:
1. Only using rolling windows that include historical data up to step $t$ (e.g., `shift(1)`, `.rolling()`, `.pct_change()`).
2. Aligning benchmark index mapping using datetime indexes so that market returns are mapped exactly day-for-day without leakages.

---

## 4. Verification and Outputs
We run [process_all.py](file:///mnt/c/Users/Saket/Desktop/Projects/Finsearch_RL/src/features/process_all.py) to process all raw files.
- **Output directory**: [data/processed/](file:///mnt/c/Users/Saket/Desktop/Projects/Finsearch_RL/data/processed)
- **Features generated**: 21 columns (5 raw + 16 engineered).
- **Quality check**: Checked for and verified that **0 NaN values** remain. Missing data from early rolling windows (warmup phase) is filled using backfill/forward-fill to prevent network NaN propagation.

---

## 5. Limitations & Future Directions
- **Feature Scaling**: Currently, features like RSI are in range $[0, 100]$, prices are in the hundreds/thousands, and returns are in decimals. Before feeding them to the neural network policy, we must normalize/standardize the state vectors. This will be handled in the Environment/Agent state preprocessing step.
- **Sentiment/Alternative Data**: We do not currently include news sentiment or order book depth, which could provide additional alpha.
