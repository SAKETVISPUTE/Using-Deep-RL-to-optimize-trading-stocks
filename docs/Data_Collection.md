# Data Collection and Validation

This document describes the design, implementation, and verification of the historical financial data collection pipeline for the Reinforcement Learning trading framework.

---

## 1. Objective
The objective is to collect daily historical financial data for the NIFTY 50 Index and a representative set of its constituent stocks from Yahoo Finance. This data serves as the historical price database from which features are engineered and environments are initialized.

---

## 2. Design Decisions & Rationale

### Why NIFTY 50?
The NIFTY 50 index represents the weighted average of 50 of the largest Indian companies listed on the National Stock Exchange (NSE). It is the benchmark index for the Indian equity market. Using NIFTY 50 and its major constituents allows us to study RL trading strategies in a highly liquid, large-cap emerging market.

### Tickers Selected
For the initial development and evaluation phases, we downloaded:
1. **`^NSEI`**: The NIFTY 50 Index (serving as the market benchmark).
2. **`RELIANCE.NS`**: Reliance Industries Limited (Energy/Conglomerate, highest index weight).
3. **`TCS.NS`**: Tata Consultancy Services Limited (Information Technology sector leader).
4. **`HDFCBANK.NS`**: HDFC Bank Limited (Financial Services sector leader).
5. **`INFY.NS`**: Infosys Limited (Information Technology sector, major constituent).

This subset provides cross-sector representation (Conglomerates, Financials, IT) which is crucial to test the agent's ability to generalise across different sector dynamics.

### Date Range
* **Start Date**: `2015-01-01`
* **End Date**: `2025-12-31`
* **Length**: 11 years (covers multiple market cycles, including the 2020 COVID-19 crash, subsequent recovery, and different interest rate regimes).

---

## 3. Financial & Technical Intuition

### Adjusted Prices vs. Nominal Prices
Nominal closing prices can experience artificial jumps (discontinuities) due to corporate actions like stock splits, mergers, and dividend payouts. For example, a 2-for-1 stock split halves the stock price overnight, which would look like a 50% loss to a naive RL agent, leading to incorrect policy updates.
To prevent this, we download adjusted prices using the `auto_adjust=True` parameter in `yfinance`. This backwards-adjusts the historical Open, High, Low, and Close prices so that price transitions represent pure economic returns.

### Parquet Format
While CSV is human-readable, we save raw data in both CSV and Parquet formats. Parquet is a columnar storage format that provides:
1. Significant disk space savings via compression.
2. Faster read/write times during training loops.
3. Strict schema preservation (e.g., maintaining `datetime` index types).

---

## 4. Implementation Details

The downloading pipeline is implemented in [data_downloader.py](file:///mnt/c/Users/Saket/Desktop/Projects/Finsearch_RL/src/utils/data_downloader.py).

### Data Downloader Class
The [DataDownloader](file:///mnt/c/Users/Saket/Desktop/Projects/Finsearch_RL/src/utils/data_downloader.py#L6) handles:
1. **Fetching**: Downloading raw data using `yfinance`.
2. **Post-Processing**: Newer versions of `yfinance` return columns formatted as a `MultiIndex` (with Price and Ticker levels). The downloader flattens this index to a single level index `['Close', 'High', 'Low', 'Open', 'Volume']` for compatibility and simplicity.
3. **Validation**: Before saving, the downloader verifies:
   - The data is not empty.
   - There are no negative prices.
   - High price is always greater than or equal to Low price.
   - Missing/null values are reported.

---

## 5. Verification Results

Running the downloader script completed successfully:
* **NIFTY 50 Index (`^NSEI`)**: Shape `(2707, 5)` (2707 trading days).
* **Constituent Stocks (`RELIANCE.NS`, `TCS.NS`, `HDFCBANK.NS`, `INFY.NS`)**: Shape `(2715, 5)` (2715 trading days).
* **Storage Location**: [data/raw/](file:///mnt/c/Users/Saket/Desktop/Projects/Finsearch_RL/data/raw)

All downloaded data passed negative-price and high/low validation checks.

---

## 6. Limitations and Future Improvements
* **Slippage and Intraday Data**: Daily close data assumes we trade at the close price without intraday slippage or execution details. For high-frequency trading, intraday bars (e.g., 5-minute or 15-minute) would be needed, but daily data is appropriate for portfolio-level swing trading strategies.
* **Survival Bias**: We selected current constituents. If we want to simulate backtests over 20 years, we must account for historical constituents that have been delisted to avoid survival bias.
