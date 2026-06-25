# Data Collection and Validation

This document describes the historical financial data collection pipeline and validation checks for the RL trading framework.

The downloader is implemented in [data_downloader.py](file:///mnt/c/Users/Saket/Desktop/Projects/Finsearch_RL/src/utils/data_downloader.py).

---

## 1. Data Source and Tickers
We fetch daily historical market data from **Yahoo Finance**. The tickers selected represent the major large-cap sectors of the Indian equity market:

1. **`^NSEI`**: NIFTY 50 Index (Market Benchmark)
2. **`RELIANCE.NS`**: Reliance Industries Limited (Conglomerate/Energy, highest index weight)
3. **`TCS.NS`**: Tata Consultancy Services Limited (IT sector leader)
4. **`HDFCBANK.NS`**: HDFC Bank Limited (Financial services leader)
5. **`INFY.NS`**: Infosys Limited (IT sector, major constituent)

### Date Range
- **Start Date**: `2015-01-01`
- **End Date**: `2025-12-31`
- **Duration**: 11 years (approx. 2,715 trading days).

---

## 2. Rationale for Adjusted Prices
Stock splits, mergers, and dividends cause large artificial jumps in nominal price charts. To ensure that price changes reflect actual economic returns, we download adjusted prices using the `auto_adjust=True` parameter in `yfinance`. This backwards-adjusts the historical Open, High, Low, and Close prices.

---

## 3. Data Validation Checks
Before raw files are stored, they pass the following checks:
1. **Empty Check**: Downloader verifies that the returned DataFrame is not empty.
2. **Missing Values**: Identifies and logs any null price values.
3. **Negative Price Check**: Verifies that Open, High, Low, and Close prices are all strictly positive ($> 0.0$).
4. **Price Relation Check**: Verifies that the High price is greater than or equal to the Low price for all rows: $High_t \ge Low_t$.

---

## 4. Verification and Storage
All downloaded files passed the validation checks and are saved in the raw data directory:
- **Location**: [data/raw/](file:///mnt/c/Users/Saket/Desktop/Projects/Finsearch_RL/data/raw)
- **Formats**: Saved as both **CSV** (for human readability) and **Parquet** (for high-performance disk storage and compression).
- **Dataset Shapes**:
  - `nsei`: 2,707 trading days
  - `reliance_ns`, `tcs_ns`, `hdfcbank_ns`, `infy_ns`: 2,715 trading days
