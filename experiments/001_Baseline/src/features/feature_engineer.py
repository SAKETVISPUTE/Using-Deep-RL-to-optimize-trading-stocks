import os
import pandas as pd
import numpy as np

class FeatureEngineer:
    """
    Statically engineers technical indicators and market features from raw OHLCV price data.
    Ensures mathematical correctness and prevents lookahead bias.
    """
    def __init__(self, include_nsei_benchmark: bool = True):
        self.include_nsei_benchmark = include_nsei_benchmark

    def compute_features(self, df: pd.DataFrame, benchmark_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Computes price, trend, momentum, volatility, and volume features.
        
        Args:
            df (pd.DataFrame): Dataframe with Close, High, Low, Open, Volume columns.
            benchmark_df (pd.DataFrame, optional): Dataframe of index benchmark (e.g., ^NSEI) to engineer market features.
            
        Returns:
            pd.DataFrame: Cleaned dataframe with original OHLCV and engineered features.
        """
        # Ensure we work on a copy to prevent SettingWithCopyWarning
        df = df.copy()
        
        # Ensure index is datetime and sorted
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        # 1. Price & Return Features
        df['Daily_Return'] = df['Close'].pct_change()
        df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # 2. Trend Features
        # Exponential Moving Averages (EMA)
        df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
        df['EMA_30'] = df['Close'].ewm(span=30, adjust=False).mean()
        # MACD: 12-day EMA - 26-day EMA
        ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema_12 - ema_26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

        # 3. Momentum Features
        # RSI (Relative Strength Index) - 14 Days
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-9)
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # ROC (Rate of Change) - 10 Days
        df['ROC'] = ((df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)) * 100

        # 4. Volatility Features
        # ATR (Average True Range) - 14 Days
        high_low = df['High'] - df['Low']
        high_close_prev = np.abs(df['High'] - df['Close'].shift(1))
        low_close_prev = np.abs(df['Low'] - df['Close'].shift(1))
        tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(window=14).mean()
        
        # Bollinger Bands - 20 Days
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        df['BB_Std'] = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (2 * df['BB_Std'])
        df['BB_Lower'] = df['BB_Middle'] - (2 * df['BB_Std'])

        # 5. Volume Features
        # Volume change
        df['Volume_Change'] = df['Volume'].pct_change()
        
        # 6. Market/Benchmark Features
        if self.include_nsei_benchmark and benchmark_df is not None:
            benchmark_df = benchmark_df.copy()
            benchmark_df.index = pd.to_datetime(benchmark_df.index)
            benchmark_df = benchmark_df.sort_index()
            
            # Compute benchmark returns
            benchmark_returns = benchmark_df['Close'].pct_change()
            
            # Align indices by mapping benchmark returns
            df['Market_Return'] = df.index.map(benchmark_returns)
            # Fill any missing values in market returns (e.g. index trading days differ slightly from stock)
            df['Market_Return'] = df['Market_Return'].ffill().fillna(0.0)

        # 7. Convert absolute price metrics into dimensionless/stationary ratios relative to Close price
        df['EMA_10'] = (df['Close'] - df['EMA_10']) / df['Close']
        df['EMA_30'] = (df['Close'] - df['EMA_30']) / df['Close']
        
        df['MACD'] = df['MACD'] / df['Close']
        df['MACD_Signal'] = df['MACD_Signal'] / df['Close']
        df['MACD_Hist'] = df['MACD_Hist'] / df['Close']
        
        df['RSI'] = df['RSI'] / 100.0
        df['ROC'] = df['ROC'] / 100.0
        df['ATR'] = df['ATR'] / df['Close']
        
        df['BB_Upper'] = (df['BB_Upper'] - df['Close']) / df['Close']
        df['BB_Lower'] = (df['Close'] - df['BB_Lower']) / df['Close']
        df['BB_Middle'] = (df['Close'] - df['BB_Middle']) / df['Close']
        df['BB_Std'] = df['BB_Std'] / df['Close']
        
        df['Volume_Change'] = np.clip(df['Volume_Change'], -3.0, 3.0)

        # 8. Clean up NaN/Inf values resulting from rolling windows/lags/divisions
        # Replace any infs (e.g. division by zero) with 0.0
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.ffill().fillna(0.0)
        
        return df


if __name__ == "__main__":
    # Self-test script to verify feature engineering works on raw data
    print("Testing FeatureEngineer...")
    raw_dir = "data/raw"
    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)
    
    reliance_path = os.path.join(raw_dir, "reliance_ns.parquet")
    nsei_path = os.path.join(raw_dir, "nsei.parquet")
    
    if os.path.exists(reliance_path) and os.path.exists(nsei_path):
        df_rel = pd.read_parquet(reliance_path)
        df_nsei = pd.read_parquet(nsei_path)
        
        fe = FeatureEngineer(include_nsei_benchmark=True)
        processed_df = fe.compute_features(df_rel, df_nsei)
        
        print("\nProcessed DataFrame sample:")
        print(processed_df.head(20))
        print(f"\nOriginal shape: {df_rel.shape}")
        print(f"Processed shape: {processed_df.shape}")
        print("Columns engineered:")
        print(list(processed_df.columns))
        
        # Verify no NaNs left
        nan_count = processed_df.isnull().sum().sum()
        print(f"Total NaN values in processed DataFrame: {nan_count}")
        
        # Save processed data
        out_path_csv = os.path.join(processed_dir, "reliance_ns_processed.csv")
        out_path_parquet = os.path.join(processed_dir, "reliance_ns_processed.parquet")
        processed_df.to_csv(out_path_csv)
        processed_df.to_parquet(out_path_parquet)
        print(f"Saved processed data to {out_path_csv} and {out_path_parquet}")
    else:
        print("Raw data files not found for self-test. Run data_downloader.py first.")
