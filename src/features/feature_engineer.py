import os
import pandas as pd
import numpy as np

class FeatureEngineer:
    """
    Computes all Version 2 technical, portfolio, and market features.
    Provides options to filter features into progressive state groups.
    """
    def __init__(self, include_nsei_benchmark: bool = True):
        self.include_nsei_benchmark = include_nsei_benchmark

    def compute_features(self, df: pd.DataFrame, benchmark_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Computes all planned V2 features and ensures they are stationary and dimensionless.
        """
        df = df.copy()
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        # ----------------------------------------------------
        # STATE 0: Raw OHLCV (preserved as columns for trade calculations)
        # ----------------------------------------------------
        
        # ----------------------------------------------------
        # STATE 1: Price Dynamics (Stationary & Dimensionless)
        # ----------------------------------------------------
        df['Daily_Return'] = df['Close'].pct_change()
        df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Gap_Return'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
        df['Intraday_Return'] = (df['Close'] - df['Open']) / df['Open']
        df['High_Low_Range'] = (df['High'] - df['Low']) / df['Close']
        df['Rolling_Mean_Return_10'] = df['Daily_Return'].rolling(window=10).mean()
        df['Rolling_Std_Return_10'] = df['Daily_Return'].rolling(window=10).std()

        # ----------------------------------------------------
        # STATE 2: Trend Indicators (Scaled relative to Close)
        # ----------------------------------------------------
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
        
        # Trend indicators scaled relative to current Close
        df['SMA_5_Ratio'] = (df['Close'] - df['SMA_5']) / df['Close']
        df['SMA_10_Ratio'] = (df['Close'] - df['SMA_10']) / df['Close']
        df['SMA_20_Ratio'] = (df['Close'] - df['SMA_20']) / df['Close']
        df['SMA_50_Ratio'] = (df['Close'] - df['SMA_50']) / df['Close']
        
        df['EMA_10_Ratio'] = (df['Close'] - df['EMA_10']) / df['Close']
        df['EMA_20_Ratio'] = (df['Close'] - df['EMA_20']) / df['Close']
        df['EMA_50_Ratio'] = (df['Close'] - df['EMA_50']) / df['Close']
        
        # MACD
        ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
        macd = ema_12 - ema_26
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        
        df['MACD_Ratio'] = macd / df['Close']
        df['MACD_Signal_Ratio'] = macd_signal / df['Close']
        df['MACD_Hist_Ratio'] = (macd - macd_signal) / df['Close']

        # ----------------------------------------------------
        # STATE 3: Momentum Indicators
        # ----------------------------------------------------
        # RSI - 14 Days
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-9)
        df['RSI_Scaled'] = (100 - (100 / (1 + rs))) / 100.0
        
        # ROC - 10 Days
        df['ROC_Ratio'] = ((df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10))
        
        # Momentum - 10 Days
        df['Momentum_Ratio'] = (df['Close'] - df['Close'].shift(10)) / df['Close']

        # Stochastic Oscillator (%K, %D)
        low_14 = df['Low'].rolling(window=14).min()
        high_14 = df['High'].rolling(window=14).max()
        stoch_k = ((df['Close'] - low_14) / (high_14 - low_14 + 1e-9))
        df['Stochastic_K'] = stoch_k
        df['Stochastic_D'] = stoch_k.rolling(window=3).mean()

        # Williams %R
        df['Williams_R'] = ((high_14 - df['Close']) / (high_14 - low_14 + 1e-9)) * -1.0

        # CCI (Commodity Channel Index) - 20 Days
        tp = (df['High'] + df['Low'] + df['Close']) / 3.0
        sma_tp = tp.rolling(window=20).mean()
        mad_tp = tp.rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
        df['CCI_Scaled'] = ((tp - sma_tp) / (0.015 * mad_tp + 1e-9)) / 100.0

        # ----------------------------------------------------
        # STATE 4: Volatility Indicators
        # ----------------------------------------------------
        # ATR - 14 Days
        high_low = df['High'] - df['Low']
        high_close_prev = np.abs(df['High'] - df['Close'].shift(1))
        low_close_prev = np.abs(df['Low'] - df['Close'].shift(1))
        tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        df['ATR_Ratio'] = tr.rolling(window=14).mean() / df['Close']
        
        # Bollinger Bands - 20 Days
        bb_middle = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        bb_upper = bb_middle + (2 * bb_std)
        bb_lower = bb_middle - (2 * bb_std)
        
        df['BB_Middle_Ratio'] = (df['Close'] - bb_middle) / df['Close']
        df['BB_Upper_Ratio'] = (bb_upper - df['Close']) / df['Close']
        df['BB_Lower_Ratio'] = (df['Close'] - bb_lower) / df['Close']
        df['BB_Width'] = (bb_upper - bb_lower) / bb_middle
        
        # Historical Volatility (rolling 20-day annualized std of returns)
        df['Hist_Volatility'] = df['Daily_Return'].rolling(window=20).std() * np.sqrt(252)
        df['Rolling_Variance_20'] = df['Daily_Return'].rolling(window=20).var()

        # ----------------------------------------------------
        # STATE 5: Volume Indicators
        # ----------------------------------------------------
        # Volume change
        df['Volume_Change'] = np.clip(df['Volume'].pct_change(), -3.0, 3.0)
        
        # On-Balance Volume (OBV)
        obv = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        df['OBV_Ratio'] = obv / (df['Volume'].rolling(window=20).mean() * 100.0 + 1e-9) # Scaled relative to rolling volume
        
        # Volume Moving Average
        vol_ma = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / (vol_ma + 1e-9)
        
        # Chaikin Money Flow (CMF) - 20 Days
        mf_multiplier = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'] + 1e-9)
        mf_volume = mf_multiplier * df['Volume']
        df['CMF'] = mf_volume.rolling(window=20).sum() / (df['Volume'].rolling(window=20).sum() + 1e-9)

        # Accumulation/Distribution Line (ADL)
        adl = mf_volume.fillna(0).cumsum()
        df['ADL_Ratio'] = adl / (df['Volume'].rolling(window=20).mean() * 100.0 + 1e-9)

        # ----------------------------------------------------
        # STATE 7: Market Context (Mapped from Benchmark)
        # ----------------------------------------------------
        if self.include_nsei_benchmark and benchmark_df is not None:
            benchmark_df = benchmark_df.copy()
            benchmark_df.index = pd.to_datetime(benchmark_df.index)
            benchmark_df = benchmark_df.sort_index()
            
            bench_returns = benchmark_df['Close'].pct_change()
            bench_vol = bench_returns.rolling(window=20).std() * np.sqrt(252)
            
            # Compute rolling market trend (simple SMA crossover on index)
            bench_sma_10 = benchmark_df['Close'].rolling(window=10).mean()
            bench_sma_30 = benchmark_df['Close'].rolling(window=30).mean()
            bench_trend = (bench_sma_10 - bench_sma_30) / bench_sma_30

            # Align indices
            df['Market_Return'] = df.index.map(bench_returns)
            df['Market_Volatility'] = df.index.map(bench_vol)
            df['Market_Trend'] = df.index.map(bench_trend)
            
            # Stock relative strength: 20-day return diff
            stock_perf = df['Close'].pct_change(periods=20)
            bench_perf = benchmark_df['Close'].pct_change(periods=20)
            df['Relative_Strength'] = stock_perf - df.index.map(bench_perf)
            
            # Fill missing benchmark metrics
            for col in ['Market_Return', 'Market_Volatility', 'Market_Trend', 'Relative_Strength']:
                df[col] = df[col].ffill().fillna(0.0)

        # Clean up any remaining NaN/Inf values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.ffill().fillna(0.0)
        
        # Drop temporary unscaled columns so they aren't fed as features
        temp_cols = ['SMA_5', 'SMA_10', 'SMA_20', 'SMA_50', 'EMA_10', 'EMA_20', 'EMA_50']
        df = df.drop(columns=[col for col in temp_cols if col in df.columns])

        return df

if __name__ == "__main__":
    print("Self-testing V2 FeatureEngineer...")
    raw_dir = "data/raw"
    reliance_path = os.path.join(raw_dir, "reliance_ns.parquet")
    nsei_path = os.path.join(raw_dir, "nsei.parquet")
    
    if os.path.exists(reliance_path) and os.path.exists(nsei_path):
        df_rel = pd.read_parquet(reliance_path)
        df_nsei = pd.read_parquet(nsei_path)
        fe = FeatureEngineer(include_nsei_benchmark=True)
        processed = fe.compute_features(df_rel, df_nsei)
        print(f"Features computed. Shape: {processed.shape}")
        print("Engineered columns:", list(processed.columns))
        print("Any NaNs?", processed.isnull().sum().sum())
    else:
        print("Raw data not found. Run data_downloader.py first.")
