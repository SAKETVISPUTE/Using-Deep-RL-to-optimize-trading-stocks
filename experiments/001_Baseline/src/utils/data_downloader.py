import os
import argparse
import pandas as pd
import yfinance as yf

class DataDownloader:
    """
    Downloader class to fetch historical stock and index data from Yahoo Finance.
    Ensures data consistency, saves raw files, and provides basic validation.
    """
    def __init__(self, output_dir="data/raw"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def fetch_data(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetches historical daily data for a given ticker from Yahoo Finance.
        """
        print(f"Fetching data for {ticker} from {start_date} to {end_date}...")
        # For quantitative research, we prefer adjusted prices to prevent artificial jumps from splits/dividends.
        df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            df.columns.name = None
        return df


    def validate_data(self, df: pd.DataFrame, ticker: str) -> bool:
        """
        Validates the downloaded dataframe for common data quality issues.
        """
        if df.empty:
            print(f"WARNING: Downloader returned empty DataFrame for {ticker}.")
            return False
        
        # Check for missing values
        null_counts = df.isnull().sum().sum()
        if null_counts > 0:
            print(f"WARNING: {ticker} data contains {null_counts} missing values.")
        
        # Check for negative prices
        for col in ['Open', 'High', 'Low', 'Close']:
            if col in df.columns:
                neg_count = (df[col] < 0).sum()
                if neg_count > 0:
                    print(f"ERROR: {ticker} contains negative values in column '{col}'.")
                    return False
        
        # Check High >= Low
        if all(c in df.columns for c in ['Open', 'High', 'Low', 'Close']):
            bad_high_low = (df['High'] < df['Low']).sum()
            if bad_high_low > 0:
                print(f"WARNING: {ticker} has {bad_high_low} rows where High < Low.")
                
        print(f"Validation passed for {ticker}. Shape: {df.shape}")
        return True

    def save_data(self, df: pd.DataFrame, ticker: str) -> str:
        """
        Saves the dataframe to CSV and Parquet format in the raw data directory.
        """
        # Clean ticker name for file safety (e.g. ^NSEI -> nsei, RELIANCE.NS -> reliance_ns)
        clean_name = ticker.replace("^", "").replace(".", "_").lower()
        
        csv_path = os.path.join(self.output_dir, f"{clean_name}.csv")
        parquet_path = os.path.join(self.output_dir, f"{clean_name}.parquet")
        
        # Save to CSV
        df.to_csv(csv_path)
        # Save to Parquet
        df.to_parquet(parquet_path)
        
        print(f"Saved {ticker} to {csv_path} and {parquet_path}")
        return parquet_path

def main():
    parser = argparse.ArgumentParser(description="Download financial historical data for NIFTY 50 and constituents.")
    parser.add_argument("--tickers", nargs="+", default=["^NSEI"], help="List of tickers to download. Default is '^NSEI' (NIFTY 50 Index).")
    parser.add_argument("--start", type=str, default="2015-01-01", help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end", type=str, default="2025-12-31", help="End date (YYYY-MM-DD).")
    parser.add_argument("--out-dir", type=str, default="data/raw", help="Output directory for raw data.")
    
    args = parser.parse_args()
    
    downloader = DataDownloader(output_dir=args.out_dir)
    
    for ticker in args.tickers:
        try:
            df = downloader.fetch_data(ticker, args.start, args.end)
            if downloader.validate_data(df, ticker):
                downloader.save_data(df, ticker)
        except Exception as e:
            print(f"Failed to download or process {ticker}: {e}")

if __name__ == "__main__":
    main()
