import os
import pandas as pd
from feature_engineer import FeatureEngineer

def process_all_data():
    raw_dir = "data/raw"
    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)
    
    nsei_path = os.path.join(raw_dir, "nsei.parquet")
    if not os.path.exists(nsei_path):
        raise FileNotFoundError(f"Benchmark file {nsei_path} is missing. Download raw data first.")
        
    df_nsei = pd.read_parquet(nsei_path)
    
    # Initialize FeatureEngineer
    fe = FeatureEngineer(include_nsei_benchmark=True)
    
    # List of stock tickers to process
    stocks = ["reliance_ns", "tcs_ns", "hdfcbank_ns", "infy_ns"]
    
    for stock in stocks:
        raw_path = os.path.join(raw_dir, f"{stock}.parquet")
        if not os.path.exists(raw_path):
            print(f"Skipping {stock} because raw data is missing.")
            continue
            
        print(f"Processing features for {stock}...")
        df_stock = pd.read_parquet(raw_path)
        
        # Compute features
        processed_df = fe.compute_features(df_stock, df_nsei)
        
        # Save output
        out_csv = os.path.join(processed_dir, f"{stock}_processed.csv")
        out_parquet = os.path.join(processed_dir, f"{stock}_processed.parquet")
        
        processed_df.to_csv(out_csv)
        processed_df.to_parquet(out_parquet)
        print(f"Saved {stock} processed data (shape: {processed_df.shape}) to {out_csv} and {out_parquet}")

    # Also process the benchmark itself (without benchmark returns mapping to self or with inclusion disabled)
    print("Processing features for benchmark index (NSEI)...")
    fe_bench = FeatureEngineer(include_nsei_benchmark=False)
    processed_nsei = fe_bench.compute_features(df_nsei)
    
    # Manually set Market_Return to Daily_Return for the benchmark index itself to prevent missing column
    processed_nsei['Market_Return'] = processed_nsei['Daily_Return']
    
    out_csv = os.path.join(processed_dir, "nsei_processed.csv")
    out_parquet = os.path.join(processed_dir, "nsei_processed.parquet")
    processed_nsei.to_csv(out_csv)
    processed_nsei.to_parquet(out_parquet)
    print(f"Saved NSEI processed data (shape: {processed_nsei.shape}) to {out_csv} and {out_parquet}")

if __name__ == "__main__":
    process_all_data()
