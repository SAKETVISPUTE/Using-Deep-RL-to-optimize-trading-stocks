import os
import sys
import pandas as pd
import numpy as np

# Add src to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.environment.trading_env import TradingEnv
from src.baselines.baseline_strategies import (
    BuyAndHoldStrategy,
    RandomStrategy,
    EMACrossoverStrategy,
    RSIStrategy,
    calculate_metrics,
    run_evaluation
)

def run_all_baselines():
    processed_dir = "data/processed"
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    stocks = ["reliance_ns", "tcs_ns", "hdfcbank_ns", "infy_ns"]
    strategies = {
        "Buy_and_Hold": BuyAndHoldStrategy(),
        "Random": RandomStrategy(),
        "EMA_Crossover": EMACrossoverStrategy(),
        "RSI_Strategy": RSIStrategy()
    }
    
    results = []
    
    for stock in stocks:
        file_path = os.path.join(processed_dir, f"{stock}_processed.parquet")
        if not os.path.exists(file_path):
            print(f"Skipping {stock} because processed data is missing.")
            continue
            
        df = pd.read_parquet(file_path)
        print(f"\n=================== Evaluating baselines for {stock.upper()} ===================")
        
        for strat_name, strategy in strategies.items():
            # Initialize identical environment for each baseline
            env = TradingEnv(
                df=df,
                initial_cash=100000.0,
                transaction_fee=0.001,
                slippage=0.0005,
                reward_type="portfolio_return"
            )
            
            # Run simulation
            history_df = run_evaluation(env, strategy)
            
            # Calculate metrics
            metrics = calculate_metrics(history_df, initial_cash=100000.0)
            metrics["Stock"] = stock
            metrics["Strategy"] = strat_name
            results.append(metrics)
            
            print(f"Strategy: {strat_name:15} | Final Value: {metrics['Final Value']:.2f} | Sharpe: {metrics['Sharpe Ratio']:.2f} | MaxDD: {metrics['Max Drawdown (%)']:.2f}% | Trades: {metrics['Number of Trades']}")

    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Reorder columns to put Stock and Strategy first
    cols = ["Stock", "Strategy", "Final Value", "Cumulative Return (%)", "Annualized Return (%)", 
            "Annualized Volatility (%)", "Sharpe Ratio", "Sortino Ratio", "Max Drawdown (%)", "Calmar Ratio", "Number of Trades"]
    results_df = results_df[cols]
    
    # Save results
    csv_out = os.path.join(results_dir, "baselines_comparison.csv")
    results_df.to_csv(csv_out, index=False)
    print(f"\nSaved all baseline evaluation metrics to {csv_out}")

if __name__ == "__main__":
    run_all_baselines()
