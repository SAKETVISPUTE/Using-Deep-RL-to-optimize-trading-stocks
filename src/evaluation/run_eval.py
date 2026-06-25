import os
import sys
import argparse
import pandas as pd
from stable_baselines3 import PPO

# Add src to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.environment.trading_env import TradingEnv
from src.evaluation.backtester import Backtester

def evaluate():
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO RL agent on a specified stock ticker.")
    parser.add_argument("--stock", type=str, default="reliance_ns", help="Stock name prefix (e.g. 'reliance_ns').")
    parser.add_argument("--reward", type=str, default="portfolio_return", help="Reward function type used in training.")
    parser.add_argument("--feature-group", type=str, default="state_full", help="Feature group: 'state_0', 'state_1', 'state_full'.")
    
    args = parser.parse_args()
    
    model_path = f"models/ppo_{args.stock}_{args.reward}_{args.feature_group}.zip"
    processed_path = f"data/processed/{args.stock}_processed.parquet"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model {model_path} not found. Train the model first.")
    if not os.path.exists(processed_path):
        raise FileNotFoundError(f"Processed stock data {processed_path} not found.")
        
    print(f"\n=================== Evaluating Model for {args.stock.upper()} ===================")
    print(f"Model: {model_path}")
    print(f"Data: {processed_path}")
    print(f"Feature Group: {args.feature_group}")
    
    # Load model
    model = PPO.load(model_path)
    
    # Load data
    df = pd.read_parquet(processed_path)
    
    # Create evaluation environment
    env = TradingEnv(
        df=df,
        initial_cash=100000.0,
        transaction_fee=0.001,
        slippage=0.0005,
        reward_type=args.reward,
        feature_group=args.feature_group
    )
    
    # Run backtester
    backtester = Backtester(env, model)
    history_df = backtester.run_backtest()
    
    # Compute metrics
    metrics = backtester.get_metrics()
    
    print("\n------------------- Evaluation Metrics -------------------")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k:30} : {v:.4f}")
        else:
            print(f"{k:30} : {v}")
            
    # Save comparison plot
    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)
    plot_path = os.path.join(out_dir, f"{args.stock}_backtest_{args.feature_group}.png")
    backtester.plot_results(benchmark_name="Buy & Hold", save_path=plot_path)

if __name__ == "__main__":
    evaluate()
