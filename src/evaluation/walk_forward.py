import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, A2C, DQN

# Add src to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.environment.trading_env import TradingEnv
from src.evaluation.backtester import Backtester
from src.baselines.baseline_strategies import calculate_metrics

def run_walk_forward_validation():
    parser = argparse.ArgumentParser(description="Run walk-forward validation for a stock ticker.")
    parser.add_argument("--stock", type=str, default="reliance_ns", help="Stock name prefix.")
    parser.add_argument("--timesteps", type=int, default=50000, help="Training timesteps per window.")
    parser.add_argument("--feature-group", type=str, default="state_full", help="Feature group: 'state_0', 'state_1', 'state_full'.")
    parser.add_argument("--reward-type", type=str, default="portfolio_return", help="Reward function type.")
    parser.add_argument("--action-space-type", type=str, default="discrete_3", help="Action space: 'discrete_3', 'discrete_7'.")
    parser.add_argument("--history-len", type=int, default=1, help="Temporal frame stacking/history window length.")
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "a2c", "dqn"], help="RL algorithm: 'ppo', 'a2c', 'dqn'.")
    parser.add_argument("--config", type=str, default=None, help="Path to hyperparameters config file.")
    
    args = parser.parse_args()
    
    processed_path = f"data/processed/{args.stock}_processed.parquet"
    if not os.path.exists(processed_path):
        raise FileNotFoundError(f"Processed data file {processed_path} not found. Run process_all.py first.")
        
    df_full = pd.read_parquet(processed_path)
    df_full.index = pd.to_datetime(df_full.index)
    df_full = df_full.sort_index()

    # Define Walk-Forward Windows
    # Format: (Train Start, Train End, Test Start, Test End)
    windows = [
        ("2015-01-01", "2020-12-31", "2021-01-01", "2021-12-31"),
        ("2016-01-01", "2021-12-31", "2022-01-01", "2022-12-31"),
        ("2017-01-01", "2022-12-31", "2023-01-01", "2023-12-31"),
        ("2018-01-01", "2023-12-31", "2024-01-01", "2024-12-31")
    ]
    
    out_of_sample_histories = []
    
    # We will track portfolio values sequentially.
    # The ending portfolio value of one test year becomes the initial cash for the next test year.
    current_cash = 100000.0
    
    print(f"\n=================== Starting Walk-Forward Validation for {args.stock.upper()} ===================")
    print(f"Feature Group: {args.feature_group}")
    
    for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
        print(f"\n--- Window {i+1}: Train [{train_start} to {train_end}] | Test [{test_start} to {test_end}] ---")
        
        # 1. Filter Train / Test Data
        train_df = df_full.loc[train_start:train_end]
        test_df = df_full.loc[test_start:test_end]
        
        if len(train_df) == 0 or len(test_df) == 0:
            print(f"Skipping window due to empty data (Train len: {len(train_df)}, Test len: {len(test_df)}).")
            continue
            
        # 2. Train PPO model on Train data
        train_env = TradingEnv(
            df=train_df,
            initial_cash=100000.0,
            transaction_fee=0.001,
            slippage=0.0005,
            reward_type=args.reward_type,
            feature_group=args.feature_group,
            action_space_type=args.action_space_type,
            history_len=args.history_len
        )
        
        # Load hyperparams from config if provided
        config_params = {}
        if args.config:
            import yaml
            with open(args.config, 'r') as f:
                loaded_config = yaml.safe_load(f)
                if loaded_config and args.algo in loaded_config:
                    config_params = loaded_config[args.algo]

        print(f"Training {args.algo.upper()} agent for {args.timesteps} steps...")
        if args.algo == "ppo":
            ppo_kwargs = {
                "policy": "MlpPolicy",
                "env": train_env,
                "learning_rate": 0.0003,
                "n_steps": 2048,
                "batch_size": 64,
                "n_epochs": 10,
                "gamma": 0.99,
                "verbose": 0
            }
            ppo_kwargs.update(config_params)
            model = PPO(**ppo_kwargs)
        elif args.algo == "a2c":
            a2c_kwargs = {
                "policy": "MlpPolicy",
                "env": train_env,
                "learning_rate": 0.0007,
                "n_steps": 5,
                "gamma": 0.99,
                "verbose": 0
            }
            a2c_kwargs.update(config_params)
            model = A2C(**a2c_kwargs)
        elif args.algo == "dqn":
            dqn_kwargs = {
                "policy": "MlpPolicy",
                "env": train_env,
                "learning_rate": 0.0001,
                "buffer_size": 10000,
                "learning_starts": 100,
                "batch_size": 32,
                "gamma": 0.99,
                "verbose": 0
            }
            dqn_kwargs.update(config_params)
            model = DQN(**dqn_kwargs)
        model.learn(total_timesteps=args.timesteps)
        
        # 3. Backtest on Test data (out-of-sample)
        # Using current_cash accumulated from previous years to simulate a continuous portfolio
        test_env = TradingEnv(
            df=test_df,
            initial_cash=current_cash,
            transaction_fee=0.001,
            slippage=0.0005,
            reward_type=args.reward_type,
            feature_group=args.feature_group,
            action_space_type=args.action_space_type,
            history_len=args.history_len
        )
        
        backtester = Backtester(test_env, model)
        history_df = backtester.run_backtest()
        
        print(f"Test Year Finished. Final Portfolio Value: {test_env.portfolio_value:.2f}")
        
        # Save history for this window
        out_of_sample_histories.append(history_df)
        
        # Update current cash for the next window
        current_cash = test_env.portfolio_value

    # 4. Concatenate and Process Out-of-Sample Histories
    if not out_of_sample_histories:
        print("No out-of-sample histories recorded. Exiting.")
        return

    # Combined test history
    combined_history = pd.concat(out_of_sample_histories, ignore_index=True)
    combined_history['date'] = pd.to_datetime(combined_history['date'])
    combined_history.sort_values('date', inplace=True)
    
    # Re-calculate index benchmark for the out-of-sample period (2021-01-01 to 2024-12-31)
    # The benchmark starts at the same initial cash (100k)
    benchmark_history = df_full.loc["2021-01-01":"2024-12-31"].copy()
    initial_bench_price = benchmark_history.iloc[0]['Close']
    combined_history['benchmark_val'] = 100000.0 * (combined_history['price'] / initial_bench_price)

    # Note: Adjust portfolio_value curve in combined_history to make it continuous starting from 100,000
    # Because the first year starts with 100k, and subsequent years carry over cash.
    # The portfolio value in each window is already continuous since we updated current_cash.
    
    # 5. Compute Combined Performance Metrics
    metrics = calculate_metrics(combined_history, initial_cash=100000.0)
    
    print("\n=================== Combined Out-of-Sample Metrics (2021-2024) ===================")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k:30} : {v:.4f}")
        else:
            print(f"{k:30} : {v}")

    # Save metrics to CSV
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    metrics_df = pd.DataFrame([metrics])
    metrics_df["Stock"] = args.stock
    metrics_df["Feature_Group"] = args.feature_group
    metrics_df["Reward_Type"] = args.reward_type
    metrics_df["Action_Space_Type"] = args.action_space_type
    metrics_df["History_Len"] = args.history_len
    metrics_df["Algorithm"] = args.algo
    
    filename_suffix = f"{args.feature_group}_{args.reward_type}_{args.action_space_type}_{args.algo}"
    if args.history_len > 1:
        filename_suffix += f"_h{args.history_len}"
        
    metrics_df.to_csv(os.path.join(results_dir, f"{args.stock}_walk_forward_metrics_{filename_suffix}.csv"), index=False)

    # 6. Plot Walk-Forward Performance Curve
    plt.figure(figsize=(14, 7))
    plt.plot(combined_history['date'], combined_history['portfolio_value'], label=f'RL Agent ({args.algo.upper()} - {args.feature_group} - {args.reward_type} - {args.action_space_type} - H{args.history_len})', color='blue', linewidth=2)
    plt.plot(combined_history['date'], combined_history['benchmark_val'], label='Buy & Hold Benchmark', color='gray', linestyle='--', linewidth=1.5)
    
    # Overlay Buy/Sell markers
    buys = combined_history[combined_history['action'] == 1]
    plt.scatter(buys['date'], buys['portfolio_value'], label='Buy Signal', color='green', marker='^', s=60, zorder=5)
    sells = combined_history[combined_history['action'] == 2]
    plt.scatter(sells['date'], sells['portfolio_value'], label='Sell Signal', color='red', marker='v', s=60, zorder=5)

    plt.title(f"Walk-Forward Backtest: {args.stock.upper()} ({args.algo.upper()} - {args.feature_group} - {args.reward_type} - {args.action_space_type} - H{args.history_len}) (2021-2024)", fontsize=14, fontweight='bold')
    plt.ylabel("Portfolio Value (₹)", fontsize=12)
    plt.xlabel("Date", fontsize=12)
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plot_path = os.path.join(results_dir, f"{args.stock}_walk_forward_{filename_suffix}.png")
    plt.savefig(plot_path, dpi=300)
    print(f"\nSaved combined walk-forward chart to {plot_path}")
    plt.close()

if __name__ == "__main__":
    run_walk_forward_validation()
