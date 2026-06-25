import os
import sys
import argparse
import pandas as pd
import numpy as np
import optuna
from stable_baselines3 import DQN

# Add src to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.environment.trading_env import TradingEnv
from src.evaluation.backtester import Backtester

def objective(trial):
    # 1. Load Data
    stock = "reliance_ns"
    processed_path = f"data/processed/{stock}_processed.parquet"
    df = pd.read_parquet(processed_path)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    
    # Train / Validation Split
    train_df = df.loc["2015-01-01":"2020-12-31"]
    val_df = df.loc["2021-01-01":"2024-12-31"]
    
    # 2. Instantiate Environments
    train_env = TradingEnv(
        df=train_df,
        initial_cash=100000.0,
        transaction_fee=0.001,
        slippage=0.0005,
        reward_type="portfolio_return",
        feature_group="state_2",
        action_space_type="discrete_3",
        history_len=1
    )
    
    val_env = TradingEnv(
        df=val_df,
        initial_cash=100000.0,
        transaction_fee=0.001,
        slippage=0.0005,
        reward_type="portfolio_return",
        feature_group="state_2",
        action_space_type="discrete_3",
        history_len=1
    )
    
    # 3. Suggest Hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    buffer_size = trial.suggest_categorical("buffer_size", [5000, 10000, 20000, 50000])
    learning_starts = trial.suggest_categorical("learning_starts", [100, 500, 1000])
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    gamma = trial.suggest_float("gamma", 0.90, 0.999)
    exploration_fraction = trial.suggest_float("exploration_fraction", 0.1, 0.4)
    exploration_final_eps = trial.suggest_float("exploration_final_eps", 0.01, 0.1)
    target_update_interval = trial.suggest_categorical("target_update_interval", [100, 500, 1000, 5000])
    
    # 4. Train DQN Agent
    model = DQN(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        learning_starts=learning_starts,
        batch_size=batch_size,
        gamma=gamma,
        exploration_fraction=exploration_fraction,
        exploration_final_eps=exploration_final_eps,
        target_update_interval=target_update_interval,
        verbose=0
    )
    
    model.learn(total_timesteps=50000)
    
    # 5. Evaluate on Validation Set
    backtester = Backtester(val_env, model)
    history_df = backtester.run_backtest()
    metrics = backtester.get_metrics()
    
    sharpe = metrics.get("Sharpe Ratio", -10.0)
    cum_return = metrics.get("Cumulative Return (%)", -100.0)
    
    # If agent performed no trades, penalize it slightly to encourage active strategies
    if metrics.get("Number of Trades", 0) == 0:
        return -5.0
        
    # We optimize for Sharpe Ratio primarily
    # Handle NaN/Inf
    if np.isnan(sharpe) or np.isinf(sharpe):
        return -10.0
        
    return float(sharpe)

def run_tuning():
    parser = argparse.ArgumentParser(description="Run Optuna hyperparameter tuning for DQN.")
    parser.add_argument("--trials", type=int, default=30, help="Number of trials.")
    parser.add_argument("--study-name", type=str, default="dqn_study", help="Optuna study name.")
    parser.add_argument("--storage", type=str, default=None, help="Optuna storage URL (e.g. sqlite:///optuna.db).")
    args = parser.parse_args()
    
    print("\n=================== Starting DQN Hyperparameter Tuning ===================")
    print(f"Number of Trials: {args.trials}")
    print(f"Study Name: {args.study_name}")
    print(f"Storage: {args.storage}")
    
    # Set up optuna logging
    optuna.logging.set_verbosity(optuna.logging.INFO)
    
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        direction="maximize"
    )
    study.optimize(objective, n_trials=args.trials)
    
    print("\n=================== Hyperparameter Tuning Results ===================")
    try:
        print(f"Best Trial value (Sharpe Ratio): {study.best_value:.4f}")
        print("Best Hyperparameters:")
        for k, v in study.best_params.items():
            print(f"  {k:30} : {v}")
            
        # Save best parameters to a YAML file
        os.makedirs("configs", exist_ok=True)
        best_params_path = f"configs/dqn_best_params.yaml"
        import yaml
        with open(best_params_path, "w") as f:
            yaml.dump({"dqn": study.best_params}, f)
        print(f"\nSuccessfully saved best hyperparameters to {best_params_path}")
    except ValueError:
        print("No trials completed yet in this process.")

if __name__ == "__main__":
    run_tuning()
