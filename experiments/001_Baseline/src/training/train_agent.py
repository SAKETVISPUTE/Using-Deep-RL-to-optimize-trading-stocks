import os
import sys
import argparse
import yaml
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

# Add src to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.environment.trading_env import TradingEnv

class TradingMetricsCallback(BaseCallback):
    """
    Custom callback to log trading environment metrics (like portfolio value and return)
    to TensorBoard during training.
    """
    def __init__(self, verbose=0):
        super(TradingMetricsCallback, self).__init__(verbose)
        self.episode_count = 0

    def _on_step(self) -> bool:
        # Check if environment sent an episode info (usually available at termination)
        # In SB3, env is wrapped in a VecEnv, so we access it via self.training_env.env_method or locals
        for env_info in self.locals.get("infos", []):
            if "episode" in env_info.keys():
                # Episode finished
                self.episode_count += 1
                
        return True

    def _on_rollout_end(self) -> None:
        # Log environment state at the end of each rollout
        # self.training_env is VecEnv. We can get the portfolio values from the environments
        portfolio_values = self.training_env.env_method("render")
        # Since render prints, we can also extract cumulative return
        cum_returns = self.training_env.env_method("get_info")
        if len(cum_returns) > 0:
            info = cum_returns[0]
            self.logger.record("trading/portfolio_value", info["portfolio_value"])
            self.logger.record("trading/cumulative_return", info["cumulative_return"])
            self.logger.record("trading/cash", info["cash"])
            self.logger.record("trading/holdings", info["holdings"])

def train():
    parser = argparse.ArgumentParser(description="Train a PPO RL agent on a specified stock ticker.")
    parser.add_argument("--stock", type=str, default="reliance_ns", help="Stock name prefix (e.g. 'reliance_ns').")
    parser.add_argument("--timesteps", type=int, default=0, help="Override total training steps. Default uses YAML config.")
    parser.add_argument("--reward", type=str, default="portfolio_return", help="Reward function type: 'portfolio_return', 'log_return', 'risk_adjusted'.")
    parser.add_argument("--config", type=str, default="configs/ppo_config.yaml", help="Path to PPO hyperparameters config file.")
    
    args = parser.parse_args()
    
    # Paths
    processed_path = f"data/processed/{args.stock}_processed.parquet"
    if not os.path.exists(processed_path):
        raise FileNotFoundError(f"Processed data file {processed_path} not found. Run process_all.py first.")
        
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)['ppo']
        
    # Override timesteps if provided
    total_timesteps = args.timesteps if args.timesteps > 0 else config['total_timesteps']
    
    print(f"\n=================== Starting PPO Training for {args.stock.upper()} ===================")
    print(f"Reward Type: {args.reward}")
    print(f"Total Timesteps: {total_timesteps}")
    print(f"Hyperparameters: {config}")
    
    # Load data
    df = pd.read_parquet(processed_path)
    
    # Create training environment
    env = TradingEnv(
        df=df,
        initial_cash=100000.0,
        transaction_fee=0.001,
        slippage=0.0005,
        reward_type=args.reward
    )
    
    # Create directories for models and logs
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs/tensorboard", exist_ok=True)
    
    tensorboard_log = "logs/tensorboard/"
    tb_name = f"ppo_{args.stock}_{args.reward}"
    
    # Instantiate PPO Agent
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=config['learning_rate'],
        n_steps=config['n_steps'],
        batch_size=config['batch_size'],
        n_epochs=config['n_epochs'],
        gamma=config['gamma'],
        gae_lambda=config['gae_lambda'],
        clip_range=config['clip_range'],
        ent_coef=config['ent_coef'],
        vf_coef=config['vf_coef'],
        max_grad_norm=config['max_grad_norm'],
        policy_kwargs=config['policy_kwargs'],
        verbose=1,
        tensorboard_log=tensorboard_log
    )
    
    # Callbacks
    metrics_callback = TradingMetricsCallback()
    
    # Train agent
    model.learn(
        total_timesteps=total_timesteps,
        tb_log_name=tb_name,
        callback=metrics_callback
    )
    
    # Save Model
    model_save_path = f"models/ppo_{args.stock}_{args.reward}.zip"
    model.save(model_save_path)
    print(f"Successfully trained and saved model to {model_save_path}")

if __name__ == "__main__":
    train()
