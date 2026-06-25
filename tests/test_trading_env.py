import os
import sys
import pandas as pd
import numpy as np

# Ensure src is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.environment.trading_env import TradingEnv

def test_environment_gym_interface():
    print("Testing Gymnasium interface compliance...")
    
    # Load processed data
    processed_file = "data/processed/reliance_ns_processed.parquet"
    if not os.path.exists(processed_file):
        print(f"Error: {processed_file} does not exist. Run process_all.py first.")
        return False
        
    df = pd.read_parquet(processed_file)
    
    # Initialize environment
    env = TradingEnv(
        df=df,
        initial_cash=100000.0,
        transaction_fee=0.001,
        slippage=0.0005,
        reward_type="portfolio_return"
    )
    
    # Check Spaces
    print(f"Action Space: {env.action_space}")
    print(f"Observation Space: {env.observation_space}")
    assert env.action_space.n == 3
    assert env.observation_space.shape[0] == (env.num_features + env.num_portfolio_features) * env.history_len
    
    # Check Reset
    obs, info = env.reset()
    print(f"Reset returned observation shape: {obs.shape}")
    print(f"Reset returned info: {info}")
    assert obs.shape[0] == (env.num_features + env.num_portfolio_features) * env.history_len
    assert isinstance(info, dict)
    assert info["portfolio_value"] == 100000.0
    
    # Check Step with action = Hold (0)
    obs, reward, terminated, truncated, info = env.step(0)
    print(f"Step with Hold: Reward: {reward:.6f}, Terminated: {terminated}, Info: {info}")
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    
    # Check Step with action = Buy (1)
    obs, reward, terminated, truncated, info = env.step(1)
    print(f"Step with Buy: Cash: {info['cash']:.2f}, Holdings: {info['holdings']:.4f}, Portfolio Value: {info['portfolio_value']:.2f}")
    assert info["holdings"] > 0
    assert info["cash"] < 100000.0
    
    # Check Step with action = Sell (2)
    obs, reward, terminated, truncated, info = env.step(2)
    print(f"Step with Sell: Cash: {info['cash']:.2f}, Holdings: {info['holdings']:.4f}, Portfolio Value: {info['portfolio_value']:.2f}")
    assert np.isclose(info["holdings"], 0.0) # all holdings liquidated with volume fraction 1.0
    
    # Run complete episode with random agent
    print("\nRunning a complete episode with a random agent...")
    obs, info = env.reset()
    done = False
    step_count = 0
    
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step_count += 1
        
        if step_count % 500 == 0:
            env.render()
            
    print(f"Episode finished in {step_count} steps.")
    print(f"Final portfolio value: {env.portfolio_value:.2f}")
    print(f"Cumulative Return: {info['cumulative_return']*100:.2f}%")
    assert step_count == env.total_steps
    print("All tests passed successfully!")
    return True

if __name__ == "__main__":
    test_environment_gym_interface()
