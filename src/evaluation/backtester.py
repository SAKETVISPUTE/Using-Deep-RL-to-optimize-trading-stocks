import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add src to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.environment.trading_env import TradingEnv
from src.baselines.baseline_strategies import calculate_metrics

class Backtester:
    """
    Evaluates a trained RL model deterministically on a TradingEnv,
    calculates professional metrics, and handles plotting.
    """
    def __init__(self, env: TradingEnv, model):
        self.env = env
        self.model = model
        self.history_df = None

    def run_backtest(self) -> pd.DataFrame:
        """
        Runs the deterministic policy rollout on the environment.
        """
        obs, info = self.env.reset()
        done = False
        
        while not done:
            # Predict action deterministically (no exploration noise)
            action, _states = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = self.env.step(int(action))
            done = terminated or truncated
            
        self.history_df = self.env.get_history_df()
        return self.history_df

    def get_metrics(self, risk_free_rate=0.06) -> dict:
        """
        Calculates all performance metrics for the backtested policy.
        """
        if self.history_df is None:
            raise ValueError("No backtest run found. Call run_backtest() first.")
            
        metrics = calculate_metrics(self.history_df, initial_cash=self.env.initial_cash, risk_free_rate=risk_free_rate)
        
        # Calculate Win Rate specifically for closed trades
        # Walk through history and detect trade rounds (Buy -> Sell or Sell -> Buy)
        history = self.history_df.to_dict('records')
        trades = []
        buy_price = None
        
        for i, step in enumerate(history):
            action = step['action']
            price = step['price']
            
            if action == 1: # Buy (Entry)
                if buy_price is None:
                    buy_price = price
            elif action == 2: # Sell (Exit)
                if buy_price is not None:
                    trade_return = (price - buy_price) / buy_price
                    trades.append(trade_return)
                    buy_price = None
                    
        win_rate = 0.0
        if len(trades) > 0:
            wins = sum(1 for t in trades if t > 0)
            win_rate = wins / len(trades)
            
        metrics["Win Rate (%)"] = float(win_rate * 100)
        metrics["Number of Trades (Closed)"] = len(trades)
        
        return metrics

    def plot_results(self, benchmark_name="Buy & Hold", save_path=None):
        """
        Generates and saves professional performance plots including:
        - Portfolio value curve vs Buy & Hold
        - Trade execution markers
        - Drawdowns
        """
        if self.history_df is None:
            raise ValueError("No backtest run found. Call run_backtest() first.")

        df = self.history_df.copy()
        df.set_index('date', inplace=True)

        # 1. Calculate Buy & Hold benchmark portfolio value for comparison
        # Buy on day 1 and hold. Portfolio value = initial_cash * (price_t / price_0)
        initial_price = df.iloc[0]['price']
        initial_cash = self.env.initial_cash
        df['benchmark_val'] = initial_cash * (df['price'] / initial_price)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

        # Plot Portfolio Value
        ax1.plot(df.index, df['portfolio_value'], label='RL Agent (PPO)', color='blue', linewidth=2)
        ax1.plot(df.index, df['benchmark_val'], label=f'Benchmark ({benchmark_name})', color='gray', linestyle='--', linewidth=1.5)
        
        # Add Trade markers
        # Buy markers (action = 1)
        buys = df[df['action'] == 1]
        ax1.scatter(buys.index, buys['portfolio_value'], label='Buy Signal', color='green', marker='^', s=80, zorder=5)
        
        # Sell markers (action = 2)
        sells = df[df['action'] == 2]
        ax1.scatter(sells.index, sells['portfolio_value'], label='Sell Signal', color='red', marker='v', s=80, zorder=5)

        ax1.set_title("Strategy Performance Comparison & Trade Logs", fontsize=14, fontweight='bold')
        ax1.set_ylabel("Portfolio Value (₹)", fontsize=12)
        ax1.legend(loc='upper left', fontsize=10)
        ax1.grid(True, linestyle=':', alpha=0.6)

        # Plot Drawdowns
        # RL Agent Drawdown
        peaks_rl = df['portfolio_value'].cummax()
        dd_rl = (df['portfolio_value'] - peaks_rl) / peaks_rl
        
        # Benchmark Drawdown
        peaks_bench = df['benchmark_val'].cummax()
        dd_bench = (df['benchmark_val'] - peaks_bench) / peaks_bench

        ax2.fill_between(df.index, dd_rl * 100, 0, label='RL Drawdown', color='blue', alpha=0.3)
        ax2.fill_between(df.index, dd_bench * 100, 0, label='Benchmark Drawdown', color='gray', alpha=0.1)
        
        ax2.set_title("Underwater Plot (Drawdowns)", fontsize=12, fontweight='bold')
        ax2.set_ylabel("Drawdown (%)", fontsize=12)
        ax2.set_xlabel("Date", fontsize=12)
        ax2.legend(loc='lower left', fontsize=10)
        ax2.grid(True, linestyle=':', alpha=0.6)

        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"Saved evaluation plot to {save_path}")
        else:
            plt.show()
            
        plt.close()
