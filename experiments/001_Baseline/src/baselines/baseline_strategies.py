import os
import sys
import numpy as np
import pandas as pd

# Add src to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

class BaseStrategy:
    """
    Abstract base class for heuristic trading strategies.
    """
    def __init__(self):
        pass

    def get_action(self, obs, env) -> int:
        """
        Args:
            obs: Observation vector from environment
            env: The TradingEnv instance (to access history/parameters)
        Returns:
            int: Action (0 = Hold, 1 = Buy, 2 = Sell)
        """
        raise NotImplementedError("Strategies must implement get_action.")

class BuyAndHoldStrategy(BaseStrategy):
    """
    Buys immediately on step 0 and holds until termination.
    """
    def get_action(self, obs, env) -> int:
        # If we have no position, Buy (1). Otherwise, Hold (0).
        if env.holdings == 0.0 and env.cash > 0.0:
            return 1
        return 0

class RandomStrategy(BaseStrategy):
    """
    Selects a random action uniformly.
    """
    def get_action(self, obs, env) -> int:
        return int(np.random.randint(0, 3))

class EMACrossoverStrategy(BaseStrategy):
    """
    EMA Crossover strategy:
    - Buy (1) when short-term EMA (EMA_10) crosses above long-term EMA (EMA_30).
    - Sell (2) when short-term EMA crosses below long-term EMA.
    - Hold (0) otherwise.
    """
    def __init__(self, ema_fast_col='EMA_10', ema_slow_col='EMA_30'):
        super().__init__()
        self.ema_fast_col = ema_fast_col
        self.ema_slow_col = ema_slow_col

    def get_action(self, obs, env) -> int:
        # We need historical steps to detect crossover.
        # env.current_step gives current index in the dataframe.
        if env.current_step < 2:
            return 0
            
        prev_row = env.df.iloc[env.current_step - 1]
        curr_row = env.df.iloc[env.current_step]
        
        prev_fast = prev_row[self.ema_fast_col]
        prev_slow = prev_row[self.ema_slow_col]
        curr_fast = curr_row[self.ema_fast_col]
        curr_slow = curr_row[self.ema_slow_col]
        
        # Bullish crossover: Fast EMA crosses above Slow EMA
        if prev_fast <= prev_slow and curr_fast > curr_slow:
            # Only buy if we do not already hold shares
            if env.holdings == 0:
                return 1
                
        # Bearish crossover: Fast EMA crosses below Slow EMA
        elif prev_fast >= prev_slow and curr_fast < curr_slow:
            # Only sell if we currently hold shares
            if env.holdings > 0:
                return 2
                
        return 0

class RSIStrategy(BaseStrategy):
    """
    RSI-based mean reversion strategy:
    - Buy (1) when RSI falls below oversold threshold (default 30).
    - Sell (2) when RSI rises above overbought threshold (default 70).
    """
    def __init__(self, rsi_col='RSI', oversold=30.0, overbought=70.0):
        super().__init__()
        self.rsi_col = rsi_col
        self.oversold = oversold
        self.overbought = overbought

    def get_action(self, obs, env) -> int:
        curr_row = env.df.iloc[env.current_step]
        rsi = curr_row[self.rsi_col]
        
        if rsi < self.oversold:
            if env.holdings == 0:
                return 1
        elif rsi > self.overbought:
            if env.holdings > 0:
                return 2
        return 0

def calculate_metrics(history_df, initial_cash=100000.0, risk_free_rate=0.06) -> dict:
    """
    Computes professional quantitative performance metrics from portfolio history.
    """
    dates = pd.to_datetime(history_df['date'])
    portfolio_values = history_df['portfolio_value'].values
    actions = history_df['action'].values
    
    # Calculate returns
    daily_returns = history_df['portfolio_value'].pct_change().dropna().values
    
    # Cumulative return
    final_val = portfolio_values[-1]
    cumulative_return = (final_val - initial_cash) / initial_cash
    
    # Annualized Return (assuming 252 trading days per year)
    num_days = len(portfolio_values)
    years = num_days / 252.0
    annualized_return = (1.0 + cumulative_return) ** (1.0 / years) - 1.0
    
    # Annualized Volatility
    annualized_vol = np.std(daily_returns) * np.sqrt(252) if len(daily_returns) > 0 else 0.0
    
    # Sharpe Ratio (daily risk-free rate adjustment)
    daily_rf = risk_free_rate / 252.0
    excess_returns = daily_returns - daily_rf
    sharpe_ratio = (np.mean(excess_returns) / np.std(daily_returns)) * np.sqrt(252) if len(daily_returns) > 0 and np.std(daily_returns) > 0 else 0.0
    
    # Sortino Ratio (downside risk only)
    downside_returns = daily_returns[daily_returns < daily_rf] - daily_rf
    downside_deviation = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0.0
    sortino_ratio = (np.mean(excess_returns) * np.sqrt(252)) / downside_deviation if downside_deviation > 0 else 0.0
    
    # Maximum Drawdown
    peaks = np.maximum.accumulate(portfolio_values)
    drawdowns = (portfolio_values - peaks) / peaks
    max_drawdown = np.min(drawdowns)
    
    # Calmar Ratio
    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown < 0 else 0.0
    
    # Number of trades executed (action = 1 or 2, excluding step 0 if it automatically records action 0)
    num_trades = np.sum((actions == 1) | (actions == 2))
    
    return {
        "Final Value": float(final_val),
        "Cumulative Return (%)": float(cumulative_return * 100),
        "Annualized Return (%)": float(annualized_return * 100),
        "Annualized Volatility (%)": float(annualized_vol * 100),
        "Sharpe Ratio": float(sharpe_ratio),
        "Sortino Ratio": float(sortino_ratio),
        "Max Drawdown (%)": float(max_drawdown * 100),
        "Calmar Ratio": float(calmar_ratio),
        "Number of Trades": int(num_trades)
    }

def run_evaluation(env, strategy) -> pd.DataFrame:
    """
    Runs a single strategy on the environment and returns the history DataFrame.
    """
    obs, info = env.reset()
    done = False
    
    while not done:
        action = strategy.get_action(obs, env)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
    return env.get_history_df()
