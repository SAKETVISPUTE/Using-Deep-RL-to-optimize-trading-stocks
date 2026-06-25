import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class TradingEnv(gym.Env):
    """
    A realistic, Gymnasium-compatible trading environment for a single asset.
    Simulates portfolio mechanics, transaction fees, slippage, and tracks performance.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        initial_cash: float = 100000.0,
        transaction_fee: float = 0.001,      # 0.1% transaction cost (brokerage + taxes)
        slippage: float = 0.0005,             # 0.05% price slippage
        reward_type: str = "portfolio_return", # "portfolio_return", "log_return", "sharpe_ratio"
        trade_volume_fraction: float = 1.0,    # Fraction of cash/position to trade (1.0 = full buy/sell)
    ):
        super(TradingEnv, self).__init__()

        self.df = df.copy().sort_index()
        self.initial_cash = initial_cash
        self.transaction_fee = transaction_fee
        self.slippage = slippage
        self.reward_type = reward_type
        self.trade_volume_fraction = trade_volume_fraction

        # Drop non-feature columns if they exist (e.g. index/date is already the index)
        # Features will be all columns except OHLCV (or we keep OHLCV + technicals, but we separate them for price calculations)
        self.feature_cols = [col for col in self.df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
        self.num_features = len(self.feature_cols)

        # Action Space: 0 = Hold, 1 = Buy, 2 = Sell
        self.action_space = spaces.Discrete(3)

        # Observation Space:
        # - Technical/Market features (shape: self.num_features)
        # - Portfolio features (shape: 3):
        #   1. Normalized Cash (cash / current_portfolio_value)
        #   2. Normalized Position (holdings * price / current_portfolio_value)
        #   3. Normalized Cumulative Return ((portfolio_value - initial_cash) / initial_cash)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_features + 3,),
            dtype=np.float32
        )

        # Episode state variables
        self.current_step = 0
        self.total_steps = len(self.df) - 1
        
        self.cash = self.initial_cash
        self.holdings = 0.0
        self.portfolio_value = self.initial_cash
        self.initial_price = 0.0
        
        # History tracking for analysis/plotting
        self.history = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_step = 0
        self.cash = self.initial_cash
        self.holdings = 0.0
        self.portfolio_value = self.initial_cash
        self.initial_price = self.df.iloc[0]['Close']
        self.history = []

        # Record initial state
        self._record_history(action=0, reward=0.0)

        observation = self._get_observation()
        info = self.get_info()

        return observation, info

    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action: {action}"

        # Current day's price data
        current_row = self.df.iloc[self.current_step]
        current_price = current_row['Close']

        # Execute trades at the current price
        # Buy: we buy at a slightly higher price due to slippage
        # Sell: we sell at a slightly lower price due to slippage
        executed_action = 0  # 0 = Hold, 1 = Buy, 2 = Sell
        prev_portfolio_value = self.portfolio_value

        if action == 1:  # Buy
            buy_price = current_price * (1.0 + self.slippage)
            # Calculate how much cash we can allocate
            cash_to_spend = self.cash * self.trade_volume_fraction
            if cash_to_spend > 0:
                # Deduct transaction fees
                total_cost_per_share = buy_price * (1.0 + self.transaction_fee)
                shares_bought = cash_to_spend / total_cost_per_share
                
                self.holdings += shares_bought
                self.cash -= cash_to_spend
                executed_action = 1
                
        elif action == 2:  # Sell
            sell_price = current_price * (1.0 - self.slippage)
            shares_to_sell = self.holdings * self.trade_volume_fraction
            if shares_to_sell > 0:
                # Add cash after deducting fees
                cash_received = shares_to_sell * sell_price * (1.0 - self.transaction_fee)
                
                self.holdings -= shares_to_sell
                self.cash += cash_received
                executed_action = 2

        # Move to next step (next day)
        self.current_step += 1
        
        # Calculate new portfolio value based on next day's close price
        next_row = self.df.iloc[self.current_step]
        next_price = next_row['Close']
        self.portfolio_value = self.cash + (self.holdings * next_price)

        # Calculate reward
        reward = self._calculate_reward(prev_portfolio_value)

        # Check termination (reached end of data)
        terminated = self.current_step >= self.total_steps
        truncated = False

        # Record state details
        self._record_history(action=executed_action, reward=reward)

        observation = self._get_observation()
        info = self.get_info()

        return observation, reward, terminated, truncated, info

    def _get_observation(self):
        # Extract features for the current step
        features = self.df.iloc[self.current_step][self.feature_cols].values.astype(np.float32)

        # Normalize portfolio state elements to prevent scale issues in Neural Networks
        norm_cash = np.array([self.cash / self.portfolio_value], dtype=np.float32)
        norm_position = np.array([(self.holdings * self.df.iloc[self.current_step]['Close']) / self.portfolio_value], dtype=np.float32)
        norm_return = np.array([(self.portfolio_value - self.initial_cash) / self.initial_cash], dtype=np.float32)

        # Concatenate features and portfolio state
        obs = np.concatenate([features, norm_cash, norm_position, norm_return])
        return obs

    def _calculate_reward(self, prev_portfolio_value):
        # 1. Simple Portfolio Return: % change in portfolio value
        if self.reward_type == "portfolio_return":
            reward = (self.portfolio_value - prev_portfolio_value) / prev_portfolio_value
            
        # 2. Log Return of Portfolio Value
        elif self.reward_type == "log_return":
            reward = np.log(self.portfolio_value / prev_portfolio_value)
            
        # 3. Sharpe-inspired (Daily return minus risk-free rate penalty or volatility penalty)
        elif self.reward_type == "risk_adjusted":
            daily_return = (self.portfolio_value - prev_portfolio_value) / prev_portfolio_value
            # Penalty for holding a volatile asset: subtract rolling standard deviation (if available) or static penalty
            vol_penalty = 0.0
            if 'BB_Std' in self.df.columns:
                current_std = self.df.iloc[self.current_step]['BB_Std']
                current_price = self.df.iloc[self.current_step]['Close']
                norm_std = current_std / current_price  # Volatility relative to price
                vol_penalty = 0.05 * norm_std * (self.holdings * current_price / self.portfolio_value)
            reward = daily_return - vol_penalty
            
        else:
            raise ValueError(f"Unknown reward type: {self.reward_type}")
            
        return float(reward)

    def get_info(self):
        return {
            "step": self.current_step,
            "cash": self.cash,
            "holdings": self.holdings,
            "portfolio_value": self.portfolio_value,
            "cumulative_return": (self.portfolio_value - self.initial_cash) / self.initial_cash,
        }

    def _record_history(self, action, reward):
        current_row = self.df.iloc[self.current_step]
        self.history.append({
            "date": self.df.index[self.current_step],
            "price": current_row['Close'],
            "cash": self.cash,
            "holdings": self.holdings,
            "portfolio_value": self.portfolio_value,
            "action": action,
            "reward": reward,
        })

    def render(self, mode="human"):
        if len(self.history) > 0:
            last = self.history[-1]
            print(
                f"Date: {last['date'].strftime('%Y-%m-%d')} | "
                f"Price: {last['price']:.2f} | "
                f"Action: {last['action']} | "
                f"Cash: {last['cash']:.2f} | "
                f"Holdings: {last['holdings']:.4f} | "
                f"Portfolio Value: {last['portfolio_value']:.2f} | "
                f"Cum Return: {((last['portfolio_value'] - self.initial_cash)/self.initial_cash)*100:.2f}%"
            )

    def get_history_df(self) -> pd.DataFrame:
        """
        Returns the episode history as a pandas DataFrame for analysis/plotting.
        """
        return pd.DataFrame(self.history)
