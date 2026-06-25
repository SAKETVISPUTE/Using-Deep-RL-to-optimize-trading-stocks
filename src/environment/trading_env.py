import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from collections import deque

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
        feature_group: str = "state_full",    # "state_0", "state_1", "state_full"
        action_space_type: str = "discrete_3", # "discrete_3", "discrete_7"
        history_len: int = 1,                 # Temporal frame stacking/history window length
    ):
        super(TradingEnv, self).__init__()

        self.df = df.copy().sort_index()
        self.initial_cash = initial_cash
        self.transaction_fee = transaction_fee
        self.slippage = slippage
        self.reward_type = reward_type
        self.trade_volume_fraction = trade_volume_fraction
        self.feature_group = feature_group
        self.action_space_type = action_space_type
        self.history_len = history_len
        self.obs_queue = deque(maxlen=self.history_len)

        # Filter feature columns based on group
        all_technicals = [col for col in self.df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
        
        state_1_cols = [
            'Daily_Return', 'Log_Return', 'Gap_Return', 'Intraday_Return',
            'High_Low_Range', 'Rolling_Mean_Return_10', 'Rolling_Std_Return_10'
        ]
        state_2_cols = state_1_cols + [
            'SMA_5_Ratio', 'SMA_10_Ratio', 'SMA_20_Ratio', 'SMA_50_Ratio',
            'EMA_10_Ratio', 'EMA_20_Ratio', 'EMA_50_Ratio',
            'MACD_Ratio', 'MACD_Signal_Ratio', 'MACD_Hist_Ratio'
        ]
        state_3_cols = state_2_cols + [
            'RSI_Scaled', 'ROC_Ratio', 'Momentum_Ratio',
            'Stochastic_K', 'Stochastic_D', 'Williams_R', 'CCI_Scaled'
        ]
        state_4_cols = state_3_cols + [
            'ATR_Ratio', 'BB_Middle_Ratio', 'BB_Upper_Ratio', 'BB_Lower_Ratio',
            'BB_Width', 'Hist_Volatility', 'Rolling_Variance_20'
        ]
        state_5_cols = state_4_cols + [
            'Volume_Change', 'OBV_Ratio', 'Volume_Ratio', 'CMF', 'ADL_Ratio'
        ]
        state_6_cols = state_5_cols
        state_7_cols = state_6_cols + [
            'Market_Return', 'Market_Volatility', 'Market_Trend', 'Relative_Strength'
        ]
        state_8_cols = state_7_cols

        if self.feature_group == "state_0":
            selected_cols = []
        elif self.feature_group == "state_1":
            selected_cols = state_1_cols
        elif self.feature_group == "state_2":
            selected_cols = state_2_cols
        elif self.feature_group == "state_3":
            selected_cols = state_3_cols
        elif self.feature_group == "state_4":
            selected_cols = state_4_cols
        elif self.feature_group == "state_5":
            selected_cols = state_5_cols
        elif self.feature_group == "state_6":
            selected_cols = state_6_cols
        elif self.feature_group == "state_7":
            selected_cols = state_7_cols
        elif self.feature_group == "state_8":
            selected_cols = state_8_cols
        elif self.feature_group == "state_full":
            selected_cols = all_technicals
        else:
            raise ValueError(f"Unknown feature_group: {self.feature_group}")
            
        self.feature_cols = [col for col in selected_cols if col in self.df.columns]
        self.num_features = len(self.feature_cols)

        # Action Space configuration
        if self.action_space_type == "discrete_3":
            self.action_space = spaces.Discrete(3)
        elif self.action_space_type == "discrete_7":
            self.action_space = spaces.Discrete(7)
        else:
            raise ValueError(f"Unknown action_space_type: {self.action_space_type}")

        # Observation Space:
        # - Technical/Market features (shape: self.num_features)
        # - Portfolio features:
        #   1. Normalized Cash (cash / current_portfolio_value)
        #   2. Normalized Position (holdings * price / current_portfolio_value)
        #   3. Normalized Cumulative Return ((portfolio_value - initial_cash) / initial_cash)
        #   For state_6, state_7, state_8, and state_full, we add:
        #   4. Average Buy Price Ratio: (Close - average_buy_price) / Close (0 if no holdings)
        #   5. Unrealized PnL Ratio: (holdings * Close - holdings * average_buy_price) / portfolio_value
        #   6. Time Since Last Trade: (current_step - last_trade_step) / 252.0 (annualized fraction)
        self.num_portfolio_features = 3
        if self.feature_group in ["state_6", "state_7", "state_8", "state_full"]:
            self.num_portfolio_features = 6

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=((self.num_features + self.num_portfolio_features) * self.history_len,),
            dtype=np.float32
        )

        # Episode state variables
        self.current_step = 0
        self.total_steps = len(self.df) - 1
        
        self.cash = self.initial_cash
        self.holdings = 0.0
        self.portfolio_value = self.initial_cash
        self.initial_price = 0.0
        self.average_buy_price = 0.0
        self.last_trade_step = 0
        
        # Reward tracking variables
        self.peak_portfolio_value = self.initial_cash
        self.running_mean_return = 0.0
        self.running_second_moment = 1e-4
        self.running_downside_moment = 1e-4
        self.step_tx_cost = 0.0
        
        # History tracking for analysis/plotting
        self.history = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_step = 0
        self.cash = self.initial_cash
        self.holdings = 0.0
        self.portfolio_value = self.initial_cash
        self.initial_price = self.df.iloc[0]['Close']
        self.average_buy_price = 0.0
        self.last_trade_step = 0
        
        # Reward tracking variables
        self.peak_portfolio_value = self.initial_cash
        self.running_mean_return = 0.0
        self.running_second_moment = 1e-4
        self.running_downside_moment = 1e-4
        self.step_tx_cost = 0.0
        
        self.history = []

        # Record initial state
        self._record_history(action=0, reward=0.0)

        # Clear and fill the observation queue with the initial raw observation
        self.obs_queue.clear()
        raw_obs = self._get_raw_observation()
        for _ in range(self.history_len):
            self.obs_queue.append(raw_obs)

        observation = np.concatenate(list(self.obs_queue))
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
        executed_action = 0  # 0 = Hold, 1 = Buy, 2 = Sell (stores mapped action for logs)
        prev_portfolio_value = self.portfolio_value

        tx_cost = 0.0
        
        # Action mapping
        trade_type = 0  # 0 = Hold, 1 = Buy, 2 = Sell
        trade_fraction = 0.0
        
        if self.action_space_type == "discrete_3":
            if action == 1:
                trade_type = 1
                trade_fraction = self.trade_volume_fraction
            elif action == 2:
                trade_type = 2
                trade_fraction = self.trade_volume_fraction
        elif self.action_space_type == "discrete_7":
            if action == 1:
                trade_type = 1
                trade_fraction = 0.25
            elif action == 2:
                trade_type = 1
                trade_fraction = 0.50
            elif action == 3:
                trade_type = 1
                trade_fraction = 1.00
            elif action == 4:
                trade_type = 2
                trade_fraction = 0.25
            elif action == 5:
                trade_type = 2
                trade_fraction = 0.50
            elif action == 6:
                trade_type = 2
                trade_fraction = 1.00

        if trade_type == 1:  # Buy
            buy_price = current_price * (1.0 + self.slippage)
            # Calculate how much cash we can allocate
            cash_to_spend = self.cash * trade_fraction
            if cash_to_spend > 0:
                # Deduct transaction fees
                total_cost_per_share = buy_price * (1.0 + self.transaction_fee)
                shares_bought = cash_to_spend / total_cost_per_share
                
                # Update average buy price before updating holdings
                total_shares_before = self.holdings
                self.holdings += shares_bought
                self.cash -= cash_to_spend
                
                # Transaction fee cost
                tx_cost = cash_to_spend * self.transaction_fee
                
                # Average buy price is calculated on the transaction price (including fee/slippage)
                if self.holdings > 0:
                    self.average_buy_price = (total_shares_before * self.average_buy_price + cash_to_spend) / self.holdings
                
                self.last_trade_step = self.current_step
                executed_action = 1
                
        elif trade_type == 2:  # Sell
            sell_price = current_price * (1.0 - self.slippage)
            shares_to_sell = self.holdings * trade_fraction
            if shares_to_sell > 0:
                # Add cash after deducting fees
                cash_received = shares_to_sell * sell_price * (1.0 - self.transaction_fee)
                
                self.holdings -= shares_to_sell
                self.cash += cash_received
                
                # Transaction fee cost
                tx_cost = shares_to_sell * sell_price * self.transaction_fee
                
                if self.holdings <= 1e-9:
                    self.holdings = 0.0
                    self.average_buy_price = 0.0
                
                self.last_trade_step = self.current_step
                executed_action = 2

        self.step_tx_cost = tx_cost

        # Move to next step (next day)
        self.current_step += 1
        
        # Calculate new portfolio value based on next day's close price
        next_row = self.df.iloc[self.current_step]
        next_price = next_row['Close']
        self.portfolio_value = self.cash + (self.holdings * next_price)
        self.peak_portfolio_value = max(self.peak_portfolio_value, self.portfolio_value)

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

    def _get_raw_observation(self):
        # Extract features for the current step
        features = self.df.iloc[self.current_step][self.feature_cols].values.astype(np.float32)

        # Normalize portfolio state elements to prevent scale issues in Neural Networks
        norm_cash = np.array([self.cash / self.portfolio_value], dtype=np.float32)
        current_price = self.df.iloc[self.current_step]['Close']
        norm_position = np.array([(self.holdings * current_price) / self.portfolio_value], dtype=np.float32)
        norm_return = np.array([(self.portfolio_value - self.initial_cash) / self.initial_cash], dtype=np.float32)

        portfolio_obs = [norm_cash, norm_position, norm_return]

        # Extra portfolio features for state_6 and above
        if self.num_portfolio_features == 6:
            # 4. Average Buy Price Ratio: (Close - average_buy_price) / Close
            if self.holdings > 0 and self.average_buy_price > 0:
                avg_buy_ratio = np.array([(current_price - self.average_buy_price) / current_price], dtype=np.float32)
            else:
                avg_buy_ratio = np.array([0.0], dtype=np.float32)
            
            # 5. Unrealized PnL Ratio: (holdings * Close - holdings * average_buy_price) / portfolio_value
            if self.holdings > 0:
                unrealized_pnl_ratio = np.array([(self.holdings * (current_price - self.average_buy_price)) / self.portfolio_value], dtype=np.float32)
            else:
                unrealized_pnl_ratio = np.array([0.0], dtype=np.float32)
            
            # 6. Time Since Last Trade: (current_step - last_trade_step) / 252.0
            time_since_trade = np.array([(self.current_step - self.last_trade_step) / 252.0], dtype=np.float32)
            
            portfolio_obs.extend([avg_buy_ratio, unrealized_pnl_ratio, time_since_trade])

        # Concatenate features and portfolio state
        obs = np.concatenate([features] + portfolio_obs)
        return obs

    def _get_observation(self):
        raw_obs = self._get_raw_observation()
        self.obs_queue.append(raw_obs)
        return np.concatenate(list(self.obs_queue))

    def _calculate_reward(self, prev_portfolio_value):
        daily_return = (self.portfolio_value - prev_portfolio_value) / prev_portfolio_value
        norm_tx_cost = self.step_tx_cost / prev_portfolio_value
        
        # 1. Simple Portfolio Return: % change in portfolio value
        if self.reward_type == "portfolio_return":
            reward = daily_return
            
        # 2. Log Return of Portfolio Value
        elif self.reward_type == "log_return":
            reward = np.log(self.portfolio_value / prev_portfolio_value)
            
        # 3. Portfolio Return minus Transaction Cost
        elif self.reward_type == "return_minus_fee":
            reward = daily_return - norm_tx_cost
            
        # 4. Portfolio Return minus Drawdown Penalty
        elif self.reward_type == "return_minus_drawdown":
            drawdown = (self.peak_portfolio_value - self.portfolio_value) / self.peak_portfolio_value
            reward = daily_return - 0.1 * drawdown
            
        # 5. Portfolio Return minus Volatility Penalty
        elif self.reward_type == "return_minus_volatility":
            vol_penalty = 0.0
            if 'Rolling_Std_Return_10' in self.df.columns:
                rolling_std = self.df.iloc[self.current_step]['Rolling_Std_Return_10']
                current_price = self.df.iloc[self.current_step]['Close']
                position_ratio = (self.holdings * current_price) / self.portfolio_value
                vol_penalty = rolling_std * position_ratio
            reward = daily_return - 0.5 * vol_penalty
            
        # 6. Differential Sharpe Reward
        elif self.reward_type == "diff_sharpe":
            eta = 0.1
            delta_A = daily_return - self.running_mean_return
            delta_B = daily_return**2 - self.running_second_moment
            
            variance = self.running_second_moment - self.running_mean_return**2
            if variance > 1e-8:
                reward = (self.running_second_moment * delta_A - 0.5 * self.running_mean_return * delta_B) / (variance ** 1.5)
            else:
                reward = daily_return
                
            self.running_mean_return += eta * delta_A
            self.running_second_moment += eta * delta_B
            
        # 7. Differential Sortino Reward
        elif self.reward_type == "diff_sortino":
            eta = 0.1
            daily_return_minus = min(daily_return, 0.0)
            delta_A = daily_return - self.running_mean_return
            delta_B_minus = daily_return_minus**2 - self.running_downside_moment
            
            downside_variance = self.running_downside_moment
            if downside_variance > 1e-8:
                reward = (self.running_downside_moment * delta_A - 0.5 * self.running_mean_return * delta_B_minus) / (downside_variance ** 1.5)
            else:
                reward = daily_return
                
            self.running_mean_return += eta * delta_A
            self.running_downside_moment += eta * delta_B_minus
            
        # 8. Hybrid Reward
        elif self.reward_type == "hybrid":
            drawdown = (self.peak_portfolio_value - self.portfolio_value) / self.peak_portfolio_value
            vol_penalty = 0.0
            if 'Rolling_Std_Return_10' in self.df.columns:
                rolling_std = self.df.iloc[self.current_step]['Rolling_Std_Return_10']
                current_price = self.df.iloc[self.current_step]['Close']
                position_ratio = (self.holdings * current_price) / self.portfolio_value
                vol_penalty = rolling_std * position_ratio
            reward = daily_return - norm_tx_cost - 0.1 * drawdown - 0.5 * vol_penalty
            
        # Sharpe-inspired (deprecated / backward compatible)
        elif self.reward_type == "risk_adjusted":
            vol_penalty = 0.0
            if 'BB_Std' in self.df.columns:
                current_std = self.df.iloc[self.current_step]['BB_Std']
                current_price = self.df.iloc[self.current_step]['Close']
                norm_std = current_std / current_price
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
