# Agent Training & Hyperparameter Configuration

This document outlines the reinforcement learning agent architecture, training pipeline, hyperparameter configuration, and a critical quantitative scaling bug we resolved during implementation.

The training pipeline is implemented in [train_agent.py](file:///mnt/c/Users/Saket/Desktop/Projects/Finsearch_RL/src/training/train_agent.py), configured via [ppo_config.yaml](file:///mnt/c/Users/Saket/Desktop/Projects/Finsearch_RL/configs/ppo_config.yaml), and runs on the custom [trading_env.py](file:///mnt/c/Users/Saket/Desktop/Projects/Finsearch_RL/src/environment/trading_env.py).

---

## 1. Agent Architecture: Proximal Policy Optimization (PPO)

We utilize the **Proximal Policy Optimization (PPO)** algorithm from Stable-Baselines3. PPO is an on-policy actor-critic algorithm that balances policy optimization with training stability.

### The Actor-Critic Network
- **Policy Network (Actor)**: Maps the 19-dimensional observation space to a probability distribution over the 3 discrete actions (Hold, Buy, Sell).
- **Value Network (Critic)**: Estimates the expected cumulative future return (value) of the current state, which is used to compute the advantage: $A(s, a) = Q(s, a) - V(s)$.
- **Architecture**: We configure a custom network topology with separate actor and critic paths:
  - Actor: 2 fully-connected layers of size 128 (`[128, 128]`)
  - Critic: 2 fully-connected layers of size 128 (`[128, 128]`)
  - Activation function: Hyperbolic Tangent (`tanh`).

---

## 2. Hyperparameter Settings

The hyperparameters are defined in [ppo_config.yaml](file:///mnt/c/Users/Saket/Desktop/Projects/Finsearch_RL/configs/ppo_config.yaml):

* **Learning Rate**: `0.0003` (controls the gradient step size).
* **Discount Factor ($\gamma$)**: `0.99` (determines how much the agent values future rewards vs. immediate returns. $\gamma=0.99$ equates to an effective horizon of 100 trading days).
* **GAE Lambda ($\lambda$)**: `0.95` (parameter for Generalized Advantage Estimation to balance bias and variance).
* **Rollout Horizon (`n_steps`)**: `2048` steps per environment before running a policy update.
* **Minibatch Size**: `64` (size of data slices fed to the optimizer).
* **Optimization Epochs (`n_epochs`)**: `10` (number of times the optimizer passes over the collected rollout data during an update).
* **Entropy Coefficient (`ent_coef`)**: `0.01` (adds an entropy bonus to the loss function to encourage the policy to explore different actions, preventing premature convergence to a single action).

---

## 3. Resolving the "NaN Logits" Scaling Bug

During our initial training tests, PyTorch raised the following error:
> `ValueError: Expected parameter logits (Tensor of shape (64, 3)) of distribution Categorical to satisfy the constraint Real(), but found invalid values: nan`

### Root Cause Analysis
Financial data features like raw price averages (`EMA_10` ~ 2000.0, `BB_Upper` ~ 2200.0) and standard deviations are extremely large compared to standard neural network inputs (which typically expect values around $[-1, 1]$).
When these raw prices were multiplied by initial neural network weights, activations escalated, causing gradient explosion. This led to weights overflow, returning `nan` for the categorical policy logits.

### The Solution: Stationarity and Dimensionless Scaling
To ensure stable gradients and prevent overflow, we modified [feature_engineer.py](file:///mnt/c/Users/Saket/Desktop/Projects/Finsearch_RL/src/features/feature_engineer.py) to convert all price-denominated indicators into **dimensionless, stationary ratios** relative to the current Close price:

1. **EMA Transformation**:
   $$EMA_{scaled} = \frac{Close_t - EMA_t}{Close_t}$$
2. **MACD Transformation**:
   $$MACD_{scaled} = \frac{MACD_t}{Close_t}$$
3. **Volatility & ATR Transformation**:
   $$ATR_{scaled} = \frac{ATR_t}{Close_t}$$
4. **Bollinger Bands Transformation**:
   $$BB\_Upper_{scaled} = \frac{BB\_Upper_t - Close_t}{Close_t}$$
   $$BB\_Lower_{scaled} = \frac{Close_t - BB\_Lower_t}{Close_t}$$
5. **RSI and ROC Scaling**: Divided by $100.0$ to scale values between approximately $[0, 1]$ and $[-1, 1]$.
6. **Volume Change**: Clipped to $[-3.0, 3.0]$ to eliminate extreme spikes.

This scaling ensures that all inputs to the RL policy fall roughly within $[-1, 1]$, making training highly stable and preventing network overflow.

---

## 4. Verification Results
We ran a test training cycle of 20,000 timesteps on `RELIANCE.NS` data:
* **Training Stability**: The script ran to completion with **0 NaNs** or errors.
* **Critic Convergence**: The value function explained variance reached **0.925**, indicating the critic network has learned to predict the expected portfolio returns of states with high accuracy.
* **Exploration Behavior**: PPO actively adjusted its trade frequency and exploration probability, registering positive cumulative returns up to +87.8% during specific rollout steps.
