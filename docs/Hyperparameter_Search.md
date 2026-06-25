# Hyperparameter Search (Baseline - V1)

This document records the hyperparameter configuration for the baseline PPO training.

In Version 1, we use a static, standard hyperparameter set. Systematic hyperparameter tuning (using Optuna or grid search) is planned for the **Version 2 Hyperparameter Tuning Campaign**.

---

## 1. Baseline Hyperparameter Set
The hyperparameters are defined in [ppo_config.yaml](file:///mnt/c/Users/Saket/Desktop/Projects/Finsearch_RL/configs/ppo_config.yaml):

| Hyperparameter | Value | Description |
|---|---|---|
| **Learning Rate** | `0.0003` | Standard step size for Adam optimizer. |
| **n_steps (Rollout)** | `2048` | Steps collected per environment before running gradient update. |
| **Batch Size** | `64` | Size of mini-batches fed to the policy gradient optimizer. |
| **n_epochs** | `10` | Number of optimization epochs per rollout buffer update. |
| **Gamma ($\gamma$)** | `0.99` | Discount factor (values near 1.0 favor long-term returns). |
| **GAE Lambda ($\lambda$)** | `0.95` | Factor to balance bias vs variance in Advantage estimation. |
| **Clip Range ($\epsilon$)** | `0.2` | PPO clip probability limit. |
| **Entropy Coef (`ent_coef`)** | `0.01` | Weight for entropy regularization to encourage exploration. |
| **Value Function Coef (`vf_coef`)** | `0.5` | Weight of value loss relative to policy loss. |
| **Max Gradient Norm** | `0.5` | Gradient clipping threshold. |
| **Network Architecture** | `pi: [128, 128], vf: [128, 128]` | Dense Layer widths. |

---

## 2. Rationale
These parameters are standard baseline values commonly used for discrete controls.
* **Large Rollout (`n_steps=2048`)**: Necessary in finance since single-day returns are extremely noisy, so the agent needs a long sequence of experiences to distinguish patterns from random noise.
* **Small Batch (`batch_size=64`)**: Accelerates policy updates and introduces stochasticity which helps the optimizer escape local minima.
