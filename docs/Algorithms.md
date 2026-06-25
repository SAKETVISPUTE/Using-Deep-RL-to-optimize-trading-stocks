# Reinforcement Learning Algorithms (Baseline - V1)

This document describes the reinforcement learning model and training configuration.

---

## 1. Primary Algorithm: Proximal Policy Optimization (PPO)

We utilize the stable and highly benchmarked **PPO** algorithm from the Stable-Baselines3 library. PPO belongs to the family of Policy Gradient methods and uses a surrogate objective to prevent the policy from shifting too far during gradient steps.

### Clipped Surrogate Objective
PPO optimizes the following objective function:

$$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t) \right]$$

Where:
* $r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{old}}(a_t | s_t)}$ is the probability ratio between the new policy and the old policy.
* $\hat{A}_t$ is the estimated advantage at step $t$.
* $\epsilon$ is the clipping parameter (default is `0.2`), which prevents the policy update from exceeding a safe boundary.

---

## 2. Policy Network Architecture
* **Policy Type**: Multi-Layer Perceptron (MLP) Actor-Critic network.
* **Hidden Layers**:
  - Actor network: 2 hidden layers of size 128 (`[128, 128]`)
  - Critic network: 2 hidden layers of size 128 (`[128, 128]`)
* **Activation Function**: Hyperbolic Tangent (`tanh`).
* **Output Layers**:
  - Actor: Softmax over the 3 discrete actions.
  - Critic: Flat value estimate representing expected cumulative return.
