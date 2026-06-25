# Research Discussion & Interpretation

This document provides a deep quantitative and reinforcement learning analysis of the Version 1 results, and lays the theoretical groundwork for the Version 2 research campaigns.

---

## 1. Analysis of V1 Policy Behavior (The Buy-and-Hold Trap)

Our experiments showed that the PPO agent trained on Reliance, TCS, and HDFC Bank converged to a strategy that executes a single buy at the beginning of each test window and holds it.

### Why did the agent converge to Buy & Hold?
In reinforcement learning, the agent learns by exploring a policy $\pi_\theta(a|s)$ and adjusting parameters based on the advantage $A(s, a)$.
1. **Strong Bull Market Drift**: Over the lookback period (2015-2020), large-cap Indian stocks experienced a steady upward drift. The state transition dynamics are such that the transition from step $t$ to $t+1$ has a positive price expectation: $\mathbb{E}[Close_{t+1} - Close_t] > 0$.
2. **Transaction Friction**: Every trade incurs a 0.1% fee and 0.05% slippage. If the agent trades frequently, it pays the friction penalty on both entry and exit:
   $$\text{Total friction per round trade} \approx 2 \times (0.001 + 0.0005) = 0.3\%$$
   To beat Buy and Hold, an active strategy must generate enough excess return (alpha) to cover this 0.3% penalty on every single trade round.
3. **Reward Shape**: The reward is simple portfolio return. Because passive holding avoids repeated friction costs and reaps the steady bull market drift, the value network (critic) estimates a very high state value for holding the asset.
4. **Policy Gradient Path**: The policy network quickly updates parameters to maximize returns by suppressing "Sell" actions (which would trigger fees and cause the agent to lose exposure to the rising stock) and reinforcing immediate "Buy" actions, ending up at a passive Buy & Hold policy.

---

## 2. The Infosys Anomaly (Why did PPO succeed?)

For Infosys (`INFY`), the rolling-window walk-forward agent successfully beat Buy & Hold (+73.24% vs. +64.56%) while reducing drawdown.

### The Dynamics of tech sector crash
In 2022, IT sector valuations collapsed (INFY dropped from a peak of ~₹1,900 to below ~₹1,300).
- **The Buy & Hold baseline** bought at the start of 2021 (near the tech rally peak) and held through the entire 2022 collapse, experiencing a maximum drawdown of -35.56%.
- **The Walk-Forward Agent** for Window 2 (tested in 2022) was trained on the 2016-2021 period, which included the covid-19 crash of 2020. This training gave the agent experience in recognizing bearish regimes. 
- During 2022, the agent did not buy immediately at the start of the year. Instead, it timed its entries, entering at lower price levels and reducing its exposure during the worst drawdown days. This saved capital and allowed the agent to ride the subsequent recovery in 2023-2024 with a higher starting balance, creating alpha.

---

## 3. Transition to Version 2 Research Campaigns
The V1 results highlight that:
1. Standard daily return rewards lead to passive holding.
2. Binary discrete actions (full buy / full sell) limit the agent's ability to scale positions.
3. We need richer state features to capture complex regimes.

Therefore, Version 2 will systematically investigate:
- **Campaign 1: State Representation**: Build progressively richer state representations (adding intraday return, range, stochastic, volume ratios).
- **Campaign 2: Reward Engineering**: Design and test risk-adjusted rewards (drawdown penalties, volatility penalties, differential Sharpe/Sortino ratios) to encourage active risk-hedging and avoid the Buy-and-Hold trap.
- **Campaign 3: Action Space**: Test fractional size allocations (Buy/Sell 25%, 50%, 100%) to allow accumulation and distribution.
