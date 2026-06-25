# Ablation Studies

This document records the results of systematic ablation studies performed to identify which components of our framework contribute to performance.

---

## Version 1: Baseline Ablations
No formal ablation studies were performed for Version 1. 

Ablations are scheduled for the **Version 2 Ablation Campaign** once we identify the best-performing state representation and reward function configurations. Planned ablations include:
1. **Removing RSI**: Test if momentum signals are crucial.
2. **Removing MACD**: Test the impact of trend features.
3. **Removing Bollinger Bands**: Test the impact of volatility envelopes.
4. **Removing Portfolio Features**: Evaluate if cash/position ratios are needed for policy stability.
5. **Removing Market Features**: Measure the importance of NIFTY 50 return index context.
6. **Removing Transaction Costs**: Demonstrate how unrealistic the agent behaves without cost friction.
