# Decision Log

This document records the justification and trade-offs for all major design decisions made throughout the project.

---

## Decision 1 — Adjusted Prices instead of Nominal Close Prices
- **Date**: 2026-06-25
- **Alternatives considered**:
  - *Nominal Close Prices*: Keep prices raw.
  - *Adjusted Close Prices*: Modify historical data.
- **Trade-offs**:
  - *Nominal Close Prices*: Easy to fetch but introduces massive artificial discontinuities (jumps) during corporate stock splits and dividends, which break the state transition dynamics of the environment.
  - *Adjusted Close Prices*: Alters the absolute historical scale of prices but preserves the exact percentage daily returns, ensuring that price series changes represent actual economic returns.
- **Final Reason**: Adjusted Close is mandatory for financial ML to prevent model updates from getting corrupted by splits/dividends.

---

## Decision 2 — Dimensionless Scaling of Technical Indicators
- **Date**: 2026-06-25
- **Alternatives considered**:
  - *Raw Price Indicators*: Keep EMA/Bollinger Bands in price terms.
  - *Standardization (Z-Score)*: Standardize using historical mean and standard deviation.
  - *Dimensionless Ratios*: Divide indicators by current Close price.
- **Trade-offs**:
  - *Raw Price Indicators*: Causes neural network overflow and gradient explosion (`nan` logits) due to high absolute values (thousands of Rupees).
  - *Standardization*: Good but requires keeping track of rolling means/stds to prevent lookahead bias.
  - *Dimensionless Ratios*: Simple, stationary, local, prevents lookahead leakage, and scales features directly between approximately $[-1.0, 1.0]$.
- **Final Reason**: Dimensionless ratio scaling successfully resolved the `nan` logits training bug while maintaining mathematical stationarity.

---

## Decision 3 — Walk-Forward Validation instead of Standard K-Fold Cross-Validation
- **Date**: 2026-06-25
- **Alternatives considered**:
  - *K-Fold Cross-Validation*: Random splits.
  - *Single Train/Test Split*: Train on 2015-2022, test on 2023-2025.
  - *Walk-Forward (Rolling) Validation*: Sequential rolling train/test windows.
- **Trade-offs**:
  - *K-Fold*: Causes major lookahead leakage (future prices leak into past training), rendering results invalid.
  - *Single Split*: Evaluates performance on only one specific market regime, leading to biased conclusions.
  - *Walk-Forward*: Computationally expensive (requires re-training the model multiple times) but mimics real-world trading deployment, preserves temporal order, and tests across multiple distinct out-of-sample years (market cycles).
- **Final Reason**: Walk-forward validation provides the most realistic, unbiased out-of-sample evaluation.

---

## Decision 4 — Dynamic Feature-Group Filtering in Environment
- **Date**: 2026-06-25
- **Alternatives considered**:
  - *Hardcoded Files*: Create separate feature calculation and training scripts for every experiment.
  - *Dynamic Filtering*: Calculate all technical indicators in a single `FeatureEngineer` module, and configure `TradingEnv` to filter the state representation columns dynamically via a `--feature-group` parameter.
- **Trade-offs**:
  - *Hardcoded Files*: Easy to isolate, but creates duplicate code and increases code maintenance and bug risk.
  - *Dynamic Filtering*: Requires robust column-filtering mapping in `TradingEnv`, but isolates changes to a single parameter, ensuring 100% consistent preprocessing, fees, and parameters across experiments.
- **Final Reason**: Dynamic filtering guarantees that experiments are strictly controlled and reproducible, minimizing implementation variance.

