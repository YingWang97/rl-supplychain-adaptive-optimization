# Reinforcement Learning for Adaptive Supply Chain Optimization under Uncertain Demand

> **IMRaD skeleton** – replace TODOs and run experiments to populate figures/tables.

## Abstract
TODO: 150–250 words. State problem, method (DQN/PPO vs classical), datasets (synthetic), main results (cost ↓, fill-rate ↑).

## 1. Introduction
- Motivation: nonstationary demand, bullwhip, lead times, expediting costs.
- Contributions (bullets): new Gym env, nonstationary demand regimes, baselines, reproducible scripts.

## 2. Related Work
- Inventory control, base-stock, (s,S), DRL for supply chains, nonstationary RL.

## 3. Problem Formulation
- Multi-echelon periodic-review. State, action, transition, cost components (holding, backlog/lost-sales, ordering, expediting), constraints (capacity).

## 4. Methods
- DQN architecture, target networks, epsilon-decay.
- PPO actor-critic with clipped objective.
- Classic heuristics.

## 5. Experimental Setup
- Synthetic demand with regime shifts (table).
- Hyperparameters, seeds, evaluation metrics.

## 6. Results
- Tables: Avg cost, fill rate, stockouts.
- Plots: Learning curves, bullwhip ratio vs time.

## 7. Discussion & Limitations
- Sample efficiency, stability, covariate shift, sim-to-real gap.

## 8. Conclusion
- Summary and future directions (transfer learning, robust RL).

## References
TODO.
