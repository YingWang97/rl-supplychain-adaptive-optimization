# RL-SupplyChain-Adaptive-Optimization

**Short description**: Deep Reinforcement Learning for adaptive, multi-echelon supply chain control under stochastic demand. Includes a reproducible Gymnasium environment, synthetic demand generator (regime shifts & promotions), baselines (DQN, PPO), classical policies (base-stock, (s,S)), experiment configs, CI-tested utilities, and paper-ready docs.

## Highlights
- **Environment**: Multi-echelon periodic-review inventory control with lead times, backlog/lost-sales, capacity, expediting, perishability (optional).
- **Uncertainty**: Nonstationary demand (piecewise-stationary, promotions, day-of-week seasonality), parameterizable autocorrelation.
- **Policies**: DQN, PPO (PyTorch), plus **base-stock** and **(s,S)** classical heuristics for comparison.
- **Reproducibility**: Deterministic seeds, Hydra-style config files, logged metrics (CSV) and artifacts.
- **Evaluation**: Regret, service level (fill rate), stockout rate, average cost, bullwhip ratio.
- **Docs**: Paper skeleton (IMRaD), experiment templates, contribution guide, and CI.

## Quickstart
```bash
# 1) Create env
python -m venv .venv && source .venv/bin/activate
pip install -U pip

# 2) Install package
pip install -e .

# 3) Run a short training job
python scripts/train.py +experiment=toy_dqn

# 4) Evaluate a saved checkpoint
python scripts/eval.py --run_dir runs/toy_dqn
```

> If you don't have `gymnasium`/`hydra-core`, the configs fall back to native argparse.

## Repo Layout
```
.
├─ src/rlsupplychain/           # Python package
│  ├─ envs/supply_chain_env.py  # Gymnasium env
│  ├─ sim/demand.py             # Stochastic demand generator
│  ├─ policies/dqn.py           # DQN agent
│  ├─ policies/ppo.py           # PPO agent
│  ├─ policies/classic.py       # Base-stock, (s,S) heuristics
│  ├─ utils/seed.py             # Seeding utilities
│  ├─ utils/replay.py           # Simple replay buffer
│  └─ eval/metrics.py           # Metrics
├─ configs/                     # YAML configs (Hydra-style, but simple YAML is OK)
│  ├─ experiment/toy_dqn.yaml
│  └─ experiment/toy_ppo.yaml
├─ scripts/
│  ├─ train.py                  # Training entrypoint
│  └─ eval.py                   # Evaluation entrypoint
├─ experiments/                 # Experiment recipes
│  └─ README.md
├─ tests/                       # PyTest unit tests
│  └─ test_env.py
├─ docs/
│  ├─ paper.md                  # IMRaD outline with citations TODO markers
│  └─ figures/                  # Placeholder
├─ .github/workflows/ci.yml     # Lint + tests
├─ pyproject.toml               # Packaging
├─ setup.cfg                    # Flake8, isort, pytest
├─ LICENSE                      # MIT
├─ CODE_OF_CONDUCT.md
├─ CONTRIBUTING.md
├─ SECURITY.md
└─ CITATION.cff
```

## Datasets
The default uses a *synthetic* generator with explicit seeds. To reproduce paper-scale results, pin seeds in configs.

## Citation
If you use this repository:
```bibtex
@software{rl_supplychain_2025,
  title        = {RL-SupplyChain-Adaptive-Optimization},
  author       = {Your Name},
  year         = {2025},
  url          = {https://github.com/yourname/rl-supplychain-adaptive-optimization},
  note         = {Deep RL for adaptive supply chain optimization under uncertain demand.}
}
```
