import numpy as np
from typing import Dict, Any

class DemandProcess:
    """
    Piecewise-stationary demand with optional seasonality and promotion spikes.
    Config example:
    {
        "base_mean": 20, "base_std": 5,
        "regimes": [{"t":0,"mean":20,"std":5},
                    {"t":26,"mean":35,"std":8}],
        "seasonal_amp": 5,
        "seasonal_period": 7,
        "promotion": {"prob":0.05, "lift":15},
        "clip_low": 0,
        "clip_high": 200
    }
    """
    def __init__(self, seed: int = 0, **cfg: Dict[str, Any]):
        self.rng = np.random.default_rng(seed)
        self.cfg = cfg
        self.base_mean = cfg.get("base_mean", 20)
        self.base_std = cfg.get("base_std", 5)
        self.regimes = cfg.get("regimes", [])
        self.seasonal_amp = cfg.get("seasonal_amp", 0.0)
        self.seasonal_period = cfg.get("seasonal_period", 7)
        self.promo = cfg.get("promotion", {"prob": 0.0, "lift": 0.0})
        self.clip_low = cfg.get("clip_low", 0)
        self.clip_high = cfg.get("clip_high", 200)

    def reset(self, seed: int = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def _regime_params(self, t: int):
        mean, std = self.base_mean, self.base_std
        for r in self.regimes:
            if t >= r.get("t", 0):
                mean = r.get("mean", mean)
                std = r.get("std", std)
        return mean, std

    def sample(self, t: int):
        mean, std = self._regime_params(t)
        # Add seasonality
        if self.seasonal_amp > 0 and self.seasonal_period > 0:
            mean = mean + self.seasonal_amp * np.sin(2*np.pi * (t % self.seasonal_period) / self.seasonal_period)
        # Draw base demand
        d = self.rng.normal(mean, std)
        # Promotion spikes
        if self.rng.uniform() < self.promo.get("prob", 0.0):
            d += self.promo.get("lift", 0.0)
        # Clip and discretize to integers
        d = int(np.clip(d, self.clip_low, self.clip_high))
        return d
