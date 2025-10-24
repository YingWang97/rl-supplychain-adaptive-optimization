import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, Any

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:  # fallback if gymnasium is missing at runtime
    class _Dummy:
        def __getattr__(self, k): raise ImportError("Please install gymnasium")
    gym = _Dummy()
    spaces = _Dummy()

from .sim.demand import DemandProcess

@dataclass
class CostParams:
    holding: float = 1.0
    backlog: float = 5.0
    order: float = 0.5
    expedite: float = 2.0

class SupplyChainEnv(gym.Env):
    """
    Periodic-review, single-SKU, multi-echelon (2-level) inventory control.
    Actions: order quantity (regular) and expedite quantity (air).
    Demand: stochastic, possibly nonstationary (regime schedule).
    """
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        horizon: int = 52,
        max_order: int = 100,
        max_expedite: int = 30,
        lead_time_regular: int = 2,
        lead_time_expedite: int = 0,
        capacity: int = 200,
        lost_sales: bool = False,
        seed: int = 42,
        cost: CostParams = CostParams(),
        demand_cfg: Dict[str, Any] = None,
    ):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.horizon = horizon
        self.t = 0

        self.max_order = max_order
        self.max_expedite = max_expedite
        self.lead_time_regular = lead_time_regular
        self.lead_time_expedite = lead_time_expedite
        self.capacity = capacity
        self.lost_sales = lost_sales
        self.cost = cost
        self.demand = DemandProcess(seed=seed, **(demand_cfg or {}))

        # State: [on_hand, backlog, pipeline_regular[L], pipeline_expedite[E]]
        obs_high = np.array([capacity, capacity, *([max_order]*lead_time_regular), *([max_expedite]*max(1, lead_time_expedite))], dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0, high=obs_high, dtype=np.float32)

        # Action: [order_regular, order_expedite]
        self.action_space = spaces.Box(low=np.array([0.0, 0.0]), high=np.array([float(max_order), float(max_expedite)]), dtype=np.float32)

        self._reset_internal()

    def _reset_internal(self):
        self.t = 0
        self.on_hand = self.capacity // 4
        self.backlog = 0
        self.pipeline_regular = [0]*self.lead_time_regular
        self.pipeline_expedite = [0]*max(1, self.lead_time_expedite)
        self.last_demand = 0
        return self._obs()

    def _obs(self):
        return np.array([self.on_hand, self.backlog, *self.pipeline_regular, *self.pipeline_expedite], dtype=np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.demand.reset(seed)
        obs = self._reset_internal()
        return obs, {}

    def step(self, action: np.ndarray):
        action = np.clip(action, self.action_space.low, self.action_space.high).astype(int)
        order_regular, order_expedite = int(action[0]), int(action[1])

        # Receive arrivals
        arrivals = 0
        if self.lead_time_regular > 0:
            arrivals += self.pipeline_regular.pop(0)
            self.pipeline_regular.append(order_regular)
        else:
            arrivals += order_regular

        if self.lead_time_expedite > 0:
            arrivals += self.pipeline_expedite.pop(0)
            self.pipeline_expedite.append(order_expedite)
        else:
            arrivals += order_expedite

        # Update inventory with arrivals
        self.on_hand = min(self.capacity, self.on_hand + arrivals)

        # Realize demand
        d = self.demand.sample(t=self.t)
        self.last_demand = d

        satisfied = min(self.on_hand, d + self.backlog)
        self.on_hand -= satisfied
        unmet = d + self.backlog - satisfied

        if self.lost_sales:
            lost = unmet
            self.backlog = 0
        else:
            lost = 0
            self.backlog = unmet

        # Costs
        holding_cost = self.cost.holding * self.on_hand
        backlog_cost = self.cost.backlog * self.backlog + 10.0 * lost  # penalize lost heavily
        order_cost = self.cost.order * order_regular
        expedite_cost = self.cost.expedite * order_expedite
        total_cost = holding_cost + backlog_cost + order_cost + expedite_cost

        reward = -float(total_cost)
        self.t += 1
        terminated = self.t >= self.horizon
        truncated = False
        info = {"demand": d, "satisfied": satisfied, "lost": lost, "cost_breakdown": {
            "holding": holding_cost, "backlog": backlog_cost, "order": order_cost, "expedite": expedite_cost
        }}
        return self._obs(), reward, terminated, truncated, info

    def render(self):
        print(f"t={self.t} on_hand={self.on_hand} backlog={self.backlog} last_demand={self.last_demand}")
