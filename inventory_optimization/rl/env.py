"""
Gymnasium-compatible inventory control environment.

InventoryEnv
    A finite-horizon, single-item, stochastic inventory control environment
    following the OpenAI Gymnasium API.

State space
-----------
    obs = [inventory_level, demand_forecast_t, demand_forecast_t+1, ..., demand_forecast_t+k]
    (k = forecast_horizon, default 3)

Action space
------------
    Continuous: order_quantity in [0, max_order]

Reward
------
    r_t = -(ordering_cost * I(q_t > 0) + holding_cost * max(inv_t, 0)
            + backorder_cost * max(-inv_t, 0))

    Negative cost is used as reward so the agent maximises reward ≡ minimises cost.

Demand
------
    Stochastic: D_t ~ N(mu, sigma^2), clipped to [0, max_demand].
    A fixed demand sequence can be provided via ``demand_sequence`` for
    reproducible benchmarking against classical models.

Usage
-----
>>> from inventory_optimization.rl.env import InventoryEnv
>>> env = InventoryEnv()
>>> obs, info = env.reset(seed=0)
>>> obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "gymnasium is required for the RL module. Install with: pip install gymnasium"
    ) from exc


class InventoryEnv(gym.Env):
    """Single-item stochastic inventory control environment.

    Parameters
    ----------
    n_periods : int
        Planning horizon length.
    ordering_cost : float
        Fixed setup cost per non-zero order (S).
    unit_cost : float
        Per-unit purchase cost (c).
    holding_cost_rate : float
        Fractional holding cost; per-unit holding = c * holding_cost_rate.
    backorder_cost : float
        Per-unit-per-period cost for unmet demand.
    mean_demand : float
        Mean per-period demand (mu).
    std_demand : float
        Std dev of per-period demand (sigma).
    max_order : float
        Maximum order quantity per period (action upper bound).
    max_inventory : float
        Maximum inventory level (observation bound).
    forecast_horizon : int
        Number of future demand forecasts included in the observation.
    demand_sequence : list[float], optional
        Fixed demand sequence for deterministic benchmarking. Overrides
        stochastic sampling if provided.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        n_periods: int = 12,
        ordering_cost: float = 160.0,
        unit_cost: float = 5.0,
        holding_cost_rate: float = 0.1,
        backorder_cost: float = 50.0,
        mean_demand: float = 100.0,
        std_demand: float = 20.0,
        max_order: float = 1000.0,
        max_inventory: float = 2000.0,
        forecast_horizon: int = 3,
        demand_sequence: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        self.n_periods = n_periods
        self.ordering_cost = ordering_cost
        self.unit_cost = unit_cost
        self._h = holding_cost_rate * unit_cost
        self.backorder_cost = backorder_cost
        self.mu = mean_demand
        self.sigma = std_demand
        self.max_order = max_order
        self.max_inventory = max_inventory
        self.forecast_horizon = forecast_horizon
        self.demand_sequence = demand_sequence

        # Gymnasium spaces
        obs_dim = 1 + forecast_horizon + 1  # inventory + forecasts + period fraction
        self.observation_space = spaces.Box(
            low=np.full(obs_dim, -max_inventory, dtype=np.float32),
            high=np.full(obs_dim, max_inventory + max_order, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.float32(0.0),
            high=np.float32(max_order),
            shape=(1,),
            dtype=np.float32,
        )

        # Episode state (initialised in reset())
        self._inventory: float = 0.0
        self._t: int = 0
        self._demands: np.ndarray = np.zeros(n_periods)
        self._rng: np.random.Generator = np.random.default_rng()
        self._total_cost: float = 0.0
        self._cost_history: List[float] = []

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)

        if self.demand_sequence is not None:
            # Pad / truncate to n_periods
            d = list(self.demand_sequence)
            if len(d) < self.n_periods:
                d = d + [0.0] * (self.n_periods - len(d))
            self._demands = np.array(d[: self.n_periods], dtype=float)
        else:
            raw = self._rng.normal(self.mu, self.sigma, self.n_periods + self.forecast_horizon)
            self._demands = np.clip(raw, 0.0, None)

        self._inventory = float(self.mu * 2)  # Start with 2-period supply
        self._t = 0
        self._total_cost = 0.0
        self._cost_history = []

        return self._obs(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        order = float(np.clip(action[0], 0.0, self.max_order))

        # Receive order (zero lead time)
        self._inventory += order

        # Satisfy demand
        demand = float(self._demands[self._t])
        self._inventory -= demand

        # Cost this period
        setup = self.ordering_cost if order > 0 else 0.0
        holding = max(self._inventory, 0.0) * self._h
        backorder = max(-self._inventory, 0.0) * self.backorder_cost
        period_cost = setup + holding + backorder
        self._total_cost += period_cost
        self._cost_history.append(period_cost)

        reward = -period_cost

        self._t += 1
        terminated = self._t >= self.n_periods
        truncated = False

        info = {
            "period": self._t,
            "demand": demand,
            "order": order,
            "inventory": self._inventory,
            "period_cost": period_cost,
            "total_cost": self._total_cost,
        }

        return self._obs(), reward, terminated, truncated, info

    def _obs(self) -> np.ndarray:
        forecasts = []
        for k in range(self.forecast_horizon):
            idx = self._t + k
            if idx < len(self._demands):
                forecasts.append(float(self._demands[idx]))
            else:
                forecasts.append(0.0)
        period_frac = self._t / self.n_periods
        obs = np.array(
            [self._inventory] + forecasts + [period_frac],
            dtype=np.float32,
        )
        return obs

    def get_episode_cost(self) -> float:
        return self._total_cost

    def get_cost_history(self) -> List[float]:
        return list(self._cost_history)
