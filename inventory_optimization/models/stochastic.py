"""
Stochastic demand inventory models.

Two complementary models are provided:

NewsvendorModel
    Single-period model (Newsvendor / critical-ratio).
    Finds the optimal order quantity Q* that minimises expected cost
    under normally distributed demand.

SafetyStockModel
    Multi-period (R, Q) policy with safety stock.
    Given a continuous-review policy, computes the reorder point (ROP)
    and safety stock (SS) required to achieve a target cycle service level
    under normally distributed, stationary demand with a fixed lead time.
    Also simulates the planning horizon to produce an expected-cost estimate.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats  # type: ignore[import]

from inventory_optimization.models.base import InventoryModel, validate_inputs


# ---------------------------------------------------------------------------
# Newsvendor (single-period)
# ---------------------------------------------------------------------------


@dataclass
class NewsvendorResult:
    """Results from the Newsvendor model."""

    optimal_quantity: float
    critical_ratio: float
    expected_cost: float
    expected_sales: float
    expected_leftover: float
    fill_rate: float
    service_level: float


class NewsvendorModel:
    """Single-period Newsvendor model with normal demand.

    The critical-ratio solution is:

        Q* = F^{-1}(CR)   where  CR = (p - c) / (p - s)

    and the expected cost is minimised.

    Parameters
    ----------
    mean_demand : float
        Expected demand for the period (mu).
    std_demand : float
        Standard deviation of demand (sigma).
    unit_cost : float
        Purchase / production cost per unit (c).
    selling_price : float
        Revenue per unit sold (p > c).
    salvage_value : float
        Recovery value per unsold unit at end of period (s <= c).
    shortage_penalty : float
        Additional penalty per unit of unmet demand (default 0).
    """

    def __init__(
        self,
        mean_demand: float,
        std_demand: float,
        unit_cost: float,
        selling_price: float,
        salvage_value: float,
        shortage_penalty: float = 0.0,
    ) -> None:
        if std_demand < 0:
            raise ValueError("std_demand must be >= 0")
        if selling_price <= unit_cost:
            raise ValueError("selling_price must be > unit_cost")
        if salvage_value > unit_cost:
            raise ValueError("salvage_value must be <= unit_cost")

        self.mu = mean_demand
        self.sigma = std_demand
        self.c = unit_cost
        self.p = selling_price
        self.s = salvage_value
        self.b = shortage_penalty
        self._dist = stats.norm(loc=mean_demand, scale=max(std_demand, 1e-9))

    def solve(self) -> NewsvendorResult:
        """Compute the optimal order quantity and expected metrics."""
        # Overage cost (ordered but not sold): c - s
        # Underage cost (demand not met): p - c + b
        co = self.c - self.s
        cu = self.p - self.c + self.b
        cr = cu / (cu + co)

        q_star = float(self._dist.ppf(cr))
        q_star = max(q_star, 0.0)

        # Expected sales = E[min(Q, D)]
        # = Q * F(Q) + mu*(1-F(Q)) - sigma * phi(z)
        # where z = (Q - mu) / sigma
        z = (q_star - self.mu) / max(self.sigma, 1e-9)
        phi_z = stats.norm.pdf(z)
        Phi_z = stats.norm.cdf(z)

        expected_sales = q_star * Phi_z + self.mu * (1 - Phi_z) - self.sigma * phi_z
        expected_leftover = q_star - expected_sales
        expected_shortage = self.mu - expected_sales

        expected_cost = (
            self.c * q_star
            - self.p * expected_sales
            - self.s * expected_leftover
            + self.b * expected_shortage
        )

        fill_rate = expected_sales / max(self.mu, 1e-9)

        return NewsvendorResult(
            optimal_quantity=round(q_star, 4),
            critical_ratio=round(cr, 4),
            expected_cost=round(expected_cost, 4),
            expected_sales=round(expected_sales, 4),
            expected_leftover=round(expected_leftover, 4),
            fill_rate=round(fill_rate, 4),
            service_level=round(float(Phi_z), 4),
        )


# ---------------------------------------------------------------------------
# Safety-stock / (R, Q) policy
# ---------------------------------------------------------------------------


@dataclass
class SafetyStockResult:
    """Results from the SafetyStockModel."""

    safety_stock: float
    reorder_point: float
    eoq: float
    service_level: float
    expected_total_cost: float
    simulation_details: List[Dict]


class SafetyStockModel:
    """Multi-period (R, Q) inventory policy with safety stock.

    Assumes:
      - Stationary, normally distributed demand: D ~ N(mu, sigma^2) per period.
      - Continuous review; order Q units when inventory falls to ROP.
      - Fixed lead time L periods.
      - Cycle service level (CSL): probability of no stockout per replenishment
        cycle.

    Key formulas:

        EOQ: Q* = sqrt(2 * S * mu / (h * c))
        Safety stock: SS = z_alpha * sigma * sqrt(L)
        Reorder point: ROP = mu * L + SS

    Parameters
    ----------
    mean_demand : float
        Average per-period demand (mu).
    std_demand : float
        Standard deviation of per-period demand (sigma).
    ordering_cost : float
        Fixed cost per order (S).
    unit_cost : float
        Unit purchase cost (c).
    carrying_charge : float
        Fractional holding cost rate per period (h).
    lead_time : int
        Replenishment lead time in periods (L).
    service_level : float
        Target cycle service level in [0, 1) (default 0.95).
    n_periods : int
        Number of periods to simulate for cost estimation (default 52).
    """

    def __init__(
        self,
        mean_demand: float,
        std_demand: float,
        ordering_cost: float,
        unit_cost: float,
        carrying_charge: float,
        lead_time: int = 1,
        service_level: float = 0.95,
        n_periods: int = 52,
    ) -> None:
        if std_demand < 0:
            raise ValueError("std_demand must be >= 0")
        if not 0 <= service_level < 1:
            raise ValueError("service_level must be in [0, 1)")
        if lead_time < 0:
            raise ValueError("lead_time must be >= 0")

        # Reuse base validation for scalar cost params
        validate_inputs([mean_demand], ordering_cost, unit_cost, carrying_charge)

        self.mu = mean_demand
        self.sigma = std_demand
        self.ordering_cost = ordering_cost
        self.unit_cost = unit_cost
        self.carrying_charge = carrying_charge
        self.lead_time = lead_time
        self.service_level = service_level
        self.n_periods = n_periods
        self._h = carrying_charge * unit_cost

    def solve(self, seed: Optional[int] = None) -> SafetyStockResult:
        """Compute policy parameters and simulate the planning horizon.

        Parameters
        ----------
        seed : int, optional
            Random seed for demand simulation reproducibility.

        Returns
        -------
        SafetyStockResult
        """
        # EOQ
        eoq = math.sqrt((2.0 * self.ordering_cost * self.mu) / self._h)

        # Safety stock
        z = float(stats.norm.ppf(self.service_level))
        ss = z * self.sigma * math.sqrt(max(self.lead_time, 1))
        rop = self.mu * self.lead_time + ss

        # Simulate
        rng = np.random.default_rng(seed)
        demands = rng.normal(self.mu, self.sigma, self.n_periods)
        demands = np.maximum(demands, 0.0)

        inventory = rop + eoq  # Start with a full order
        pending_order: Optional[Tuple[int, float]] = None  # (arrival_period, qty)
        total_holding = 0.0
        total_ordering = 0.0
        details: List[Dict] = []

        for t, d in enumerate(demands):
            # Receive pending order if it arrives this period
            if pending_order and pending_order[0] == t:
                inventory += pending_order[1]
                pending_order = None

            inventory -= d
            inventory = max(inventory, 0.0)  # No backorders in simulation
            total_holding += inventory * self._h

            # Trigger replenishment if inventory at or below ROP and no order pending
            replenishment = 0.0
            if inventory <= rop and pending_order is None:
                replenishment = eoq
                total_ordering += self.ordering_cost
                pending_order = (t + self.lead_time, eoq)

            details.append(
                {
                    "Period": t + 1,
                    "Demand": round(float(d), 2),
                    "Ending_Inventory": round(float(inventory), 2),
                    "Order_Placed": round(replenishment, 2),
                }
            )

        expected_total_cost = total_holding + total_ordering

        return SafetyStockResult(
            safety_stock=round(ss, 4),
            reorder_point=round(rop, 4),
            eoq=round(eoq, 4),
            service_level=self.service_level,
            expected_total_cost=round(expected_total_cost, 4),
            simulation_details=details,
        )
