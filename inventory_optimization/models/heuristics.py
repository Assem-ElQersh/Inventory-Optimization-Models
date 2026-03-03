"""
Heuristic lot-sizing models.

All heuristics use numpy prefix-sum arrays so that cumulative demand and
carrying-cost segments are computed in O(1) per inner-loop iteration,
reducing the overall complexity from O(n^3) to O(n^2).

Models
------
SilverMealHeuristic
    Minimize average cost per period.  Stop extending the current order
    when adding one more period increases the average cost.

LeastUnitCostHeuristic
    Minimize average cost per unit ordered.  Stop when cost-per-unit rises.

LotForLotModel
    Order exactly the period's demand every period (no inventory held).

PartPeriodBalancingModel
    Match cumulative carrying cost to the setup cost (balancing heuristic).

ThreeMonthReplenishmentModel
    Fixed three-period review interval.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from inventory_optimization.models.base import InventoryModel


def _build_prefix(demand: List[float]) -> np.ndarray:
    """Return prefix sum array: prefix[i] = sum(demand[:i])."""
    prefix = np.zeros(len(demand) + 1)
    for i, d in enumerate(demand):
        prefix[i + 1] = prefix[i] + d
    return prefix


class SilverMealHeuristic(InventoryModel):
    """Silver-Meal heuristic: minimise average cost per period.

    At each replenishment point the algorithm extends the current order
    by one period at a time and stops as soon as the average cost per
    period (ordering + holding) starts to increase.

    Carrying cost for ordering at *period* to cover T periods:

        C_carry = h * sum_{t=1}^{T-1} d[period+t] * t

    where h = carrying_charge * unit_cost.

    Time complexity: O(n^2).
    Optimality: No — heuristic.
    """

    def calculate_cost(self) -> Tuple[float, List[Dict]]:
        demand = self.demand
        n = len(demand)
        h = self._h
        prefix = _build_prefix(demand)

        replenishments: Dict[int, float] = {}
        period = 0

        while period < n:
            # Weighted-period carrying cost accumulator
            # CC(T) = h * sum_{t=1}^{T-1} demand[period+t] * t
            # Incrementally: CC(T) = CC(T-1) + h * demand[period+T-1] * (T-1)
            carrying_cost = 0.0
            min_avg = float("inf")
            best_T = 1

            for T in range(1, n - period + 1):
                # Increment carrying cost: the new period's demand is held for T-1 extra periods
                if T > 1:
                    carrying_cost += h * demand[period + T - 1] * (T - 1)

                avg_cost = (self.ordering_cost + carrying_cost) / T

                if avg_cost > min_avg:
                    break

                min_avg = avg_cost
                best_T = T

            # Order cumulative demand for best_T periods (O(1) via prefix)
            replenishments[period] = float(
                prefix[period + best_T] - prefix[period]
            )
            period += best_T

        total_cost, details = self.evaluate_plan(replenishments)
        return total_cost, details


class LeastUnitCostHeuristic(InventoryModel):
    """Least Unit Cost heuristic: minimise average cost per unit.

    At each replenishment point extend the order one period at a time
    and stop as soon as the average cost per unit rises.

    Carrying cost when ordering Q = sum(demand[period:period+T]) at *period*:

        C_carry = h * sum_{t=0}^{T-2} (Q - prefix[period+t+1] + prefix[period])

    Using the prefix array this is O(1) per T step after incremental update.

    Time complexity: O(n^2).
    Optimality: No — heuristic.
    """

    def calculate_cost(self) -> Tuple[float, List[Dict]]:
        demand = self.demand
        n = len(demand)
        h = self._h
        prefix = _build_prefix(demand)

        replenishments: Dict[int, float] = {}
        period = 0

        while period < n:
            carrying_cost = 0.0
            cumulative_demand = 0.0
            min_cpu = float("inf")
            best_T = 1

            for T in range(1, n - period + 1):
                cumulative_demand += demand[period + T - 1]

                # Holding cost for the newly accumulated quantity:
                # Units ordered at period that will be consumed in period+T-1
                # are held for T-1 periods.
                if T > 1:
                    carried = cumulative_demand - (
                        prefix[period + T] - prefix[period]
                    )
                    # Recalculate correctly: holding = h * ending_inventory per period
                    # Ending inventory after period period+t-1 when order covers T periods:
                    # = prefix[period+T] - prefix[period+t]  (= remaining demand)
                    # For t in 0..T-2
                    pass

                # Recompute carrying cost fully (still O(n) per outer period,
                # but we amortise via incremental update below)
                # Incremental: adding period+T-1 increases holding for ALL
                # previous periods by demand[period+T-1] (it's now "extra" inventory
                # held during those periods).
                # More precisely:
                # CC(T) = h * sum_{t=0}^{T-2} (prefix[period+T] - prefix[period+t+1])
                # Incremental: CC(T) = CC(T-1) + h * demand[period+T-1] * (T-1)
                # (each of the T-1 previous periods now holds demand[period+T-1] extra)

                # Reset and use incremental formula
                pass

            # Rewrite with proper incremental logic
            carrying_cost = 0.0
            cumulative_demand = 0.0
            min_cpu = float("inf")
            best_T = 1

            for T in range(1, n - period + 1):
                d_new = demand[period + T - 1]
                # Adding d_new to the order means we carry d_new extra units
                # for the first T-1 periods already in the order
                if T > 1:
                    carrying_cost += h * d_new * (T - 1)
                cumulative_demand += d_new

                if cumulative_demand == 0:
                    continue  # Skip zero-demand periods (cost per unit undefined)

                cpu = (self.ordering_cost + carrying_cost) / cumulative_demand

                if cpu > min_cpu:
                    break

                min_cpu = cpu
                best_T = T

            replenishments[period] = float(
                prefix[period + best_T] - prefix[period]
            )
            period += best_T

        total_cost, details = self.evaluate_plan(replenishments)
        return total_cost, details


class LotForLotModel(InventoryModel):
    """Lot-for-Lot: order exactly what each period demands.

    Results in zero ending inventory every period, so holding cost is zero.
    Ordering cost is maximised (one order per period).

    Time complexity: O(n).
    Optimality: Optimal when ordering_cost = 0 or demand has many zero periods.
    """

    def calculate_cost(self) -> Tuple[float, List[Dict]]:
        replenishments = {p: d for p, d in enumerate(self.demand) if d > 0}
        total_cost, details = self.evaluate_plan(replenishments)
        return total_cost, details


class PartPeriodBalancingModel(InventoryModel):
    """Part-Period Balancing heuristic.

    Extends the current order until the cumulative carrying cost most closely
    equals the ordering/setup cost. The "economic part-period" (EPP) is:

        EPP = ordering_cost / (h)

    Accumulate part-periods T * demand[period+T-1] incrementally.

    Time complexity: O(n^2).
    Optimality: No — heuristic.
    """

    def calculate_cost(self) -> Tuple[float, List[Dict]]:
        demand = self.demand
        n = len(demand)
        h = self._h
        prefix = _build_prefix(demand)

        replenishments: Dict[int, float] = {}
        period = 0

        while period < n:
            cumulative_carrying = 0.0
            min_diff = float("inf")
            best_T = 1

            for T in range(1, n - period + 1):
                # Part-periods: demand at period+T-1 is carried for T-1 periods
                cumulative_carrying += h * (T - 1) * demand[period + T - 1]
                diff = abs(cumulative_carrying - self.ordering_cost)

                if diff > min_diff:
                    break

                min_diff = diff
                best_T = T

            replenishments[period] = float(
                prefix[period + best_T] - prefix[period]
            )
            period += best_T

        total_cost, details = self.evaluate_plan(replenishments)
        return total_cost, details


class ThreeMonthReplenishmentModel(InventoryModel):
    """Fixed three-period review interval.

    Orders cumulative demand for the next three periods at the start of
    each interval (or fewer periods if near the horizon end).

    Time complexity: O(n).
    Optimality: No — fixed-interval heuristic.
    """

    def __init__(self, *args, review_interval: int = 3, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if review_interval < 1:
            raise ValueError("review_interval must be >= 1")
        self.review_interval = review_interval

    def calculate_cost(self) -> Tuple[float, List[Dict]]:
        demand = self.demand
        n = len(demand)
        prefix = _build_prefix(demand)

        replenishments: Dict[int, float] = {}
        period = 0

        while period < n:
            end = min(period + self.review_interval, n)
            replenishments[period] = float(prefix[end] - prefix[period])
            period += self.review_interval

        total_cost, details = self.evaluate_plan(replenishments)
        return total_cost, details
