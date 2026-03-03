"""
EOQ-based inventory models.

EOQModel
    Classic Economic Order Quantity adapted for time-varying demand.
    NOTE: EOQ is a *stationary* formula (constant demand assumption).
    Applying it to time-varying demand is a heuristic adaptation; the
    implementation clearly marks this and does not claim optimality.

EOQTimeSupplyModel
    Converts the EOQ quantity into a time-supply (number of periods to
    cover per order) and places fixed-interval orders accordingly.
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

from inventory_optimization.models.base import InventoryModel


class EOQModel(InventoryModel):
    """EOQ heuristic adapted for time-varying demand.

    The classic EOQ formula is:

        Q* = sqrt(2 * S * D_avg / (h * c))

    where D_avg is the average per-period demand, S is the ordering cost,
    h is the carrying charge, and c is the unit cost.

    Because demand varies over time, this is a **heuristic adaptation**:
    each time stock runs out the model orders the cumulative demand quantity
    (starting from the current period) that is closest in magnitude to Q*.
    This guarantees zero ending inventory before the next order.

    Time complexity: O(n^2) worst case.
    Optimality: No — heuristic adaptation of a stationary formula.
    """

    def _eoq(self) -> float:
        avg_demand = sum(self.demand) / len(self.demand)
        return math.sqrt(
            (2.0 * self.ordering_cost * avg_demand) / self._h
        )

    def _best_order_quantity(self, start_period: int) -> float:
        """Return the cumulative demand amount (from start_period) closest to EOQ."""
        target = self._eoq()
        cumulative = 0.0
        best_qty = 0.0
        closest_diff = float("inf")

        for p in range(start_period, len(self.demand)):
            cumulative += self.demand[p]
            diff = abs(cumulative - target)
            if diff < closest_diff:
                closest_diff = diff
                best_qty = cumulative

        return best_qty

    def calculate_cost(self) -> Tuple[float, List[Dict]]:
        replenishments: Dict[int, float] = {}
        inventory = 0.0
        period = 0

        while period < len(self.demand):
            if inventory < self.demand[period]:
                qty = self._best_order_quantity(period)
                replenishments[period] = qty
                inventory += qty
            inventory -= self.demand[period]
            period += 1

        total_cost, details = self.evaluate_plan(replenishments)
        return total_cost, details


class EOQTimeSupplyModel(InventoryModel):
    """EOQ expressed as a time supply (fixed review interval).

    Converts Q* into a number of periods T* = Q* / D_avg, then orders
    sum(demand[t : t+T*]) at the start of each interval.

    Time complexity: O(n).
    Optimality: No — heuristic; same stationary-demand caveat as EOQModel.
    """

    def calculate_cost(self) -> Tuple[float, List[Dict]]:
        avg_demand = sum(self.demand) / len(self.demand)
        if avg_demand == 0:
            # All zero demand: one "empty" order covers everything
            total_cost, details = self.evaluate_plan({0: 0.0})
            return total_cost, details
        eoq = math.sqrt((2.0 * self.ordering_cost * avg_demand) / self._h)
        periods_to_cover = max(1, round(eoq / avg_demand))

        replenishments: Dict[int, float] = {}
        period = 0

        while period < len(self.demand):
            end = min(period + periods_to_cover, len(self.demand))
            replenishments[period] = sum(self.demand[period:end])
            period += periods_to_cover

        total_cost, details = self.evaluate_plan(replenishments)
        return total_cost, details
