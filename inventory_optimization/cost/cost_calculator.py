"""
Standalone cost calculator for inventory replenishment plans.

Decouples cost accounting from model logic so cost functions can be
reused, tested, and extended independently (e.g., adding backorder
costs or service-level penalties).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple


class CostCalculator:
    """Calculate costs for a deterministic inventory replenishment plan.

    Assumptions
    -----------
    - Zero lead time.
    - No backorders; unfilled demand is not modelled here.
    - Purchase cost is excluded (treated as a sunk cost constant).
    - Holding cost accrues on *ending* inventory each period.

    Parameters
    ----------
    ordering_cost : float
        Fixed setup cost per order (S).
    unit_cost : float
        Unit purchase cost (c).
    carrying_charge : float
        Fractional holding rate per period (h); per-unit holding cost = c * h.
    backorder_cost : float, optional
        Penalty cost per unit of backorder per period (default 0 = no backorders).
    """

    def __init__(
        self,
        ordering_cost: float,
        unit_cost: float,
        carrying_charge: float,
        backorder_cost: float = 0.0,
    ) -> None:
        self.ordering_cost = ordering_cost
        self.unit_cost = unit_cost
        self.carrying_charge = carrying_charge
        self.backorder_cost = backorder_cost
        self._h = carrying_charge * unit_cost

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        demand: List[float],
        replenishments: Dict[int, float],
    ) -> Tuple[float, List[Dict], Dict[str, float]]:
        """Simulate a plan and return cost breakdown.

        Parameters
        ----------
        demand : list[float]
            Per-period demand (0-based period indices).
        replenishments : dict[int, float]
            Maps period index to order quantity.

        Returns
        -------
        total_cost : float
            ordering_cost + holding_cost + backorder_cost.
        details : list[dict]
            Per-period simulation records.
        breakdown : dict[str, float]
            Cost component totals: ``ordering``, ``holding``, ``backorder``.
        """
        inventory = 0.0
        holding_cost = 0.0
        total_backorder_cost = 0.0
        details: List[Dict] = []

        for period, d in enumerate(demand):
            start_inv = inventory
            order = replenishments.get(period, 0.0)
            inventory += order
            ending = inventory - d

            if ending >= 0:
                holding_cost += ending * self._h
            else:
                total_backorder_cost += abs(ending) * self.backorder_cost

            details.append(
                {
                    "Period": period + 1,
                    "Starting_Inventory": round(start_inv, 4),
                    "Replenishment": round(order, 4),
                    "Requirements": d,
                    "Ending_Inventory": round(ending, 4),
                }
            )
            inventory = ending

        # Enforce zero ending inventory
        if inventory != 0.0:
            details[-1]["Replenishment"] = round(
                details[-1]["Replenishment"] + inventory, 4
            )
            details[-1]["Ending_Inventory"] = 0.0

        total_ordering = self.ordering_cost * len(replenishments)
        total_cost = total_ordering + holding_cost + total_backorder_cost

        breakdown = {
            "ordering": round(total_ordering, 4),
            "holding": round(holding_cost, 4),
            "backorder": round(total_backorder_cost, 4),
        }
        return round(total_cost, 4), details, breakdown

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def holding_cost_segment(
        self,
        demand: List[float],
        start: int,
        end: int,
        prefix: Optional[List[float]] = None,
    ) -> float:
        """Holding cost when ordering sum(demand[start:end]) at period *start*.

        Equivalent to: h * sum_{k=start}^{end-1} sum(demand[k+1:end])

        Uses prefix sums for O(1) per call when *prefix* is supplied.

        Parameters
        ----------
        demand : list[float]
        start : int  (inclusive, 0-based)
        end : int    (exclusive, 0-based)
        prefix : list[float], optional
            Cumulative sum array where prefix[i] = sum(demand[:i]).
        """
        if prefix is None:
            prefix = [0.0] * (len(demand) + 1)
            for i, v in enumerate(demand):
                prefix[i + 1] = prefix[i] + v

        cost = 0.0
        for k in range(start, end):
            remaining = prefix[end] - prefix[k + 1]
            cost += remaining * self._h
        return cost
