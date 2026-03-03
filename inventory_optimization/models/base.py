"""
Base inventory model class and shared input validation.
"""
from __future__ import annotations

import json
from typing import Dict, List, Tuple


def validate_inputs(
    demand: List[float],
    ordering_cost: float,
    unit_cost: float,
    carrying_charge: float,
) -> None:
    """Validate parameters shared by all inventory models.

    Raises:
        TypeError:  If demand is not a list/sequence.
        ValueError: If any parameter violates domain constraints.
    """
    if not hasattr(demand, "__iter__"):
        raise TypeError("demand must be a list or sequence of numeric values")
    demand = list(demand)
    if len(demand) == 0:
        raise ValueError("demand list cannot be empty")
    if any(d < 0 for d in demand):
        raise ValueError("All demand values must be >= 0")
    if ordering_cost <= 0:
        raise ValueError(f"ordering_cost must be > 0, got {ordering_cost}")
    if unit_cost <= 0:
        raise ValueError(f"unit_cost must be > 0, got {unit_cost}")
    if carrying_charge < 0:
        raise ValueError(f"carrying_charge must be >= 0, got {carrying_charge}")


class InventoryModel:
    """Base class for deterministic, single-item, zero-lead-time inventory models.

    Assumptions (shared by all deterministic subclasses):
        - Zero ending inventory is enforced at the planning horizon end.
        - No backorders or lost sales.
        - No lead time.
        - Purchase cost is excluded from the objective (treated as sunk).

    Parameters
    ----------
    demand : list[float]
        Per-period demand over the planning horizon.
    ordering_cost : float
        Fixed cost per replenishment order (setup cost S).
    unit_cost : float
        Unit purchase / production cost (c).
    carrying_charge : float
        Fractional holding cost rate per period (h); holding cost = c * h.
    """

    def __init__(
        self,
        demand: List[float],
        ordering_cost: float,
        unit_cost: float,
        carrying_charge: float,
    ) -> None:
        validate_inputs(demand, ordering_cost, unit_cost, carrying_charge)
        self.demand: List[float] = list(demand)
        self.ordering_cost: float = ordering_cost
        self.unit_cost: float = unit_cost
        self.carrying_charge: float = carrying_charge
        self._h: float = carrying_charge * unit_cost  # holding cost per unit per period

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    def calculate_cost(self) -> Tuple[float, List[Dict]]:
        """Compute the optimal/heuristic replenishment plan and its total cost.

        Returns
        -------
        total_cost : float
            Sum of ordering costs and holding costs (purchase cost excluded).
        details : list[dict]
            Period-by-period breakdown with keys:
            Period, Starting_Inventory, Replenishment, Requirements, Ending_Inventory.
        """
        raise NotImplementedError("Subclasses must implement calculate_cost().")

    # ------------------------------------------------------------------
    # Shared cost accounting
    # ------------------------------------------------------------------

    def evaluate_plan(self, replenishments: Dict[int, float]) -> Tuple[float, List[Dict]]:
        """Evaluate a replenishment plan and return (total_cost, period_details).

        Parameters
        ----------
        replenishments : dict[int, float]
            Maps period index (0-based) to order quantity.

        Returns
        -------
        total_cost : float
        details : list[dict]
        """
        inventory = 0.0
        carrying_cost = 0.0
        details: List[Dict] = []

        for period, demand in enumerate(self.demand):
            starting_inventory = inventory
            replenishment = replenishments.get(period, 0.0)
            inventory += replenishment
            ending_inventory = inventory - demand

            carrying_cost += max(ending_inventory, 0.0) * self._h

            details.append(
                {
                    "Period": period + 1,
                    "Starting_Inventory": starting_inventory,
                    "Replenishment": replenishment,
                    "Requirements": demand,
                    "Ending_Inventory": ending_inventory,
                }
            )
            inventory = ending_inventory

        # Enforce zero ending inventory at horizon end
        if inventory != 0.0:
            details[-1]["Replenishment"] = details[-1]["Replenishment"] + inventory
            details[-1]["Ending_Inventory"] = 0.0

        total_ordering_cost = self.ordering_cost * len(replenishments)
        total_cost = carrying_cost + total_ordering_cost
        return total_cost, details

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    def save_results(self, file_name: str, results: object) -> None:
        """Persist results to a JSON file."""
        with open(file_name, "w") as fh:
            json.dump(results, fh, indent=4)
