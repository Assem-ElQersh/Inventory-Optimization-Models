"""
Wagner-Whitin dynamic programming algorithm for optimal lot-sizing.

The Wagner-Whitin (WW) algorithm finds the globally optimal replenishment
schedule for a finite-horizon, deterministic, time-varying demand problem.

Recurrence
----------
    C[0] = 0
    C[j] = min_{0 <= i < j} { C[i] + S + H(i, j) }   for j = 1 .. n

where S is the fixed ordering cost and H(i, j) is the holding cost when
ordering sum(demand[i..j-1]) at the start of period i:

    H(i, j) = h * sum_{k=i}^{j-1} sum(demand[k+1..j-1])
             = h * sum_{k=i}^{j-1} (prefix[j] - prefix[k+1])

Prefix sums make H(i, j) computable in O(j-i) without recomputing
full-horizon costs at every state, reducing complexity from O(n^3) to O(n^2).

Time complexity: O(n^2)
Space complexity: O(n^2)  (split table)
Optimality: Yes — globally optimal for the deterministic, zero-lead-time model.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from inventory_optimization.models.base import InventoryModel


class WagnerWhitinModel(InventoryModel):
    """Optimal lot-sizing via the Wagner-Whitin O(n^2) DP algorithm.

    Parameters are inherited from :class:`~inventory_optimization.models.base.InventoryModel`.
    """

    def calculate_cost(self) -> Tuple[float, List[Dict]]:
        """Solve the Wagner-Whitin DP and return the optimal cost and plan.

        Returns
        -------
        total_cost : float
            Optimal total cost (ordering + holding; purchase cost excluded).
        details : list[dict]
            Period-by-period inventory simulation for the optimal plan.
        """
        n = len(self.demand)
        h = self._h

        # Prefix sums: prefix[i] = sum(demand[0:i])
        prefix = np.zeros(n + 1)
        for i, d in enumerate(self.demand):
            prefix[i + 1] = prefix[i] + d

        # DP tables
        C = np.full(n + 1, np.inf)
        split = np.full(n + 1, -1, dtype=int)
        C[0] = 0.0

        for j in range(1, n + 1):
            for i in range(j):
                # Holding cost: order placed at i to cover periods i..j-1
                holding = h * float(
                    sum(prefix[j] - prefix[k + 1] for k in range(i, j))
                )
                cost = C[i] + self.ordering_cost + holding
                if cost < C[j]:
                    C[j] = cost
                    split[j] = i

        # Backtrack to recover replenishment periods and quantities
        replenishments: Dict[int, float] = {}
        j = n
        while j > 0:
            i = int(split[j])
            replenishments[i] = float(prefix[j] - prefix[i])
            j = i

        total_cost, details = self.evaluate_plan(replenishments)
        return total_cost, details

    def optimal_cost(self) -> float:
        """Return only the optimal cost scalar (convenience wrapper)."""
        cost, _ = self.calculate_cost()
        return cost
