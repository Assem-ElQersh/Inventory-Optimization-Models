"""
inventory_optimization
======================
Classical and learning-based inventory control models.

Quick start
-----------
>>> from inventory_optimization.models import WagnerWhitinModel
>>> demand = [200, 300, 180, 70, 50, 20, 10, 30, 90, 120, 150, 190]
>>> model = WagnerWhitinModel(demand, ordering_cost=160, unit_cost=5, carrying_charge=0.1)
>>> cost, details = model.calculate_cost()
>>> print(f"Optimal cost: {cost:.2f}")
"""
from inventory_optimization.models import (  # noqa: F401
    EOQModel,
    EOQTimeSupplyModel,
    InventoryModel,
    LeastUnitCostHeuristic,
    LotForLotModel,
    NewsvendorModel,
    PartPeriodBalancingModel,
    SafetyStockModel,
    SilverMealHeuristic,
    ThreeMonthReplenishmentModel,
    WagnerWhitinModel,
    validate_inputs,
)

__version__ = "1.0.0"
__all__ = [
    "InventoryModel",
    "validate_inputs",
    "EOQModel",
    "EOQTimeSupplyModel",
    "WagnerWhitinModel",
    "SilverMealHeuristic",
    "LeastUnitCostHeuristic",
    "LotForLotModel",
    "PartPeriodBalancingModel",
    "ThreeMonthReplenishmentModel",
    "NewsvendorModel",
    "SafetyStockModel",
]
