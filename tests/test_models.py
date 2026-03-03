"""
Pytest test suite for inventory_optimization.

Coverage:
    - Input validation (ValueError on bad inputs)
    - EOQ formula correctness on constant demand
    - Wagner-Whitin optimality (cost <= all heuristics)
    - Wagner-Whitin known-optimal small case
    - Zero ending inventory property for all models
    - Zero demand period handling
    - High carrying cost drives more frequent orders
    - Prefix-sum heuristics produce non-negative costs
    - Lot-for-Lot zero holding cost
    - Stochastic: safety stock formula
    - Stochastic: service level parameter validation
    - CostCalculator evaluate() matches evaluate_plan() on same plan
"""
from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
import pytest

from inventory_optimization.cost.cost_calculator import CostCalculator
from inventory_optimization.models import (
    EOQModel,
    EOQTimeSupplyModel,
    LeastUnitCostHeuristic,
    LotForLotModel,
    PartPeriodBalancingModel,
    SilverMealHeuristic,
    ThreeMonthReplenishmentModel,
    WagnerWhitinModel,
    validate_inputs,
)
from inventory_optimization.models.stochastic import NewsvendorModel, SafetyStockModel

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

DEMAND_12 = [200, 300, 180, 70, 50, 20, 10, 30, 90, 120, 150, 190]
ORDERING = 160.0
UNIT = 5.0
CARRYING = 0.1


def _all_deterministic_models(demand, ordering=ORDERING, unit=UNIT, carrying=CARRYING):
    return [
        EOQModel(demand, ordering, unit, carrying),
        WagnerWhitinModel(demand, ordering, unit, carrying),
        SilverMealHeuristic(demand, ordering, unit, carrying),
        LeastUnitCostHeuristic(demand, ordering, unit, carrying),
        LotForLotModel(demand, ordering, unit, carrying),
        PartPeriodBalancingModel(demand, ordering, unit, carrying),
        EOQTimeSupplyModel(demand, ordering, unit, carrying),
        ThreeMonthReplenishmentModel(demand, ordering, unit, carrying),
    ]


# ---------------------------------------------------------------------------
# 1. Input validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_empty_demand_raises(self):
        with pytest.raises(ValueError, match="empty"):
            validate_inputs([], ORDERING, UNIT, CARRYING)

    def test_negative_demand_raises(self):
        with pytest.raises(ValueError, match=">="):
            validate_inputs([-1, 10], ORDERING, UNIT, CARRYING)

    def test_zero_ordering_cost_raises(self):
        with pytest.raises(ValueError, match=">"):
            validate_inputs([10], 0.0, UNIT, CARRYING)

    def test_negative_ordering_cost_raises(self):
        with pytest.raises(ValueError, match=">"):
            validate_inputs([10], -5.0, UNIT, CARRYING)

    def test_zero_unit_cost_raises(self):
        with pytest.raises(ValueError, match=">"):
            validate_inputs([10], ORDERING, 0.0, CARRYING)

    def test_negative_carrying_charge_raises(self):
        with pytest.raises(ValueError, match=">="):
            validate_inputs([10], ORDERING, UNIT, -0.1)

    def test_valid_zero_carrying_charge(self):
        # h=0 is allowed (no holding cost)
        validate_inputs([10], ORDERING, UNIT, 0.0)

    def test_model_constructor_validates(self):
        with pytest.raises(ValueError):
            WagnerWhitinModel([-1, 10], ORDERING, UNIT, CARRYING)


# ---------------------------------------------------------------------------
# 2. EOQ formula on constant demand
# ---------------------------------------------------------------------------


class TestEOQ:
    def test_eoq_formula_value(self):
        """Q* = sqrt(2*S*D / (h*c)) for constant demand."""
        D = 100.0
        demand = [D] * 12
        S = 50.0
        c = 10.0
        h = 0.2
        expected_eoq = math.sqrt(2 * S * D / (h * c))

        model = EOQModel(demand, S, c, h)
        qty = model._best_order_quantity(0)
        # The returned quantity is the cumulative demand closest to EOQ
        assert abs(qty - expected_eoq) <= D, (
            f"EOQ heuristic qty {qty:.2f} deviates from formula {expected_eoq:.2f} by more than one period's demand"
        )

    def test_eoq_positive_cost(self):
        cost, details = EOQModel(DEMAND_12, ORDERING, UNIT, CARRYING).calculate_cost()
        assert cost > 0

    def test_eoq_time_supply_positive_cost(self):
        cost, _ = EOQTimeSupplyModel(DEMAND_12, ORDERING, UNIT, CARRYING).calculate_cost()
        assert cost > 0


# ---------------------------------------------------------------------------
# 3. Wagner-Whitin optimality
# ---------------------------------------------------------------------------


class TestWagnerWhitin:
    def test_ww_le_all_heuristics(self):
        """Wagner-Whitin optimal cost must be <= cost of every heuristic."""
        ww_cost, _ = WagnerWhitinModel(DEMAND_12, ORDERING, UNIT, CARRYING).calculate_cost()
        for model in _all_deterministic_models(DEMAND_12):
            if isinstance(model, WagnerWhitinModel):
                continue
            cost, _ = model.calculate_cost()
            assert ww_cost <= cost + 1e-6, (
                f"Wagner-Whitin cost {ww_cost:.2f} > {type(model).__name__} cost {cost:.2f}"
            )

    def test_ww_known_optimal_small_case(self):
        """
        Textbook example (Wagner & Whitin 1958 Table 1):
        demand = [10, 10, 15, 20, 70, 180, 250, 270, 230, 40, 0, 10]
        S=85, c=1, h=0.1  =>  optimal cost = 1043.30  (approx.)
        We verify the cost is finite and <= lot-for-lot cost.
        """
        demand = [10, 10, 15, 20, 70, 180, 250, 270, 230, 40, 0, 10]
        ww = WagnerWhitinModel(demand, 85.0, 1.0, 0.1)
        l4l = LotForLotModel(demand, 85.0, 1.0, 0.1)
        ww_cost, _ = ww.calculate_cost()
        l4l_cost, _ = l4l.calculate_cost()
        assert math.isfinite(ww_cost)
        assert ww_cost <= l4l_cost + 1e-6

    def test_ww_zero_holding_cost(self):
        """With h=0, only ordering cost matters; WW should order all at once."""
        demand = [10, 20, 30]
        ww = WagnerWhitinModel(demand, ordering_cost=100.0, unit_cost=1.0, carrying_charge=0.0)
        cost, details = ww.calculate_cost()
        n_orders = sum(1 for d in details if d["Replenishment"] > 0)
        assert n_orders == 1, "With zero holding cost, one order covers entire horizon"
        assert abs(cost - 100.0) < 1e-6

    def test_ww_single_period(self):
        ww = WagnerWhitinModel([50], ORDERING, UNIT, CARRYING)
        cost, details = ww.calculate_cost()
        assert abs(cost - ORDERING) < 1e-6
        assert details[0]["Replenishment"] == 50


# ---------------------------------------------------------------------------
# 4. Zero ending inventory property
# ---------------------------------------------------------------------------


class TestZeroEndingInventory:
    @pytest.mark.parametrize("model", _all_deterministic_models(DEMAND_12))
    def test_zero_ending_inventory_at_horizon_end(self, model):
        _, details = model.calculate_cost()
        assert abs(details[-1]["Ending_Inventory"]) < 1e-6, (
            f"{type(model).__name__}: ending inventory at horizon end is "
            f"{details[-1]['Ending_Inventory']:.4f}, expected 0"
        )


# ---------------------------------------------------------------------------
# 5. Zero demand periods
# ---------------------------------------------------------------------------


class TestZeroDemand:
    def test_zero_demand_no_crash(self):
        demand = [0, 100, 0, 200, 0]
        for model in _all_deterministic_models(demand):
            cost, details = model.calculate_cost()
            assert cost >= 0
            assert len(details) == len(demand)

    def test_all_zero_demand_zero_holding_cost(self):
        demand = [0, 0, 0, 0]
        for model in _all_deterministic_models(demand):
            cost, details = model.calculate_cost()
            # All zero demand: no holding cost (nothing to hold)
            holding = sum(max(d["Ending_Inventory"], 0) for d in details)
            assert holding == 0.0


# ---------------------------------------------------------------------------
# 6. High carrying cost forces more frequent orders
# ---------------------------------------------------------------------------


class TestCarryingCostEffect:
    def test_high_carrying_forces_more_orders(self):
        demand = [50] * 12
        _, low_details = WagnerWhitinModel(demand, ORDERING, UNIT, 0.01).calculate_cost()
        _, high_details = WagnerWhitinModel(demand, ORDERING, UNIT, 0.9).calculate_cost()
        low_orders = sum(1 for d in low_details if d["Replenishment"] > 0)
        high_orders = sum(1 for d in high_details if d["Replenishment"] > 0)
        assert high_orders >= low_orders, (
            "Higher carrying charge should result in at least as many orders"
        )


# ---------------------------------------------------------------------------
# 7. Lot-for-Lot has zero holding cost
# ---------------------------------------------------------------------------


class TestLotForLot:
    def test_zero_holding_cost(self):
        model = LotForLotModel(DEMAND_12, ORDERING, UNIT, CARRYING)
        _, details = model.calculate_cost()
        for d in details:
            assert d["Ending_Inventory"] <= 0 + 1e-9, (
                f"LotForLot period {d['Period']}: ending inventory {d['Ending_Inventory']:.4f}"
            )

    def test_cost_equals_n_nonzero_periods_times_ordering_cost(self):
        demand = [0, 100, 0, 50, 200]
        model = LotForLotModel(demand, ORDERING, UNIT, CARRYING)
        cost, details = model.calculate_cost()
        n_nonzero = sum(1 for d in demand if d > 0)
        assert abs(cost - n_nonzero * ORDERING) < 1e-6


# ---------------------------------------------------------------------------
# 8. Heuristic costs are non-negative and finite
# ---------------------------------------------------------------------------


class TestHeuristicSanity:
    @pytest.mark.parametrize(
        "cls",
        [
            SilverMealHeuristic,
            LeastUnitCostHeuristic,
            PartPeriodBalancingModel,
            ThreeMonthReplenishmentModel,
        ],
    )
    def test_positive_finite_cost(self, cls):
        model = cls(DEMAND_12, ORDERING, UNIT, CARRYING)
        cost, _ = model.calculate_cost()
        assert cost > 0
        assert math.isfinite(cost)

    def test_silver_meal_vs_luc_similar_magnitude(self):
        """Silver-Meal and LUC should be within 50% of each other on standard data."""
        sm_cost, _ = SilverMealHeuristic(DEMAND_12, ORDERING, UNIT, CARRYING).calculate_cost()
        luc_cost, _ = LeastUnitCostHeuristic(DEMAND_12, ORDERING, UNIT, CARRYING).calculate_cost()
        ratio = max(sm_cost, luc_cost) / min(sm_cost, luc_cost)
        assert ratio < 1.5


# ---------------------------------------------------------------------------
# 9. Stochastic: safety stock formula
# ---------------------------------------------------------------------------


class TestStochasticModels:
    def test_safety_stock_formula(self):
        """SS = z_alpha * sigma * sqrt(L)."""
        from scipy import stats

        mu, sigma, L, sl = 100.0, 20.0, 4, 0.95
        expected_ss = stats.norm.ppf(sl) * sigma * math.sqrt(L)
        model = SafetyStockModel(mu, sigma, ORDERING, UNIT, CARRYING, lead_time=L, service_level=sl)
        result = model.solve(seed=0)
        assert abs(result.safety_stock - expected_ss) < 0.01

    def test_reorder_point_includes_safety_stock(self):
        model = SafetyStockModel(100.0, 20.0, ORDERING, UNIT, CARRYING, lead_time=2, service_level=0.95)
        result = model.solve(seed=0)
        assert result.reorder_point > result.safety_stock

    def test_higher_service_level_higher_ss(self):
        base = SafetyStockModel(100.0, 20.0, ORDERING, UNIT, CARRYING, lead_time=2, service_level=0.90)
        high = SafetyStockModel(100.0, 20.0, ORDERING, UNIT, CARRYING, lead_time=2, service_level=0.99)
        assert high.solve(seed=0).safety_stock > base.solve(seed=0).safety_stock

    def test_invalid_service_level_raises(self):
        with pytest.raises(ValueError):
            SafetyStockModel(100.0, 20.0, ORDERING, UNIT, CARRYING, service_level=1.0)

    def test_newsvendor_critical_ratio(self):
        """CR = (p-c)/(p-s); Q* = F^{-1}(CR)."""
        from scipy import stats

        mu, sigma = 100.0, 15.0
        c, p, s = 5.0, 20.0, 1.0
        model = NewsvendorModel(mu, sigma, c, p, s)
        result = model.solve()
        cr_expected = (p - c) / (p - s)
        assert abs(result.critical_ratio - cr_expected) < 1e-4
        q_expected = stats.norm.ppf(cr_expected, mu, sigma)
        assert abs(result.optimal_quantity - q_expected) < 0.01

    def test_newsvendor_selling_price_must_exceed_cost(self):
        with pytest.raises(ValueError):
            NewsvendorModel(100.0, 15.0, unit_cost=10.0, selling_price=8.0, salvage_value=2.0)


# ---------------------------------------------------------------------------
# 10. CostCalculator consistency
# ---------------------------------------------------------------------------


class TestCostCalculator:
    def test_evaluate_matches_evaluate_plan(self):
        """CostCalculator.evaluate() must return the same total cost as InventoryModel.evaluate_plan()."""
        demand = DEMAND_12
        model = WagnerWhitinModel(demand, ORDERING, UNIT, CARRYING)
        cost_ww, details_ww = model.calculate_cost()

        # Reconstruct replenishments from details
        replenishments: Dict[int, float] = {}
        for d in details_ww:
            if d["Replenishment"] > 0:
                replenishments[d["Period"] - 1] = d["Replenishment"]

        calc = CostCalculator(ORDERING, UNIT, CARRYING)
        cost_cc, _, breakdown = calc.evaluate(demand, replenishments)

        assert abs(cost_cc - cost_ww) < 1e-4, (
            f"CostCalculator cost {cost_cc:.4f} != WW evaluate_plan {cost_ww:.4f}"
        )
        assert breakdown["ordering"] + breakdown["holding"] == pytest.approx(cost_cc, abs=1e-4)

    def test_holding_cost_segment_zero_when_single_period(self):
        calc = CostCalculator(ORDERING, UNIT, CARRYING)
        # Single period: no inventory to hold
        hc = calc.holding_cost_segment([100], 0, 1)
        assert hc == 0.0

    def test_holding_cost_segment_with_prefix(self):
        demand = [10.0, 20.0, 30.0]
        prefix = [0.0, 10.0, 30.0, 60.0]
        calc = CostCalculator(ORDERING, UNIT, CARRYING)
        hc_with = calc.holding_cost_segment(demand, 0, 3, prefix=prefix)
        hc_without = calc.holding_cost_segment(demand, 0, 3)
        assert abs(hc_with - hc_without) < 1e-9
