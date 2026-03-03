# Inventory Optimization Models

A research-grade Python library for deterministic and stochastic inventory optimization, combining classical Operations Research algorithms with a learning-based Reinforcement Learning agent.

---

## Architecture

```
inventory_optimization/
├── models/
│   ├── base.py           # InventoryModel base class + validate_inputs()
│   ├── eoq.py            # EOQModel, EOQTimeSupplyModel
│   ├── wagner_whitin.py  # WagnerWhitinModel — O(n²) DP (globally optimal)
│   ├── heuristics.py     # SilverMeal, LeastUnitCost, LotForLot, PartPeriod, ThreeMonth
│   └── stochastic.py     # NewsvendorModel, SafetyStockModel
├── cost/
│   └── cost_calculator.py  # Standalone cost evaluator (decoupled from models)
├── gui/
│   └── app.py            # 3-tab Tkinter GUI with embedded Matplotlib plots
├── rl/
│   ├── env.py            # InventoryEnv (Gymnasium-compatible)
│   └── agent.py          # PPO training + classical vs. RL comparison dashboard
└── tests/
    └── test_models.py    # Full pytest suite
```

---

## Installation

### Core (deterministic models + GUI)

```bash
pip install -e .
```

### With RL support

```bash
pip install -e ".[rl]"
```

### Full development environment

```bash
pip install -e ".[all]"
```

### Launch the GUI

```bash
inventory-gui
```

---

## Models

### Deterministic Models

| Model | Type | Optimal? | Time Complexity |
|---|---|---|---|
| EOQ | Analytical heuristic | No (stationary assumption) | O(n²) |
| Wagner-Whitin | Dynamic programming | **Yes** | O(n²) |
| Silver-Meal | Heuristic | No | O(n²) |
| Least Unit Cost | Heuristic | No | O(n²) |
| Part-Period Balancing | Heuristic | No | O(n²) |
| Lot-for-Lot | Policy | No (optimal when S=0) | O(n) |
| EOQ Time Supply | Heuristic | No | O(n) |
| Three-Month Replenishment | Fixed-interval | No | O(n) |

### Stochastic Models

| Model | Type | Description |
|---|---|---|
| Newsvendor | Single-period | Optimal order quantity under uncertain demand (critical ratio) |
| Safety Stock | (R, Q) policy | Continuous review with safety stock targeting a cycle service level |

### RL Agent

| Agent | Framework | Description |
|---|---|---|
| PPO | stable-baselines3 | Proximal Policy Optimization on stochastic inventory episodes |

---

## Mathematical Formulations

### Economic Order Quantity (EOQ)

For **constant** demand $D$ per period, ordering cost $S$, unit cost $c$, and carrying charge $h$:

$$Q^* = \sqrt{\frac{2 S D}{h \cdot c}}$$

> **Note:** Applying EOQ to time-varying demand is a heuristic adaptation. The formula is stationary by assumption. This library makes that explicit.

### Wagner-Whitin DP Recurrence

For a horizon of $n$ periods with per-period demand $d_1, d_2, \ldots, d_n$:

With prefix sums $P[i] = \sum_{t=1}^{i} d_t$, 
the inner sum becomes $P[j-1] - P[k]$, 
computable in $O(1)$.

where $H(i, j)$ is the holding cost when ordering $\sum_{k=i}^{j-1} d_k$ at period $i$ to cover periods $i$ through $j-1$:

$$H(i, j) = h \cdot c \cdot \sum_{k=i}^{j-1} \left( \sum_{t=k+1}^{j-1} d_t \right)$$

With prefix sums $P[i] = \sum_{t=0}^{i-1} d_t$, the inner sum becomes $P[j] - P[k+1]$, computable in $O(1)$, giving overall $O(n^2)$ complexity.

### Silver-Meal Stopping Criterion

Extend the current order to cover $T$ periods while:

$$\frac{S + C_{\text{carry}}(T)}{T} \le \frac{S + C_{\text{carry}}(T-1)}{T-1}$$

where $C_{\text{carry}}(T) = h \cdot c \cdot \sum_{t=1}^{T-1} d_t \cdot t$.

Stop as soon as the average cost per period increases.

### Least Unit Cost Stopping Criterion

Extend while:

$$\frac{S + C_{\text{carry}}(T)}{\sum_{t=0}^{T-1} d_t} \le \frac{S + C_{\text{carry}}(T-1)}{\sum_{t=0}^{T-2} d_t}$$

### Safety Stock (Continuous Review)

For normally distributed per-period demand $D \sim \mathcal{N}(\mu, \sigma^2)$ with lead time $L$ and target cycle service level $\alpha$:

$$SS = z_\alpha \cdot \sigma \cdot \sqrt{L}$$

$$ROP = \mu \cdot L + SS$$

$$Q^* = \sqrt{\frac{2 S \mu}{h \cdot c}}$$

where $z_\alpha = \Phi^{-1}(\alpha)$ is the standard normal inverse CDF.

### Newsvendor Critical Ratio

Single-period model with overage cost $c_o = c - s$ and underage cost $c_u = p - c$:

$$Q^* = F^{-1}\!\left(\frac{c_u}{c_u + c_o}\right)$$

---

## Quick Start

### Run a single model

```python
from inventory_optimization import WagnerWhitinModel

demand = [200, 300, 180, 70, 50, 20, 10, 30, 90, 120, 150, 190]
model = WagnerWhitinModel(demand, ordering_cost=160, unit_cost=5, carrying_charge=0.1)
cost, details = model.calculate_cost()

print(f"Optimal total cost: {cost:.2f}")
for row in details:
    print(row)
```

### Compare all models

```python
from inventory_optimization import (
    WagnerWhitinModel, SilverMealHeuristic, LeastUnitCostHeuristic,
    LotForLotModel, EOQModel
)

demand = [200, 300, 180, 70, 50, 20, 10, 30, 90, 120, 150, 190]
params = dict(ordering_cost=160, unit_cost=5, carrying_charge=0.1)

models = {
    "Wagner-Whitin": WagnerWhitinModel(demand, **params),
    "Silver-Meal":   SilverMealHeuristic(demand, **params),
    "Least Unit Cost": LeastUnitCostHeuristic(demand, **params),
    "Lot-for-Lot":   LotForLotModel(demand, **params),
    "EOQ":           EOQModel(demand, **params),
}

ww_cost, _ = models["Wagner-Whitin"].calculate_cost()
for name, model in models.items():
    cost, _ = model.calculate_cost()
    gap = (cost - ww_cost) / ww_cost * 100
    print(f"{name:<22}  cost={cost:8.2f}  gap={gap:+.1f}%")
```

### Stochastic model

```python
from inventory_optimization import SafetyStockModel

model = SafetyStockModel(
    mean_demand=100, std_demand=20,
    ordering_cost=160, unit_cost=5, carrying_charge=0.1,
    lead_time=2, service_level=0.95, n_periods=52,
)
result = model.solve(seed=42)
print(f"Safety stock: {result.safety_stock:.1f}")
print(f"Reorder point: {result.reorder_point:.1f}")
print(f"EOQ: {result.eoq:.1f}")
```

### RL comparison (requires `pip install -e "[rl]"`)

```python
from inventory_optimization.rl.agent import run_comparison

df, fig = run_comparison(
    demand_sequence=[200, 300, 180, 70, 50, 20, 10, 30, 90, 120, 150, 190],
    timesteps=50_000,
)
print(df.to_string(index=False))
fig.savefig("rl_comparison.png")
```

---

## Running Tests

```bash
pytest                        # run all tests
pytest -v                     # verbose output
pytest --cov=inventory_optimization  # with coverage
```

---

## Design Constraints

All deterministic models share the same assumptions (explicitly documented):

- **Zero lead time** — orders arrive instantly.
- **No backorders or lost sales** — demand is always fulfilled.
- **Zero ending inventory enforced** — the final replenishment is trimmed to prevent excess.
- **Purchase cost excluded** — treated as sunk; only ordering and holding costs are optimised.

These constraints are intentional for classical lot-sizing. The stochastic and RL models relax them (safety stock, backorder penalties, lead time).

---

## Research Extension: Classical OR vs. RL

The `inventory_optimization.rl` module enables the experiment:

> *"Does a PPO agent learn a policy competitive with the Wagner-Whitin optimal under stochastic demand?"*

The comparison framework (`run_comparison`) evaluates each method on the same demand episodes, reports the **optimality gap** (% above WW cost), and generates publication-quality plots.

```
Method                     Mean Cost   Gap vs WW (%)
Wagner-Whitin (Optimal)      1234.56        0.0%
Silver-Meal                  1289.34       +4.4%
Lot-for-Lot                  2560.00     +107.4%
PPO                          1310.00       +6.1%  (after 50k steps)
```

---

## License

MIT License. See the [LICENSE](LICENSE) file for details.
