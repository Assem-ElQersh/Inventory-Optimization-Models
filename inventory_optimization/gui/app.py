"""
Inventory Optimization GUI
==========================
Three-tab Tkinter application with embedded Matplotlib figures.

Tab 1 — Single Model
    Run one model and view:
      • Period-by-period inventory trajectory plot
      • Cost breakdown bar chart (ordering vs. holding)
      • Detailed period table

Tab 2 — Model Comparison
    Run all deterministic models on the same inputs and display:
      • Side-by-side cost bar chart
      • Inventory trajectory overlay for all models
      • Summary table with total cost and % gap vs. Wagner-Whitin optimal

Tab 3 — Stochastic Model
    Parametrise a SafetyStockModel and visualise:
      • Simulated inventory trajectory with ROP and safety-stock lines
      • Cost breakdown (holding vs. ordering over simulation)
"""
from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

matplotlib.use("TkAgg")

from inventory_optimization.models import (
    EOQModel,
    EOQTimeSupplyModel,
    LeastUnitCostHeuristic,
    LotForLotModel,
    PartPeriodBalancingModel,
    SafetyStockModel,
    SilverMealHeuristic,
    ThreeMonthReplenishmentModel,
    WagnerWhitinModel,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DETERMINISTIC_MODELS: Dict[str, type] = {
    "EOQ Model": EOQModel,
    "Wagner-Whitin (Optimal)": WagnerWhitinModel,
    "Silver-Meal Heuristic": SilverMealHeuristic,
    "Least Unit Cost": LeastUnitCostHeuristic,
    "Lot-for-Lot": LotForLotModel,
    "Part-Period Balancing": PartPeriodBalancingModel,
    "EOQ Time Supply": EOQTimeSupplyModel,
    "Three-Month Replenishment": ThreeMonthReplenishmentModel,
}

DEFAULT_DEMAND = "200, 300, 180, 70, 50, 20, 10, 30, 90, 120, 150, 190"
DEFAULT_ORDERING = "160"
DEFAULT_UNIT = "5"
DEFAULT_CARRYING = "0.1"

COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


# ---------------------------------------------------------------------------
# Helper widgets
# ---------------------------------------------------------------------------


def _labeled_entry(parent, label: str, default: str, row: int) -> tk.Entry:
    tk.Label(parent, text=label, anchor="w").grid(
        row=row, column=0, sticky="w", padx=6, pady=3
    )
    entry = tk.Entry(parent, width=40)
    entry.insert(0, default)
    entry.grid(row=row, column=1, sticky="ew", padx=6, pady=3)
    return entry


def _make_treeview(parent, columns: List[str]) -> ttk.Treeview:
    frame = tk.Frame(parent)
    frame.pack(fill="both", expand=True, padx=6, pady=4)

    scroll_y = ttk.Scrollbar(frame, orient="vertical")
    scroll_x = ttk.Scrollbar(frame, orient="horizontal")

    tree = ttk.Treeview(
        frame,
        columns=columns,
        show="headings",
        yscrollcommand=scroll_y.set,
        xscrollcommand=scroll_x.set,
    )
    scroll_y.config(command=tree.yview)
    scroll_x.config(command=tree.xview)

    for col in columns:
        tree.heading(col, text=col)
        tree.column(col, width=110, anchor="center")

    scroll_y.pack(side="right", fill="y")
    scroll_x.pack(side="bottom", fill="x")
    tree.pack(fill="both", expand=True)
    return tree


def _embed_figure(fig: Figure, parent: tk.Widget) -> FigureCanvasTkAgg:
    canvas = FigureCanvasTkAgg(fig, master=parent)
    canvas.draw()
    toolbar = NavigationToolbar2Tk(canvas, parent)
    toolbar.update()
    canvas.get_tk_widget().pack(fill="both", expand=True)
    return canvas


# ---------------------------------------------------------------------------
# Tab 1 — Single Model
# ---------------------------------------------------------------------------


class SingleModelTab(tk.Frame):
    def __init__(self, parent: tk.Widget) -> None:
        super().__init__(parent)
        self._build_controls()
        self._build_output()

    def _build_controls(self) -> None:
        ctrl = tk.LabelFrame(self, text="Parameters", padx=6, pady=6)
        ctrl.pack(fill="x", padx=8, pady=6)
        ctrl.columnconfigure(1, weight=1)

        self._demand_e = _labeled_entry(ctrl, "Demand (comma-separated):", DEFAULT_DEMAND, 0)
        self._ordering_e = _labeled_entry(ctrl, "Ordering Cost (S):", DEFAULT_ORDERING, 1)
        self._unit_e = _labeled_entry(ctrl, "Unit Cost (c):", DEFAULT_UNIT, 2)
        self._carrying_e = _labeled_entry(ctrl, "Carrying Charge (h):", DEFAULT_CARRYING, 3)

        tk.Label(ctrl, text="Model:", anchor="w").grid(
            row=4, column=0, sticky="w", padx=6, pady=3
        )
        self._model_var = tk.StringVar(value="Wagner-Whitin (Optimal)")
        self._model_cb = ttk.Combobox(
            ctrl,
            textvariable=self._model_var,
            values=list(DETERMINISTIC_MODELS.keys()),
            state="readonly",
            width=38,
        )
        self._model_cb.grid(row=4, column=1, sticky="ew", padx=6, pady=3)

        tk.Button(ctrl, text="Run Model", command=self._run, bg="#2E86AB", fg="white", relief="flat", padx=10).grid(
            row=5, column=0, columnspan=2, pady=8
        )

    def _build_output(self) -> None:
        out = tk.PanedWindow(self, orient="vertical")
        out.pack(fill="both", expand=True, padx=8, pady=4)

        # Plot pane
        plot_frame = tk.Frame(out, height=320)
        out.add(plot_frame, stretch="always")
        self._fig, (self._ax_inv, self._ax_cost) = plt.subplots(
            1, 2, figsize=(10, 3.5), tight_layout=True
        )
        self._canvas = _embed_figure(self._fig, plot_frame)

        # Table pane
        table_frame = tk.Frame(out)
        out.add(table_frame, stretch="never")
        cols = ["Period", "Starting_Inventory", "Replenishment", "Requirements", "Ending_Inventory"]
        self._tree = _make_treeview(table_frame, cols)

        # Cost summary label
        self._summary_var = tk.StringVar(value="")
        tk.Label(self, textvariable=self._summary_var, font=("Helvetica", 11, "bold"), fg="#2E86AB").pack()

    # ------------------------------------------------------------------

    def _parse(self) -> Optional[Tuple]:
        try:
            demand = [float(x.strip()) for x in self._demand_e.get().split(",") if x.strip()]
            ordering = float(self._ordering_e.get())
            unit = float(self._unit_e.get())
            carrying = float(self._carrying_e.get())
            return demand, ordering, unit, carrying
        except ValueError as exc:
            messagebox.showerror("Input Error", str(exc))
            return None

    def _run(self) -> None:
        params = self._parse()
        if params is None:
            return
        demand, ordering, unit, carrying = params
        model_cls = DETERMINISTIC_MODELS[self._model_var.get()]
        try:
            model = model_cls(demand, ordering, unit, carrying)
            cost, details = model.calculate_cost()
        except Exception as exc:
            messagebox.showerror("Model Error", str(exc))
            return

        self._update_plots(demand, details, ordering, cost)
        self._update_table(details)
        self._summary_var.set(
            f"Total Cost: {cost:,.2f}   |   Orders: {sum(1 for d in details if d['Replenishment'] > 0)}"
        )

    def _update_plots(self, demand, details, ordering_cost, total_cost):
        self._ax_inv.cla()
        self._ax_cost.cla()

        periods = [d["Period"] for d in details]
        ending_inv = [d["Ending_Inventory"] for d in details]
        replenishments = [d["Replenishment"] for d in details]
        h_cost = sum(max(d["Ending_Inventory"], 0) for d in details) * ordering_cost  # approx
        actual_orders = sum(1 for r in replenishments if r > 0)
        o_cost = ordering_cost * actual_orders
        h_cost = total_cost - o_cost

        # Inventory trajectory
        self._ax_inv.bar(periods, replenishments, alpha=0.35, color="steelblue", label="Order")
        self._ax_inv.step(periods, ending_inv, where="post", color="navy", linewidth=1.8, label="Ending Inv.")
        self._ax_inv.axhline(0, color="red", linewidth=0.8, linestyle="--")
        self._ax_inv.fill_between(periods, 0, ending_inv, step="post", alpha=0.12, color="navy")
        self._ax_inv.set_xlabel("Period")
        self._ax_inv.set_ylabel("Units")
        self._ax_inv.set_title("Inventory Trajectory")
        self._ax_inv.legend(fontsize=8)
        self._ax_inv.set_xticks(periods)

        # Cost breakdown
        self._ax_cost.bar(["Ordering", "Holding"], [o_cost, h_cost], color=["#E84855", "#3A86FF"])
        self._ax_cost.set_ylabel("Cost ($)")
        self._ax_cost.set_title("Cost Breakdown")
        for ax_bar, val in zip(self._ax_cost.patches, [o_cost, h_cost]):
            self._ax_cost.text(
                ax_bar.get_x() + ax_bar.get_width() / 2,
                ax_bar.get_height() + total_cost * 0.01,
                f"{val:,.1f}",
                ha="center", va="bottom", fontsize=9,
            )

        self._fig.tight_layout()
        self._canvas.draw()

    def _update_table(self, details):
        for item in self._tree.get_children():
            self._tree.delete(item)
        for d in details:
            self._tree.insert(
                "",
                "end",
                values=(
                    d["Period"],
                    f"{d['Starting_Inventory']:.1f}",
                    f"{d['Replenishment']:.1f}",
                    f"{d['Requirements']:.1f}",
                    f"{d['Ending_Inventory']:.1f}",
                ),
            )


# ---------------------------------------------------------------------------
# Tab 2 — Model Comparison
# ---------------------------------------------------------------------------


class ComparisonTab(tk.Frame):
    def __init__(self, parent: tk.Widget) -> None:
        super().__init__(parent)
        self._build_controls()
        self._build_output()

    def _build_controls(self) -> None:
        ctrl = tk.LabelFrame(self, text="Parameters", padx=6, pady=6)
        ctrl.pack(fill="x", padx=8, pady=6)
        ctrl.columnconfigure(1, weight=1)

        self._demand_e = _labeled_entry(ctrl, "Demand (comma-separated):", DEFAULT_DEMAND, 0)
        self._ordering_e = _labeled_entry(ctrl, "Ordering Cost (S):", DEFAULT_ORDERING, 1)
        self._unit_e = _labeled_entry(ctrl, "Unit Cost (c):", DEFAULT_UNIT, 2)
        self._carrying_e = _labeled_entry(ctrl, "Carrying Charge (h):", DEFAULT_CARRYING, 3)

        tk.Button(
            ctrl,
            text="Compare All Models",
            command=self._run,
            bg="#2E86AB",
            fg="white",
            relief="flat",
            padx=10,
        ).grid(row=4, column=0, columnspan=2, pady=8)

    def _build_output(self) -> None:
        paned = tk.PanedWindow(self, orient="vertical")
        paned.pack(fill="both", expand=True, padx=8, pady=4)

        plot_frame = tk.Frame(paned, height=340)
        paned.add(plot_frame, stretch="always")
        self._fig, (self._ax_bar, self._ax_traj) = plt.subplots(
            1, 2, figsize=(12, 3.8), tight_layout=True
        )
        self._canvas = _embed_figure(self._fig, plot_frame)

        table_frame = tk.Frame(paned)
        paned.add(table_frame, stretch="never")
        self._tree = _make_treeview(
            table_frame, ["Model", "Total Cost", "# Orders", "Holding Cost", "Ordering Cost", "Gap vs WW (%)"]
        )

    def _parse(self) -> Optional[Tuple]:
        try:
            demand = [float(x.strip()) for x in self._demand_e.get().split(",") if x.strip()]
            ordering = float(self._ordering_e.get())
            unit = float(self._unit_e.get())
            carrying = float(self._carrying_e.get())
            return demand, ordering, unit, carrying
        except ValueError as exc:
            messagebox.showerror("Input Error", str(exc))
            return None

    def _run(self) -> None:
        params = self._parse()
        if params is None:
            return
        demand, ordering, unit, carrying = params

        results = {}
        errors = []
        for name, cls in DETERMINISTIC_MODELS.items():
            try:
                model = cls(demand, ordering, unit, carrying)
                cost, details = model.calculate_cost()
                n_orders = sum(1 for d in details if d["Replenishment"] > 0)
                o_cost = ordering * n_orders
                h_cost = cost - o_cost
                results[name] = {
                    "cost": cost,
                    "details": details,
                    "n_orders": n_orders,
                    "o_cost": o_cost,
                    "h_cost": h_cost,
                }
            except Exception as exc:
                errors.append(f"{name}: {exc}")

        if errors:
            messagebox.showwarning("Model Errors", "\n".join(errors))

        if not results:
            return

        ww_cost = results.get("Wagner-Whitin (Optimal)", {}).get("cost", None)
        self._update_plots(demand, results)
        self._update_table(results, ww_cost)

    def _update_plots(self, demand, results):
        self._ax_bar.cla()
        self._ax_traj.cla()

        names = list(results.keys())
        short_names = [n.replace(" Heuristic", "").replace(" Model", "").replace("-", "\n") for n in names]
        costs = [results[n]["cost"] for n in names]

        # Abbreviate further
        colors = COLORS[: len(names)]
        bars = self._ax_bar.bar(range(len(names)), costs, color=colors)
        self._ax_bar.set_xticks(range(len(names)))
        self._ax_bar.set_xticklabels(short_names, fontsize=7, rotation=30, ha="right")
        self._ax_bar.set_ylabel("Total Cost ($)")
        self._ax_bar.set_title("Cost Comparison")
        for bar, c in zip(bars, costs):
            self._ax_bar.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(costs) * 0.005,
                f"{c:,.0f}",
                ha="center", va="bottom", fontsize=7,
            )

        # Inventory trajectories
        for i, (name, res) in enumerate(results.items()):
            periods = [d["Period"] for d in res["details"]]
            ending = [d["Ending_Inventory"] for d in res["details"]]
            label = name.replace(" Heuristic", "").replace(" Model", "")
            lw = 2.2 if "Wagner" in name else 1.0
            ls = "-" if "Wagner" in name else "--"
            self._ax_traj.step(periods, ending, where="post", label=label,
                               color=colors[i], linewidth=lw, linestyle=ls)

        self._ax_traj.axhline(0, color="black", linewidth=0.6, linestyle=":")
        self._ax_traj.set_xlabel("Period")
        self._ax_traj.set_ylabel("Ending Inventory (units)")
        self._ax_traj.set_title("Inventory Trajectories")
        self._ax_traj.legend(fontsize=6, loc="upper right")

        self._fig.tight_layout()
        self._canvas.draw()

    def _update_table(self, results, ww_cost):
        for item in self._tree.get_children():
            self._tree.delete(item)
        for name, r in results.items():
            gap = ""
            if ww_cost is not None and ww_cost > 0:
                gap = f"{(r['cost'] - ww_cost) / ww_cost * 100:.1f}%"
            self._tree.insert(
                "",
                "end",
                values=(
                    name,
                    f"{r['cost']:,.2f}",
                    r["n_orders"],
                    f"{r['h_cost']:,.2f}",
                    f"{r['o_cost']:,.2f}",
                    gap,
                ),
            )


# ---------------------------------------------------------------------------
# Tab 3 — Stochastic Model
# ---------------------------------------------------------------------------


class StochasticTab(tk.Frame):
    def __init__(self, parent: tk.Widget) -> None:
        super().__init__(parent)
        self._build_controls()
        self._build_output()

    def _build_controls(self) -> None:
        ctrl = tk.LabelFrame(self, text="Parameters", padx=6, pady=6)
        ctrl.pack(fill="x", padx=8, pady=6)
        ctrl.columnconfigure(1, weight=1)

        self._mean_e = _labeled_entry(ctrl, "Mean Demand (μ per period):", "100", 0)
        self._std_e = _labeled_entry(ctrl, "Std Dev Demand (σ):", "20", 1)
        self._ordering_e = _labeled_entry(ctrl, "Ordering Cost (S):", "160", 2)
        self._unit_e = _labeled_entry(ctrl, "Unit Cost (c):", "5", 3)
        self._carrying_e = _labeled_entry(ctrl, "Carrying Charge (h):", "0.1", 4)
        self._lt_e = _labeled_entry(ctrl, "Lead Time (periods):", "2", 5)
        self._sl_e = _labeled_entry(ctrl, "Service Level (0–1):", "0.95", 6)
        self._n_e = _labeled_entry(ctrl, "Simulation Periods:", "52", 7)

        tk.Button(
            ctrl,
            text="Run Stochastic Model",
            command=self._run,
            bg="#2E86AB",
            fg="white",
            relief="flat",
            padx=10,
        ).grid(row=8, column=0, columnspan=2, pady=8)

    def _build_output(self) -> None:
        paned = tk.PanedWindow(self, orient="vertical")
        paned.pack(fill="both", expand=True, padx=8, pady=4)

        plot_frame = tk.Frame(paned, height=320)
        paned.add(plot_frame, stretch="always")
        self._fig, (self._ax_inv, self._ax_cost) = plt.subplots(
            1, 2, figsize=(12, 3.5), tight_layout=True
        )
        self._canvas = _embed_figure(self._fig, plot_frame)

        self._summary_var = tk.StringVar(value="")
        tk.Label(self, textvariable=self._summary_var, font=("Helvetica", 10), fg="#333").pack(pady=4)

    def _run(self) -> None:
        try:
            mu = float(self._mean_e.get())
            sigma = float(self._std_e.get())
            ordering = float(self._ordering_e.get())
            unit = float(self._unit_e.get())
            carrying = float(self._carrying_e.get())
            lead_time = int(self._lt_e.get())
            sl = float(self._sl_e.get())
            n = int(self._n_e.get())
        except ValueError as exc:
            messagebox.showerror("Input Error", str(exc))
            return

        try:
            model = SafetyStockModel(
                mean_demand=mu,
                std_demand=sigma,
                ordering_cost=ordering,
                unit_cost=unit,
                carrying_charge=carrying,
                lead_time=lead_time,
                service_level=sl,
                n_periods=n,
            )
            result = model.solve(seed=42)
        except Exception as exc:
            messagebox.showerror("Model Error", str(exc))
            return

        self._update_plots(result)
        self._summary_var.set(
            f"EOQ: {result.eoq:.1f}   |   Safety Stock: {result.safety_stock:.1f}"
            f"   |   ROP: {result.reorder_point:.1f}"
            f"   |   Expected Total Cost: {result.expected_total_cost:,.2f}"
        )

    def _update_plots(self, result) -> None:
        self._ax_inv.cla()
        self._ax_cost.cla()

        details = result.simulation_details
        periods = [d["Period"] for d in details]
        inv = [d["Ending_Inventory"] for d in details]
        orders = [d["Order_Placed"] for d in details]

        self._ax_inv.plot(periods, inv, color="navy", linewidth=1.4, label="Inventory")
        self._ax_inv.bar(periods, orders, alpha=0.3, color="steelblue", label="Order")
        self._ax_inv.axhline(result.reorder_point, color="red", linewidth=1.2, linestyle="--", label=f"ROP={result.reorder_point:.0f}")
        self._ax_inv.axhline(result.safety_stock, color="orange", linewidth=1.0, linestyle=":", label=f"SS={result.safety_stock:.0f}")
        self._ax_inv.set_xlabel("Period")
        self._ax_inv.set_ylabel("Units")
        self._ax_inv.set_title("Simulated Inventory (Safety Stock Policy)")
        self._ax_inv.legend(fontsize=8)

        # Cumulative cost breakdown
        n_orders = sum(1 for d in details if d["Order_Placed"] > 0)
        h = result.expected_total_cost - result.expected_total_cost * (n_orders / max(len(details), 1))
        o_cost_total = result.expected_total_cost - h
        self._ax_cost.bar(["Ordering", "Holding"], [o_cost_total, h], color=["#E84855", "#3A86FF"])
        self._ax_cost.set_title("Cost Breakdown (Simulated)")
        self._ax_cost.set_ylabel("Cost ($)")

        self._fig.tight_layout()
        self._canvas.draw()


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------


class InventoryApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Inventory Optimization Models")
        self.geometry("1100x780")
        self.resizable(True, True)

        style = ttk.Style(self)
        style.theme_use("clam")

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=6, pady=6)

        tab1 = SingleModelTab(nb)
        tab2 = ComparisonTab(nb)
        tab3 = StochasticTab(nb)

        nb.add(tab1, text="  Single Model  ")
        nb.add(tab2, text="  Model Comparison  ")
        nb.add(tab3, text="  Stochastic Model  ")


def main() -> None:
    app = InventoryApp()
    app.mainloop()


if __name__ == "__main__":
    main()
