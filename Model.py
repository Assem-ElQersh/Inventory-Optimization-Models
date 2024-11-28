import json
import math
import tkinter as tk
from collections import defaultdict
from tkinter import messagebox, ttk

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class InventoryModel:
    """Base class for inventory models with time-varying demand."""

    def __init__(self, demand, ordering_cost, unit_cost, carrying_charge):
        self.demand = demand  # List of demand for each period
        self.ordering_cost = ordering_cost
        self.unit_cost = unit_cost
        self.carrying_charge = carrying_charge

    def calculate_cost(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def calculate_total_cost(self, replenishments):
        """Calculate total cost for a given replenishment plan and store period details."""
        total_cost = 0
        inventory = 0  # Current inventory level
        carrying_cost = 0
        details = []  # To store details for each period

        for period, demand in enumerate(self.demand, start=1):
            starting_inventory = inventory
            replenishment = replenishments.get(period - 1, 0)
            inventory += replenishment  # New inventory arrives

            # Ending inventory after fulfilling demand
            ending_inventory = inventory - demand

            # Calculate carrying cost using ending inventory
            carrying_cost += max(ending_inventory, 0) * self.carrying_charge * self.unit_cost

            # Store details for this period
            details.append({
                "Month": period,
                "Starting Inventory": starting_inventory,
                "Replenishment": replenishment,
                "Requirements": demand,
                "Ending Inventory": ending_inventory
            })

            # Update inventory for the next period
            inventory = ending_inventory

        # Ensure zero-ending inventory at the end of the horizon
        if inventory != 0:
            details[-1]["Replenishment"] += inventory
            details[-1]["Ending Inventory"] = 0

        # Calculate total replenishment and carrying costs, excluding purchase cost
        total_ordering_cost = self.ordering_cost * len(replenishments)
        total_cost_excluding_purchase = carrying_cost + total_ordering_cost

        return total_cost_excluding_purchase, details

    def save_results_to_json(self, file_name, results):
        with open(file_name, 'w') as json_file:
            json.dump(results, json_file, indent=4)


class EOQModel(InventoryModel):
    """Fixed EOQ model for time-varying demand."""

    def calculate_order_quantity(self, start_period):
        # Calculate EOQ based on average demand
        avg_demand = sum(self.demand) / len(self.demand)
        eoq = math.sqrt((2 * self.ordering_cost * avg_demand) / (self.carrying_charge * self.unit_cost))

        # Calculate cumulative demand to cover zero ending inventory
        cumulative_demand = 0
        period = start_period
        best_order_quantity = eoq  # Start with EOQ as the best option
        closest_difference = float('inf')

        while period < len(self.demand):
            cumulative_demand += self.demand[period]
            difference = abs(cumulative_demand - eoq)
            if difference < closest_difference:
                closest_difference = difference
                best_order_quantity = cumulative_demand
            period += 1

        return best_order_quantity

    def calculate_cost(self):
        replenishments = {}
        inventory = 0
        period = 0
        while period < len(self.demand):
            if inventory < self.demand[period]:
                # Calculate the order quantity closest to EOQ, covering demand to zero ending inventory
                order_quantity = self.calculate_order_quantity(period)
                replenishments[period] = order_quantity
                inventory += order_quantity
            # Deduct demand for this period
            inventory -= self.demand[period]
            period += 1
        # Calculate total cost based on the generated replenishments
        return self.calculate_total_cost(replenishments)


class WagnerWhitinModel(InventoryModel):
    """Optimal lot-sizing using the Wagner-Whitin dynamic programming approach."""

    def calculate_cost(self):
        n = len(self.demand)
        costs = [float('inf')] * (n + 1)
        costs[0] = 0
        orders = defaultdict(dict)

        for j in range(1, n + 1):
            min_cost = float('inf')
            best_order = None
            for i in range(j):
                # Set order quantity to exactly cover demand from i to j, making ending inventory zero
                order_quantity = sum(self.demand[i:j])
                replenishments = orders[i].copy()
                replenishments[i] = order_quantity
                try:
                    cost, _ = self.calculate_total_cost(replenishments)
                    if cost < min_cost:
                        min_cost = cost
                        best_order = replenishments
                except ValueError:
                    continue
            costs[j] = min_cost
            orders[j] = best_order if best_order is not None else {}

        # Ensure zero-ending inventory at horizon end
        if orders[n]:
            last_period = max(orders[n].keys())
            orders[n][last_period] = sum(self.demand[last_period:])

        return costs[-1],self.calculate_total_cost(replenishments)

class SilverMealHeuristic(InventoryModel):
    """Silver-Meal heuristic for lot-sizing."""

    def calculate_cost(self):
        replenishments = {}
        period = 0

        while period < len(self.demand):
            cumulative_demand = 0
            min_avg_cost_per_period = float('inf')
            T = 0  # Number of periods included in the current order

            # Determine the optimal number of periods to cover with this replenishment
            for T in range(1, len(self.demand) - period + 1):
                # Accumulate demand for exactly T periods
                cumulative_demand += self.demand[period + T - 1]

                # Calculate carrying costs based on the remaining inventory for each period
                carrying_cost = sum(
                    self.demand[period + t] * t * self.carrying_charge * self.unit_cost
                    for t in range(1, T)
                )

                # Calculate the average cost per period, including ordering and carrying costs
                avg_cost = (self.ordering_cost + carrying_cost) / T

                # Stop if including another period increases the average cost per period
                if avg_cost > min_avg_cost_per_period:
                    # Remove the last added period's demand from cumulative demand
                    cumulative_demand -= self.demand[period + T - 1]
                    T -= 1
                    break

                # Update the minimum average cost per period
                min_avg_cost_per_period = avg_cost

            # Place an order to cover exactly T periods, ensuring zero-ending inventory before the next replenishment
            replenishments[period] = cumulative_demand
            period += T  # Move to the next period after covering T periods

        # Adjust the last order if necessary to ensure zero-ending inventory at the end of the horizon
        if period < len(self.demand):
            last_demand = sum(self.demand[period:])
            replenishments[period] = last_demand

        return self.calculate_total_cost(replenishments)

class LeastUnitCostHeuristic(InventoryModel):
    """Least Unit Cost heuristic for lot-sizing."""

    def calculate_cost(self):
        replenishments = {}
        period = 0

        while period < len(self.demand):
            cumulative_demand = 0
            min_cost_per_unit = float('inf')
            T = 0  # Number of periods included in the current order

            # Determine the optimal number of periods to cover with this replenishment
            for T in range(1, len(self.demand) - period + 1):
                cumulative_demand += self.demand[period + T - 1]  # Accumulate demand for exactly T periods

                # Calculate carrying costs based on ending inventory for each period in the current order
                carrying_cost = sum(
                    (cumulative_demand - sum(self.demand[period:period + t + 1])) * self.carrying_charge * self.unit_cost
                    for t in range(T - 1)
                )

                # Calculate the cost per unit including ordering and carrying costs
                cost_per_unit = (self.ordering_cost + carrying_cost) / cumulative_demand

                # Stop if including another period would increase the cost per unit
                if cost_per_unit > min_cost_per_unit:
                    # Break if adding another period increases the cost per unit
                    cumulative_demand -= self.demand[period + T - 1]  # Remove the last added period's demand
                    T -= 1
                    break

                # Update the minimum cost per unit
                min_cost_per_unit = cost_per_unit

            # Place an order to cover exactly T periods, ensuring zero-ending inventory before the next replenishment
            replenishments[period] = cumulative_demand
            period += T  # Move to the next period after covering T periods

        # Adjust the last order if necessary to ensure zero-ending inventory at the end of the horizon
        if period < len(self.demand):
            last_demand = sum(self.demand[period:])
            replenishments[period] = last_demand

        return self.calculate_total_cost(replenishments)

class EOQTimeSupplyModel(InventoryModel):
    """EOQ model where the order quantity is interpreted as a time supply."""

    def calculate_cost(self):
        # Calculate EOQ based on average demand
        avg_demand = sum(self.demand) / len(self.demand)
        eoq = math.sqrt((2 * self.ordering_cost * avg_demand) / (self.carrying_charge * self.unit_cost))

        # Calculate the number of periods this EOQ quantity would cover
        periods_to_cover = eoq / avg_demand
        replenishments = {}
        period = 0

        while period < len(self.demand):
            # Determine the total demand for the calculated time supply period
            time_supply_demand = sum(self.demand[period: min(period + int(round(periods_to_cover)), len(self.demand))])

            # Place replenishment order for this time supply quantity
            replenishments[period] = time_supply_demand

            # Move to the next period after covering the time supply
            period += int(round(periods_to_cover))

        # Calculate total cost based on the generated replenishments
        return self.calculate_total_cost(replenishments)

class LotForLotModel(InventoryModel):
    """Lot for Lot model for lot-sizing."""

    def calculate_cost(self):
        replenishments = {}
        
        # Place an order for exactly the demand in each period
        for period, demand in enumerate(self.demand):
            replenishments[period] = demand  # Order exactly what is needed each period
        
        # Calculate the total cost for this replenishment plan
        return self.calculate_total_cost(replenishments)



class PartPeriodBalancingModel(InventoryModel):
    """Part-Period Balancing heuristic for lot-sizing."""

    def calculate_cost(self):
        replenishments = {}
        period = 0

        while period < len(self.demand):
            cumulative_demand = 0
            cumulative_carrying_cost = 0
            min_cost_difference = float('inf')
            best_T = 0

            # Determine the optimal number of periods to cover for this order
            for T in range(1, len(self.demand) - period + 1):
                cumulative_demand += self.demand[period + T - 1]

                # Calculate carrying costs for each period in this cumulative demand
                cumulative_carrying_cost += (T - 1) * self.demand[period + T - 1] * self.carrying_charge * self.unit_cost
                
                # Check the difference between cumulative carrying cost and setup cost
                cost_difference = abs(cumulative_carrying_cost - self.ordering_cost)

                # Stop if adding another period increases the difference
                if cost_difference > min_cost_difference:
                    break

                # Update the best period count if this T is closer to the setup cost
                min_cost_difference = cost_difference
                best_T = T

            # Place an order for the demand over the selected number of periods
            replenishments[period] = sum(self.demand[period:period + best_T])
            period += best_T  # Move to the next period after covering best_T periods

        # Calculate total cost based on the generated replenishments
        return self.calculate_total_cost(replenishments)
    


class ThreeMonthReplenishmentModel(InventoryModel):
    """Three-Month Replenishment model for lot-sizing."""

    def calculate_cost(self):
        replenishments = {}
        period = 0

        while period < len(self.demand):
            # Determine demand for the next three periods, or remaining periods if less than three
            periods_to_cover = min(3, len(self.demand) - period)
            cumulative_demand = sum(self.demand[period:period + periods_to_cover])

            # Place an order covering the demand for these periods
            replenishments[period] = cumulative_demand
            
            # Move to the next period after covering three months
            period += periods_to_cover

        # Calculate the total cost based on the generated replenishments
        return self.calculate_total_cost(replenishments)

class InventoryGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Inventory Management Models")
        self.root.geometry("800x600")

        # Default values
        default_demand = "200, 300, 180, 70, 50, 20, 10, 30, 90, 120, 150, 190"  # Example demand values
        default_ordering_cost = 160
        default_unit_cost = 5
        default_carrying_charge = 0.1

        # Inputs for the GUI
        self.demand_label = tk.Label(root, text="Demand (comma-separated):")
        self.demand_label.pack()
        self.demand_entry = tk.Entry(root, width=50)
        self.demand_entry.insert(0, default_demand)
        self.demand_entry.pack()

        self.ordering_cost_label = tk.Label(root, text="Ordering Cost:")
        self.ordering_cost_label.pack()
        self.ordering_cost_entry = tk.Entry(root)
        self.ordering_cost_entry.insert(0, default_ordering_cost)
        self.ordering_cost_entry.pack()

        self.unit_cost_label = tk.Label(root, text="Unit Cost:")
        self.unit_cost_label.pack()
        self.unit_cost_entry = tk.Entry(root)
        self.unit_cost_entry.insert(0, default_unit_cost)
        self.unit_cost_entry.pack()

        self.carrying_charge_label = tk.Label(root, text="Carrying Charge (as a decimal):")
        self.carrying_charge_label.pack()
        self.carrying_charge_entry = tk.Entry(root)
        self.carrying_charge_entry.insert(0, default_carrying_charge)
        self.carrying_charge_entry.pack()

        # Dropdown for selecting the model
        self.model_label = tk.Label(root, text="Select Model:")
        self.model_label.pack()
        self.model_var = tk.StringVar()
        self.model_dropdown = ttk.Combobox(
            root, textvariable=self.model_var, state="readonly",
            values=["EOQ Model", "Wagner-Whitin Model", "Silver-Meal Heuristic",
                    "Least Unit Cost", "EOQ Time Supply", "Lot For Lot",
                    "Part Period Balancing", "Three-Month Replenishment"]
        )
        self.model_dropdown.pack()

        # Buttons
        self.calculate_button = tk.Button(root, text="Calculate", command=self.calculate)
        self.calculate_button.pack()

        self.save_button = tk.Button(root, text="Save to JSON", command=self.save_to_json)
        self.save_button.pack()

        # Results area
        self.results_text = tk.Text(root, height=15, width=80)
        self.results_text.pack()

    def calculate(self):
        try:
            # Parse input values
            demand = list(map(int, self.demand_entry.get().split(',')))
            ordering_cost = float(self.ordering_cost_entry.get())
            unit_cost = float(self.unit_cost_entry.get())
            carrying_charge = float(self.carrying_charge_entry.get())

            # Determine model and initialize
            model = self.model_var.get()
            if model == "EOQ Model":
                inventory_model = EOQModel(demand, ordering_cost, unit_cost, carrying_charge)
            elif model == "Wagner-Whitin Model":
                inventory_model = WagnerWhitinModel(demand, ordering_cost, unit_cost, carrying_charge)
            elif model == "Silver-Meal Heuristic":
                inventory_model = SilverMealHeuristic(demand, ordering_cost, unit_cost, carrying_charge)
            elif model == "Least Unit Cost":
                inventory_model = LeastUnitCostHeuristic(demand, ordering_cost, unit_cost, carrying_charge)
            elif model == "EOQ Time Supply":
                inventory_model = EOQTimeSupplyModel(demand, ordering_cost, unit_cost, carrying_charge)
            elif model == "Lot For Lot":
                inventory_model = LotForLotModel(demand, ordering_cost, unit_cost, carrying_charge)
            elif model == "Part Period Balancing":
                inventory_model = PartPeriodBalancingModel(demand, ordering_cost, unit_cost, carrying_charge)
            elif model == "Three-Month Replenishment":
                inventory_model = ThreeMonthReplenishmentModel(demand, ordering_cost, unit_cost, carrying_charge)
            else:
                raise ValueError("Please select a valid model.")

            # Calculate and display results
            total_cost = inventory_model.calculate_cost()
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"Total Cost: {total_cost}\n")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def save_to_json(self):
        try:
            demand = list(map(int, self.demand_entry.get().split(',')))
            ordering_cost = float(self.ordering_cost_entry.get())
            unit_cost = float(self.unit_cost_entry.get())
            carrying_charge = float(self.carrying_charge_entry.get())

            # Create the data dictionary
            data = {
                "demand_rate": demand,
                "ordering_cost": ordering_cost,
                "unit_cost": unit_cost,
                "carrying_charge": carrying_charge
            }

            # Save to JSON file
            with open('inventory_data.json', 'w') as f:
                json.dump(data, f, indent=4)

            messagebox.showinfo("Success", "Data saved to inventory_data.json")
        except Exception as e:
            messagebox.showerror("Error", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = InventoryGUI(root)
    root.mainloop()


