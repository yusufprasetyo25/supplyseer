from ortools.sat.python import cp_model
import pandas as pd
import polars as pl
import itertools

class SupplyChainModel(cp_model.CpModel):
    def __init__(self, products, locations, time_periods=7):
        super().__init__()
        self.products = products  # Dictionary {product_id: name}
        self.locations = locations  # Dictionary {location_id: name}
        self.time_periods = time_periods
        self.inventory_vars = {}  # (product_id, location_id) -> list of inventory variables over time periods
        self.production_vars = {}  # (product_id, location_id) -> list of production variables over time periods
        self.total_cost = self.NewIntVar(0, 10**9, "total_cost")

    def add_inventory_constraint(self, product_id, location_id, min_stock, max_stock, initial_stock, holding_cost):
        # Validate inputs
        if product_id not in self.products or location_id not in self.locations:
            raise ValueError("Invalid product or location ID")

        # Create inventory variables for each time period
        self.inventory_vars[(product_id, location_id)] = [
            self.NewIntVar(min_stock, max_stock, f"inv_{product_id}_{location_id}_{t}")
            for t in range(self.time_periods)
        ]

        # Set initial inventory constraint
        self.Add(self.inventory_vars[(product_id, location_id)][0] == initial_stock)

        # Calculate inventory holding cost
        inventory_cost = sum(self.inventory_vars[(product_id, location_id)][t] * holding_cost 
                             for t in range(self.time_periods))
        
        return inventory_cost

    def add_production_constraint(self, product_id, location_id, max_production, production_cost):
        # Create production variables for each time period
        self.production_vars[(product_id, location_id)] = [
            self.NewIntVar(0, max_production, f"prod_{product_id}_{location_id}_{t}")
            for t in range(self.time_periods)
        ]

        # Calculate production cost
        production_cost_var = sum(self.production_vars[(product_id, location_id)][t] * production_cost 
                                  for t in range(self.time_periods))

        return production_cost_var

    def add_demand_constraint(self, product_id, location_id, demand_series):
        # Validate that demand matches the number of time periods
        demand = demand_series.values.tolist()
        if len(demand) != self.time_periods:
            raise ValueError("Demand list must match the number of time periods")

        inv_vars = self.inventory_vars[(product_id, location_id)]
        prod_vars = self.production_vars[(product_id, location_id)]

        # Add inventory balance constraints for each time period
        for t in range(1, self.time_periods):
            self.Add(inv_vars[t] == inv_vars[t-1] + prod_vars[t] - demand[t])

    def set_objective(self, total_inventory_cost, total_production_cost):
        # Set the objective to minimize the total cost (inventory cost + production cost)
        self.Add(self.total_cost == total_inventory_cost + total_production_cost)
        self.Minimize(self.total_cost)

class SupplyChainSolver(cp_model.CpSolver):
    def __init__(self):
        super().__init__()
        self.solution_history = []

    def solve_with_timeout(self, model, timeout_seconds):
        self.parameters.max_time_in_seconds = timeout_seconds
        result = self.Solve(model)
        self.solution_history.append({
            'status': self.StatusName(result),
            'objective_value': self.ObjectiveValue() if result in [cp_model.OPTIMAL, cp_model.FEASIBLE] else None
        })
        return result

    def print_solution(self, model):
        print("Solution:")
        for (product_id, location_id), inv_vars in model.inventory_vars.items():
            print(f"Product {product_id} at Location {location_id}:")
            print("  Inventory levels:", [self.Value(var) for var in inv_vars])
            print("  Production quantities:", [self.Value(var) for var in model.production_vars[(product_id, location_id)]])
        print(f"Total Cost: {self.Value(model.total_cost)}")

    def get_solution_summary(self):
        return {
            'num_solutions': len(self.solution_history),
            'last_status': self.solution_history[-1]['status'] if self.solution_history else None,
            'best_objective': min((sol['objective_value'] for sol in self.solution_history if sol['objective_value'] is not None), default=None)
        }
    


class TruckDriverScheduleModel(cp_model.CpModel):
    def __init__(self, num_drivers=4, num_windows=3, num_days=3):
        super().__init__()
        self.num_drivers = num_drivers
        self.num_windows = num_windows
        self.num_days = num_days
        self.assignments = {}  # (driver, day, window) -> assignment variable
        self._initialize_variables()
        
    def _initialize_variables(self):
        for d in range(self.num_drivers):
            for day in range(self.num_days):
                for w in range(self.num_windows):
                    self.assignments[(d, day, w)] = self.NewBoolVar(f"assignment_d{d}_day{day}_w{w}")

    def add_constraints(self):
        # Each delivery window must be covered by one driver
        for day in range(self.num_days):
            for w in range(self.num_windows):
                self.AddExactlyOne(self.assignments[(d, day, w)] for d in range(self.num_drivers))

        # No driver can cover more than one window per day
        for d in range(self.num_drivers):
            for day in range(self.num_days):
                self.AddAtMostOne(self.assignments[(d, day, w)] for w in range(self.num_windows))

        # Each driver must be assigned at least two delivery windows over the n days
        min_windows_per_driver = (self.num_windows * self.num_days) // self.num_drivers
        if self.num_windows * self.num_days % self.num_drivers == 0:
            max_windows_per_driver = min_windows_per_driver
        else:
            max_windows_per_driver = min_windows_per_driver + 1

        for d in range(self.num_drivers):
            windows_worked = [
                self.assignments[(d, day, w)]
                for day in range(self.num_days)
                for w in range(self.num_windows)
            ]
            self.Add(min_windows_per_driver <= sum(windows_worked))
            self.Add(sum(windows_worked) <= max_windows_per_driver)

        # New constraint: If a driver had window 2 (evening) on day t-1, they cannot have window 0 (morning) on day t
        for d in range(self.num_drivers):
            for day in range(1, self.num_days):
                self.Add(self.assignments[(d, day - 1, 2)] + self.assignments[(d, day, 0)] <= 1)

class TruckDriverScheduleSolver(cp_model.CpSolver):
    def __init__(self):
        super().__init__()
        self.solution_history = []

    def solve_with_callback(self, model, solution_limit=5):
        solution_printer = DriverSchedulePrinter(
            model.assignments, model.num_drivers, model.num_days, model.num_windows, solution_limit
        )
        result = self.Solve(model, solution_printer)
        return result

class DriverSchedulePrinter(cp_model.CpSolverSolutionCallback):
    """Print intermediate solutions."""
    def __init__(self, assignments, num_drivers, num_days, num_windows, limit):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self._assignments = assignments
        self._num_drivers = num_drivers
        self._num_days = num_days
        self._num_windows = num_windows
        self._solution_count = 0
        self._solution_limit = limit

    def on_solution_callback(self):
        self._solution_count += 1
        print(f"Solution {self._solution_count}")
        for day in range(self._num_days):
            print(f"Day {day}")
            for d in range(self._num_drivers):
                is_working = False
                for w in range(self._num_windows):
                    if self.Value(self._assignments[(d, day, w)]):
                        is_working = True
                        print(f"  Driver {d} covers delivery window {w}")
                if not is_working:
                    print(f"  Driver {d} is off duty")
        if self._solution_count >= self._solution_limit:
            print(f"Stop search after {self._solution_limit} solutions")
            self.StopSearch()

    def solution_count(self):
        return self._solution_count