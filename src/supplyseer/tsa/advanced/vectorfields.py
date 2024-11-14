import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata


class StockoutProbability:
    def __init__(self, inventory_data=None, demand_data=None, num_points=20):
        # Accept custom inventory and demand data
        self.inventory = self._process_input_data(inventory_data, (1, 100), num_points)
        self.demand = self._process_input_data(demand_data, (1, 50), num_points)
        self.X, self.Y = np.meshgrid(self.inventory, self.demand)
        self.Z = None

    def _process_input_data(self, data, default_range, num_points):
        if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
            return data.to_numpy().flatten()
        elif isinstance(data, np.ndarray):
            return data.flatten()
        elif data is None:
            return np.linspace(default_range[0], default_range[1], num_points)
        else:
            raise ValueError("Data must be a pandas DataFrame, Series, numpy array, or None.")

    def compute_probability(self):
        # Stockout probability is inversely proportional to inventory
        self.Z = np.exp(-self.X / (self.Y + 1))

    def plot_probability_surface(self, style: str = "dark_background"):
        if self.Z is None:
            self.compute_probability()
        
        plt.style.use(style)
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(self.X, self.Y, self.Z, cmap='viridis', edgecolor='k', alpha=0.8)
        ax.set_title('Stockout Probability as Function of Inventory and Demand', fontsize=16)
        ax.set_xlabel('Inventory Level')
        ax.set_ylabel('Demand Rate')
        ax.set_zlabel('Stockout Probability')
        plt.show()

class DemandInventorySimulation:
    def __init__(self, demand_data, inventory_data):
        # Process demand and inventory data, accepting multiple input formats
        self.demand = self._process_input_data(demand_data)
        self.inventory = self._process_input_data(inventory_data)
        
        # Ensure that demand and inventory data have matching lengths
        if len(self.demand) != len(self.inventory):
            raise ValueError("Demand and inventory data must have the same length.")
        
        # Initialize differences and interpolated vector field
        self.dY_data = None
        self.dX_data = None
        self.dX = None
        self.dY = None
        self.X = None
        self.Y = None

    def _process_input_data(self, data):
        # Converts input data to a flat numpy array regardless of input type
        if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
            return data.to_numpy().flatten()
        elif isinstance(data, np.ndarray):
            return data.flatten()
        elif isinstance(data, list):
            return np.array(data)
        else:
            raise ValueError("Data must be a pandas DataFrame, Series, numpy array, or list.")

    def compute_differences(self):
        # Calculate the differences for dY and dX to represent changes in inventory and demand
        self.dY_data = np.diff(self.inventory)
        self.dX_data = np.diff(self.demand)
        
        # Adjust historical data arrays to match the length of the differences
        historical_inventory = self.inventory[:-1]
        historical_demand = self.demand[:-1]
        
        # Define a grid based on the historical data ranges for interpolation
        inventory_grid = np.linspace(np.min(historical_inventory), np.max(historical_inventory), 50)
        demand_grid = np.linspace(np.min(historical_demand), np.max(historical_demand), 50)
        
        # Create meshgrid for the interpolated vector field
        self.X, self.Y = np.meshgrid(demand_grid, inventory_grid)
        
        # Interpolate dY and dX to create a smooth vector field
        self.dY = griddata((historical_demand, historical_inventory), self.dY_data, (self.X, self.Y), method='linear', fill_value=0)
        self.dX = griddata((historical_demand, historical_inventory), self.dX_data, (self.X, self.Y), method='linear', fill_value=0)

        return {
            "gradient_inventory": self.dY,
            "gradient_demand": self.dX,
            "demand": self.demand,
            "inventory": self.inventory
        }

    def plot_simulation_vector_field(self, style: str = "dark_background"):
        # Plot the data-driven vector field
        plt.style.use(style)
        if self.dX is None or self.dY is None:
           _ = self.compute_differences()
        plt.figure(figsize=(12, 8))
        plt.streamplot(self.X, self.Y, self.dX, self.dY, color='#39FF14', linewidth=1, density=1.5, arrowstyle='->', arrowsize=1.5)
        plt.quiver(self.X, self.Y, self.dX, self.dY, color='grey', scale=500, width=0.005, alpha=0.4)
        plt.title("Vector Field of Inventory Levels vs. Demand Dynamics (Data-Driven)", fontsize=16)
        plt.xlabel("Demand Level")
        plt.ylabel("Inventory Level")
        plt.grid(True)
        plt.show()
