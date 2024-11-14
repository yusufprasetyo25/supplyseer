import pytest
import numpy as np
import pandas as pd
from src.supplyseer.tsa.advanced.vectorfields import StockoutProbability, DemandInventorySimulation  # replace with your module path

def test_stockout_probability_initialization():
    # Test default initialization
    stockout_prob = StockoutProbability()
    assert stockout_prob.inventory.shape == (20,)
    assert stockout_prob.demand.shape == (20,)

    # Test custom inventory and demand data with np.array
    inventory_data = np.linspace(10, 100, 10)
    demand_data = np.linspace(5, 50, 10)
    stockout_prob_custom = StockoutProbability(inventory_data=inventory_data, demand_data=demand_data)
    assert stockout_prob_custom.inventory.shape == (10,)
    assert stockout_prob_custom.demand.shape == (10,)

def test_stockout_probability_compute_probability():
    stockout_prob = StockoutProbability()
    stockout_prob.compute_probability()
    assert stockout_prob.Z is not None
    assert stockout_prob.Z.shape == stockout_prob.X.shape

def test_stockout_probability_invalid_data():
    # Test invalid data input
    with pytest.raises(ValueError):
        StockoutProbability(inventory_data="invalid data")

def test_demand_inventory_simulation_initialization():
    # Test initialization with matching length demand and inventory data
    demand_data = pd.Series([10, 12, 14, 13, 12, 11])
    inventory_data = pd.Series([100, 95, 85, 78, 70, 60])
    sim = DemandInventorySimulation(demand_data=demand_data, inventory_data=inventory_data)
    assert sim.demand.shape[0] == sim.inventory.shape[0]
    assert sim.dY_data is None
    assert sim.dX_data is None

def test_demand_inventory_simulation_compute_differences():
    # Test computing differences with simple example data
    demand_data = np.array([10, 12, 15, 14, 13])
    inventory_data = np.array([100, 95, 85, 80, 75])
    sim = DemandInventorySimulation(demand_data=demand_data, inventory_data=inventory_data)
    diffs = sim.compute_differences()
    
    # Check that the differences have the correct shapes
    assert sim.dY_data.shape == (4,)
    assert sim.dX_data.shape == (4,)
    assert diffs["gradient_inventory"].shape == sim.X.shape
    assert diffs["gradient_demand"].shape == sim.Y.shape

def test_demand_inventory_simulation_mismatched_data_length():
    # Test initialization with mismatched demand and inventory data lengths
    demand_data = np.array([10, 12, 15, 14])
    inventory_data = np.array([100, 95, 85])
    with pytest.raises(ValueError):
        DemandInventorySimulation(demand_data=demand_data, inventory_data=inventory_data)
