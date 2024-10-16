import numpy as np
from pydantic import BaseModel, Field
from typing import List, Tuple, Optional, Dict
from enum import Enum
from datetime import datetime

class EventType(str, Enum):
    DEMAND_SPIKE = "demand_spike"
    INVENTORY_CHANGE = "inventory_change"
    STOCKOUT = "stockout"
    RESTOCK = "restock"

class SupplyChainEvent(BaseModel):
    """
    Pydantic model for supply chain events with validation
    """
    time: float = Field(..., ge=0)
    event_type: EventType
    location_id: int = Field(..., ge=0)
    magnitude: float = Field(..., ge=0)
    inventory_level: float = Field(..., ge=0)
    demand_level: float = Field(..., ge=0)
    timestamp: datetime = Field(default_factory=datetime.now)

class SupplyChainData(BaseModel):
    sales_demand: List[float] = Field(..., description="Sales demand at each time step")
    inventory_levels: List[float] = Field(..., description="Inventory levels at each time step")
    time_points: List[float] = Field(..., description="Time points for each demand and inventory entry")
    max_time: float = Field(..., description="Maximum time horizon for simulation")

    

class MultivariateHawkesProcess:
    def __init__(self, baseline: np.ndarray, alpha: np.ndarray, beta: np.ndarray):
        """
        Initialize the Multivariate Hawkes Process with baseline intensities, influence matrix, and decay rates.
        
        :param baseline: Baseline intensity for each dimension (shape: [D,])
        :param alpha: Influence matrix for event types (shape: [D, D])
        :param beta: Decay rates for each dimension (shape: [D, D])
        """
        self.baseline = baseline
        self.alpha = alpha
        self.beta = beta
        self.num_dimensions = len(baseline)
        self.history = [[] for _ in range(self.num_dimensions)]
        self.events = []
        # Track inventory levels for each location
        self.inventory_levels = [10.0 for _ in range(self.num_dimensions)]  # Initial inventory levels

    def simulate(self, max_time: float):
        """
        Simulate the multivariate Hawkes process up to a maximum time.

        :param max_time: The maximum time horizon for the simulation.
        :return: List of events for each dimension.
        """
        t = 0
        while t < max_time:
            # Calculate intensity for each dimension
            intensities = self._calculate_intensities(t)
            total_intensity = np.sum(intensities)

            # Sample next event time
            if total_intensity == 0:
                break
            dt = np.random.exponential(1.0 / total_intensity)
            t += dt
            # Calculate intensity for each dimension
            intensities = self._calculate_intensities(t)
            total_intensity = np.sum(intensities)

            # Sample next event time
            if total_intensity == 0:
                break


            # Break if beyond max_time
            if t >= max_time:
                break

            # Determine which dimension the event occurs in
            probs = intensities / total_intensity
            dimension = np.random.choice(self.num_dimensions, p=probs)

            # Update inventory level based on the type of event
            if dimension == 0:
                # Demand spike reduces inventory
                magnitude = np.random.uniform(0.1, 1.0)
                self.inventory_levels[dimension] = max(0.0, self.inventory_levels[dimension] - magnitude)
            else:
                # Inventory change restocks inventory
                magnitude = np.random.uniform(0.1, 1.0)
                self.inventory_levels[dimension] += magnitude

            # Record the event
            self.history[dimension].append(t)
            event = SupplyChainEvent(
                time=t,
                event_type=EventType.DEMAND_SPIKE if dimension == 0 else EventType.INVENTORY_CHANGE,
                location_id=dimension,
                magnitude=magnitude,
                inventory_level=self.inventory_levels[dimension],  # Correctly assign the inventory level
                demand_level=self.baseline[dimension] if dimension == 0 else self.inventory_levels[dimension]  # Add demand level
            )
            self.events.append(event)

        return self.history, self.events

    def _calculate_intensities(self, current_time):
        """
        Calculate the intensity for each dimension at the current time.

        :param current_time: The current time in the simulation.
        :return: Intensities for each dimension.
        """
        intensities = np.copy(self.baseline)
        for d in range(self.num_dimensions):
            for past_dim in range(self.num_dimensions):
                for event_time in self.history[past_dim]:
                    if event_time < current_time:
                        intensities[d] += self.alpha[d, past_dim] * np.exp(-self.beta[d, past_dim] * (current_time - event_time))
        return intensities

def simulate_supply_chain(data: SupplyChainData) -> Dict[str, list]:
    # Set up parameters for the Hawkes Process
    num_dimensions = 2
    baseline = np.array([0.1, 0.2])  # Baseline intensity for sales demand and inventory levels
    alpha = np.array([[0.5, 0.3], [0.4, 0.6]])  # Influence matrix
    beta = np.array([[1.0, 1.0], [1.0, 1.0]])  # Decay rates

    # Initialize the Hawkes Process
    hawkes_process = MultivariateHawkesProcess(baseline, alpha, beta)

    # Simulate the process
    history, events = hawkes_process.simulate(data.max_time)

    # Store results in a dictionary
    result = {
        "sales_demand_events": history[0],
        "inventory_level_events": history[1],
        "events": [event.dict() for event in events]
    }
    
    return result