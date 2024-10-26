import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta

class SupplyChainNode:
    def __init__(self, name, dataframe):
        self.name = name
        self.stock_level = dataframe['stock_level'].iloc[0]
        self.sales_demand = dataframe['sales_demand'].values
        self.timestamps = dataframe.index
        
        # Store reorder parameters as single values instead of Series
        self.reorder_point = (dataframe['reorder_point'].iloc[0] 
                            if 'reorder_point' in dataframe 
                            else self.stock_level * 0.5)
        self.target_stock = (dataframe['target_stock'].iloc[0] 
                           if 'target_stock' in dataframe 
                           else self.stock_level)
        self.replenishment_lead_time = (dataframe['replenishment_lead_time'].iloc[0] 
                                      if 'replenishment_lead_time' in dataframe 
                                      else 1)
        
        self.pending_orders = []
        
        self.results = pd.DataFrame(index=dataframe.index, 
                                  columns=['stock_level', 'sales_demand', 'potential_energy', 'ordered_amount'])
        self.results['sales_demand'] = self.sales_demand
        self.results['ordered_amount'] = 0
        self.results.iloc[0, self.results.columns.get_loc('stock_level')] = self.stock_level
    def update_inventory(self, incoming, outgoing, timestamp_idx):
        """Update inventory using numeric index instead of timestamp"""
        current_inventory = self.results.iloc[timestamp_idx]['stock_level']
        net_flow = incoming - outgoing
        new_inventory = current_inventory + net_flow
        new_inventory = max(new_inventory, 0)  # Prevent negative inventory
        
        # Store the new inventory level
        self.results.iloc[timestamp_idx, self.results.columns.get_loc('stock_level')] = new_inventory
        
        # Update potential energy
        self.calculate_potential_energy(timestamp_idx)
        
        return net_flow
    
    def check_and_place_order(self, current_timestamp_idx):
        """Check if we need to place a replenishment order"""
        current_stock = float(self.results.iloc[current_timestamp_idx]['stock_level'])
        
        if current_stock <= self.reorder_point:
            order_amount = self.target_stock - current_stock
            arrival_time = current_timestamp_idx + self.replenishment_lead_time
            self.pending_orders.append((arrival_time, order_amount))
            self.results.iloc[current_timestamp_idx, self.results.columns.get_loc('ordered_amount')] = order_amount
            print(f"{self.name} placed order for {order_amount:.2f} units, arriving in {self.replenishment_lead_time} days")


    def process_pending_orders(self, current_timestamp_idx):
        """Process any pending orders that have arrived"""
        arrived_orders = [order for order in self.pending_orders if order[0] <= current_timestamp_idx]
        self.pending_orders = [order for order in self.pending_orders if order[0] > current_timestamp_idx]
        
        total_arrived = sum(order[1] for order in arrived_orders)
        if total_arrived > 0:
            self.update_inventory(incoming=total_arrived, outgoing=0, timestamp_idx=current_timestamp_idx)
            print(f"{self.name} received {total_arrived:.2f} units from external supplier")

    def calculate_potential_energy(self, timestamp_idx):
        """Calculate potential energy using numeric index"""
        stock_level = self.results.iloc[timestamp_idx]['stock_level']
        potential_energy = stock_level * 0.5
        self.results.iloc[timestamp_idx, self.results.columns.get_loc('potential_energy')] = potential_energy

class SupplyChainNetwork:
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges

    def calculate_diffusion_flow(self, from_node, to_node, timestamp_idx, diffusion_coefficient=0.1):
        """Calculate diffusion flow using numeric index"""
        from_stock = self.nodes[from_node].results.iloc[timestamp_idx]['stock_level']
        to_stock = self.nodes[to_node].results.iloc[timestamp_idx]['stock_level']
        
        inventory_difference = from_stock - to_stock
        diffusion_flow = diffusion_coefficient * inventory_difference
        
        return np.clip(diffusion_flow, -from_stock, to_stock)

    def simulate_flow(self):
        # Get timestamps from any node
        first_node = next(iter(self.nodes.values()))
        timesteps = first_node.timestamps
        num_steps = len(timesteps)
        
        kinetic_energy_history = []
        time_deltas = []
        initial_time = timesteps[0]
        
        # Simulate for each timestep
        for t in range(num_steps):
            if t > 0:  # Copy previous state to current timestep for all nodes
                for node in self.nodes.values():
                    prev_stock = node.results.iloc[t-1]['stock_level']
                    node.results.iloc[t, node.results.columns.get_loc('stock_level')] = prev_stock
            
            print(f"\nTimestep {t}: {timesteps[t]}")
            total_kinetic_energy = 0
            
            # Record time delta
            time_delta = (timesteps[t] - initial_time).total_seconds() / (24 * 3600)
            time_deltas.append(time_delta)

            for node in self.nodes.values():
                node.process_pending_orders(t)
            
            # Process sales demand for retail nodes
            for node_name, node in self.nodes.items():
                if 'Retail' in node_name:
                    sales_demand = node.sales_demand[t]
                    node.update_inventory(incoming=0, outgoing=sales_demand, timestamp_idx=t)
            
            # Calculate flows between nodes
            for (from_node, to_node), edge_data in self.edges.items():
                potential = edge_data['potential']
                max_flow_rate = edge_data['max_flow_rate']
                
                # Calculate forces
                demand_factor = self.nodes[to_node].sales_demand[t] * 0.1  # Increased influence
                from_stock = self.nodes[from_node].results.iloc[t]['stock_level']
                to_stock = self.nodes[to_node].results.iloc[t]['stock_level']
                
                # Enhanced magnetic force calculation
                inventory_ratio = to_stock / max(1, from_stock)
                magnetic_force = potential + demand_factor * (1 - inventory_ratio)
                
                # Calculate flows
                base_flow = magnetic_force * (from_stock - to_stock)
                diffusion_flow = self.calculate_diffusion_flow(from_node, to_node, t)
                total_flow = np.clip(base_flow + diffusion_flow, -max_flow_rate, max_flow_rate)
                
                if abs(total_flow) > 0.01:
                    # Update inventories
                    self.nodes[from_node].update_inventory(0, total_flow, t)
                    self.nodes[to_node].update_inventory(total_flow, 0, t)
                    
                    # Calculate kinetic energy
                    delta_kinetic_energy = -total_flow * magnetic_force
                    total_kinetic_energy += delta_kinetic_energy
                    
                    print(f"Flow from {from_node} to {to_node}: {total_flow:.2f}")

            # Check inventory levels and place orders if needed
            for node in self.nodes.values():
                if 'Warehouse' in node.name:  # Only warehouses can place orders
                    node.check_and_place_order(t)
                    
            kinetic_energy_history.append(total_kinetic_energy)
            
            # Print current state
            for node_name, node in self.nodes.items():
                inventory = node.results.iloc[t]['stock_level']
                print(f"{node_name}: Stock Level = {inventory:.2f}")
        
        # Plotting
        plt.figure(figsize=(12, 6))
        
        # Plot Kinetic Energy
        plt.subplot(2, 1, 1)
        plt.plot(time_deltas, kinetic_energy_history, label="Total Kinetic Energy", color='r')
        plt.xlabel('Days from Start')
        plt.ylabel('Kinetic Energy')
        plt.title('Kinetic Energy Over Time')
        plt.legend()
        plt.grid(True)
        
        # Plot Stock Levels
        plt.subplot(2, 1, 2)
        for node_name, node in self.nodes.items():
            plt.plot(time_deltas, node.results['stock_level'], 
                    label=f"{node_name} Stock Level")
        plt.xlabel('Days from Start')
        plt.ylabel('Stock Level')
        plt.title('Stock Levels Over Time')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
