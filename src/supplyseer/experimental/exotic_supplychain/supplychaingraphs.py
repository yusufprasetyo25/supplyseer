import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
import networkx as nx


class SupplyChainNode:
    def __init__(self, name, dataframe, flow_mode='projected_stock', verbose=False):
        """
        Initialize node with specified flow mode
        flow_mode options: 
            - 'projected_stock': Use reorder points with projected stock levels
            - 'diffusion': Use diffusion-based flow between nodes, reorder points for warehouses
            - 'hybrid': Use diffusion for store-warehouse flow, reorder points for warehouses
        """
        self.name = name
        self.timestamps = dataframe.index
        self.flow_mode = flow_mode
        self.is_warehouse = 'Warehouse' in name
        self.verbose = verbose
        
        # Initial stock level
        self.stock_level = dataframe['stock_level'].iloc[0]
        
        # Store demand and sales data
        self.forecast_demand = dataframe['forecast_demand'].values if 'forecast_demand' in dataframe else np.zeros(len(dataframe))
        self.actual_sales = dataframe['actual_sales'].values if 'actual_sales' in dataframe else np.zeros(len(dataframe))
        
        # Reorder parameters
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
        self.pending_flows = []
        
        # Results DataFrame with expanded columns
        self.results = pd.DataFrame(index=dataframe.index, 
                                  columns=['stock_level', 
                                         'forecast_demand',
                                         'actual_sales',
                                         'unused_potential',
                                         'potential_energy',
                                         'ordered_amount',
                                         'stockout'])
        
        # Initialize results
        self.results.loc[dataframe.index[0], "stock_level"] = self.stock_level
        # self.results['stock_level'].iloc[0] = self.stock_level
        self.results['forecast_demand'] = self.forecast_demand
        self.results['actual_sales'] = self.actual_sales
        self.results['ordered_amount'] = 0
        self.results['stockout'] = 0
        
        # Calculate initial unused potential
        self.calculate_unused_potential(0)

    def check_and_place_order(self, current_timestamp_idx):
        """Check if we need to place a replenishment order based on flow mode"""
        if (self.flow_mode in ['diffusion', 'hybrid'] and not self.is_warehouse):
            return
            
        if self.flow_mode == 'projected_stock' or self.is_warehouse:
            if current_timestamp_idx < 2:
                return
                
            current_stock = float(self.results.iloc[current_timestamp_idx]['stock_level'])
            prev_sales = self.results.iloc[current_timestamp_idx-2:current_timestamp_idx]['actual_sales'].mean()
            projected_stock = current_stock - (prev_sales * self.replenishment_lead_time)
            
            if projected_stock <= self.reorder_point:
                order_amount = self.target_stock - current_stock
                arrival_time = current_timestamp_idx + self.replenishment_lead_time
                self.pending_orders.append((arrival_time, order_amount))
                self.results.iloc[current_timestamp_idx, self.results.columns.get_loc('ordered_amount')] = order_amount
                
                if self.verbose:
                    print(f"{self.name} placed order for {order_amount:.2f} units, arriving in {self.replenishment_lead_time} days")
                    print(f"  Current stock: {current_stock:.2f}, Projected stock: {projected_stock:.2f}, "
                          f"Avg daily sales: {prev_sales:.2f}")

    def process_pending_orders(self, current_timestamp_idx):
        """Process any pending orders that have arrived"""
        arrived_orders = [order for order in self.pending_orders if order[0] <= current_timestamp_idx]
        self.pending_orders = [order for order in self.pending_orders if order[0] > current_timestamp_idx]
        
        total_arrived = sum(order[1] for order in arrived_orders)
        if total_arrived > 0:
            self.update_inventory(incoming=total_arrived, outgoing=0, timestamp_idx=current_timestamp_idx)
            if self.verbose:
                print(f"{self.name} received {total_arrived:.2f} units from external supplier")

    def calculate_potential_energy(self, timestamp_idx):
        """Calculate potential energy using numeric index"""
        stock_level = self.results.iloc[timestamp_idx]['stock_level']
        potential_energy = stock_level * 0.5
        self.results.iloc[timestamp_idx, self.results.columns.get_loc('potential_energy')] = potential_energy
        
    def calculate_unused_potential(self, timestamp_idx):
        """Calculate unused demand potential"""
        available = self.results.iloc[timestamp_idx]['stock_level']
        sales = self.results.iloc[timestamp_idx]['actual_sales']
        unused = max(0, available - sales)
        self.results.iloc[timestamp_idx, self.results.columns.get_loc('unused_potential')] = unused
        return unused
        
    def update_inventory(self, incoming, outgoing, timestamp_idx):
        current_inventory = self.results.iloc[timestamp_idx]['stock_level']
        net_flow = incoming - outgoing
        new_inventory = max(current_inventory + net_flow, 0)
        
        self.results.iloc[timestamp_idx, self.results.columns.get_loc('stock_level')] = new_inventory
        self.results.iloc[timestamp_idx, self.results.columns.get_loc('stockout')] = 1 if new_inventory <= 0 else 0
        
        self.calculate_potential_energy(timestamp_idx)
        self.calculate_unused_potential(timestamp_idx)
        
        return net_flow
    
    def add_pending_flow(self, flow_amount, arrival_time):
        self.pending_flows.append((arrival_time, flow_amount))
    
    def add_pending_flow(self, flow_amount, arrival_time):
        """Add a pending flow from diffusion"""
        self.pending_flows.append((arrival_time, flow_amount))
        
    def process_pending_flows(self, current_timestamp_idx):
        """Process any pending flows that have arrived"""
        arrived_flows = [flow for flow in self.pending_flows if flow[0] <= current_timestamp_idx]
        self.pending_flows = [flow for flow in self.pending_flows if flow[0] > current_timestamp_idx]
        
        total_arrived = sum(flow[1] for flow in arrived_flows)
        if total_arrived > 0:
            self.update_inventory(incoming=total_arrived, outgoing=0, timestamp_idx=current_timestamp_idx)
            if self.verbose:
                print(f"{self.name} received {total_arrived:.2f} units from diffusion flow")


class SupplyChainNetwork:
    def __init__(self, nodes, edges, flow_mode='projected_stock', verbose=False):
        """
        Initialize network with specified flow mode
        flow_mode options: 
            - 'projected_stock': Use reorder points with projected stock levels
            - 'diffusion': Use diffusion-based flow between nodes, reorder points for warehouses
            - 'hybrid': Use diffusion for store-warehouse flow, reorder points for warehouses
        """
        self.nodes = nodes
        self.edges = edges
        self.flow_mode = flow_mode
        self.verbose = verbose
        
        # Ensure all nodes have same flow mode as network
        for node in self.nodes.values():
            node.flow_mode = flow_mode
        
        # New tracking for network-wide metrics
        self.network_metrics = pd.DataFrame(index=next(iter(nodes.values())).timestamps,
                                          columns=['total_forecast_demand',
                                                 'total_actual_sales',
                                                 'total_unused_potential',
                                                 'system_kinetic_energy'])


    def calculate_edge_potential(self, from_node, to_node, timestamp_idx):
        """Calculate potential between nodes considering demand and unused potential"""
        base_potential = self.edges[(from_node, to_node)]['potential']
        
        # Get demand metrics for destination node
        to_node_obj = self.nodes[to_node]
        forecast = to_node_obj.results.iloc[timestamp_idx]['forecast_demand']
        unused = to_node_obj.results.iloc[timestamp_idx]['unused_potential']
        current_stock = to_node_obj.results.iloc[timestamp_idx]['stock_level']
        
        # No potential if store is at or above target stock
        if current_stock >= to_node_obj.target_stock:
            return 0
        
        # Adjust potential based on forecast and unused potential
        demand_factor = forecast / (current_stock + 1)  # Higher when stock is low
        unused_factor = 1 - (unused / (current_stock + 1))  # Lower when unused potential is high
        
        # Additional factor based on how far from target stock
        target_factor = 1 - (current_stock / to_node_obj.target_stock)
        
        return base_potential * demand_factor * unused_factor * target_factor

    def calculate_diffusion_flow(self, from_node, to_node, timestamp_idx, diffusion_coefficient=0.1):
        """Calculate diffusion flow using numeric index with target stock limit"""
        from_stock = self.nodes[from_node].results.iloc[timestamp_idx]['stock_level']
        to_stock = self.nodes[to_node].results.iloc[timestamp_idx]['stock_level']
        to_node_obj = self.nodes[to_node]
        
        # Only flow if receiving node is below its target stock
        if to_stock >= to_node_obj.target_stock:
            return 0
        
        inventory_difference = from_stock - to_stock
        diffusion_flow = diffusion_coefficient * inventory_difference
        
        # Limit flow to not exceed target stock of receiving node
        max_allowed_flow = to_node_obj.target_stock - to_stock
        diffusion_flow = min(diffusion_flow, max_allowed_flow)
        
        return np.clip(diffusion_flow, -from_stock, to_stock)

    def get_filtered_metrics(self, filter_string='store'):
        """
        Get metrics for nodes whose names contain the filter string
        """
        filtered_nodes = {name: node for name, node in self.nodes.items() 
                         if filter_string.lower() in name.lower()}
        
        if not filtered_nodes:
            print(f"No nodes found containing '{filter_string}'")
            return None
        
        # Create DataFrame with metrics for filtered nodes
        metrics_df = pd.DataFrame(index=self.network_metrics.index)
        
        # Aggregate metrics for filtered nodes
        metrics_df['forecast_demand'] = sum(node.results['forecast_demand'] 
                                          for node in filtered_nodes.values())
        metrics_df['actual_sales'] = sum(node.results['actual_sales'] 
                                       for node in filtered_nodes.values())
        metrics_df['unused_potential'] = sum(node.results['unused_potential'] 
                                           for node in filtered_nodes.values())
        metrics_df['total_stock'] = sum(node.results['stock_level'] 
                                      for node in filtered_nodes.values())
        
        metrics_df['stockout'] = sum(node.results['stockout'] 
                                      for node in filtered_nodes.values())
        
        # Add node-specific columns
        for name, node in filtered_nodes.items():
            metrics_df[f'{name}_stock'] = node.results['stock_level']
            metrics_df[f'{name}_forecast'] = node.results['forecast_demand']
            metrics_df[f'{name}_sales'] = node.results['actual_sales']
            metrics_df[f'{name}_unused'] = node.results['unused_potential']
        
        return metrics_df
    
    def plot_filtered_results(self, filter_string='store', time_deltas=None):
        """
        Plot results for nodes whose names contain the filter string
        """
        metrics_df = self.get_filtered_metrics(filter_string)
        if metrics_df is None:
            return
        
        if time_deltas is None:
            time_deltas = range(len(metrics_df))

        plt.figure(figsize=(15, 12))
        
        # Plot 1: Aggregated Metrics
        plt.subplot(3, 1, 1)
        plt.plot(time_deltas, metrics_df['forecast_demand'], 
                label='Total Forecast Demand', color='b')
        plt.plot(time_deltas, metrics_df['actual_sales'], 
                label='Total Actual Sales', color='g')
        plt.plot(time_deltas, metrics_df['unused_potential'], 
                label='Total Unused Potential', color='r', linestyle='--')
        plt.xlabel('Days from Start')
        plt.ylabel('Units')
        plt.title(f'Aggregated Metrics for {filter_string.title()} Nodes')
        plt.legend()
        plt.grid(True)
        
        # Plot 2: Individual Stock Levels
        plt.subplot(3, 1, 2)
        stock_columns = [col for col in metrics_df.columns if 'stock' in col]
        for col in stock_columns:
            plt.plot(time_deltas, metrics_df[col], 
                    label=col.replace('_stock', ''))
        plt.xlabel('Days from Start')
        plt.ylabel('Stock Level')
        plt.title(f'Individual Stock Levels for {filter_string.title()} Nodes')
        plt.legend()
        plt.grid(True)
        
        # Plot 3: Sales vs Forecast Comparison
        plt.subplot(3, 1, 3)
        filtered_nodes = [name for name in self.nodes.keys() 
                         if filter_string.lower() in name.lower()]
        
        for node_name in filtered_nodes:
            plt.plot(time_deltas, metrics_df[f'{node_name}_forecast'], 
                    label=f'{node_name} Forecast', linestyle='--')
            plt.plot(time_deltas, metrics_df[f'{node_name}_sales'], 
                    label=f'{node_name} Actual')
        plt.xlabel('Days from Start')
        plt.ylabel('Units')
        plt.title(f'Forecast vs Actual Sales for {filter_string.title()} Nodes')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

    def _calculate_in_transit(self, node, timestamp_idx):
        """Calculate total in-transit inventory for a node at given timestamp"""
        pending_orders = sum(order[1] for order in node.pending_orders 
                            if order[0] > timestamp_idx)
        pending_flows = sum(flow[1] for flow in node.pending_flows 
                        if flow[0] > timestamp_idx)
        return pending_orders + pending_flows

    def _calculate_lead_time_performance(self):
        """Calculate lead time performance metrics for all nodes"""
        lead_times = {}
        for node_name, node in self.nodes.items():
            target_lead_time = node.replenishment_lead_time
            actual_lead_times = []
            
            for t in range(len(node.results)):
                # Calculate actual lead time based on pending orders and flows
                pending_items = node.pending_orders + node.pending_flows
                if pending_items:
                    # Only consider orders that are within reasonable timeframe
                    # (e.g., within 2x the target lead time)
                    max_reasonable_leadtime = target_lead_time * 2
                    remaining_times = []
                    for arrival_time, _ in pending_items:
                        if arrival_time > t:  # Not yet arrived
                            lead_time = arrival_time - t
                            if lead_time <= max_reasonable_leadtime:
                                remaining_times.append(lead_time)
                    
                    if remaining_times:
                        avg_lead_time = np.mean(remaining_times)
                    else:
                        avg_lead_time = target_lead_time
                    actual_lead_times.append(avg_lead_time)
                else:
                    actual_lead_times.append(target_lead_time)
            
            lead_times[node_name] = {
                'target_lead_time': target_lead_time,
                'actual_lead_times': actual_lead_times,
                'avg_lead_time': np.mean(actual_lead_times),
                'max_lead_time': np.max(actual_lead_times),
                'min_lead_time': np.min(actual_lead_times),
                'lead_time_variability': np.std(actual_lead_times)
            }
        
        return lead_times

    def get_node_metrics(self, node_name):
        """Enhanced node metrics including lead time performance"""
        if node_name not in self.nodes:
            print(f"Node '{node_name}' not found")
            return None
            
        node = self.nodes[node_name]
        lead_times = self._calculate_lead_time_performance()[node_name]
        
        return {
            'data': node.results,
            'summary': {
                'avg_forecast': node.results['forecast_demand'].mean(),
                'avg_sales': node.results['actual_sales'].mean(),
                'avg_unused': node.results['unused_potential'].mean(),
                'min_stock': node.results['stock_level'].min(),
                'max_stock': node.results['stock_level'].max(),
                'stockout_days': (node.results['stock_level'] == 0).sum(),
                'avg_in_transit': np.mean([self._calculate_in_transit(node, t) 
                                        for t in range(len(node.results))]),
                'lead_time_performance': lead_times
            }
        }

    def visualize_network(self, figsize=(14, 8), with_labels=True):
        G = nx.DiGraph()
        
        # Add nodes - keep original sizes
        for node_name, node in self.nodes.items():
            current_stock = node.results['stock_level'].iloc[-1]
            G.add_node(node_name, 
                    stock_level=current_stock,
                    type='Warehouse' if 'Warehouse' in node_name else 'Store')
        
        # Add edges
        for (from_node, to_node), edge_data in self.edges.items():
            G.add_edge(from_node, to_node, 
                    max_flow_rate=edge_data['max_flow_rate'],
                    potential=edge_data['potential'])
        
        plt.figure(figsize=figsize)
        
        # Keep original node sizes
        node_colors = ['lightblue' if 'Warehouse' in node else 'lightgreen' 
                    for node in G.nodes()]
        node_sizes = [1000 + float(G.nodes[node]['stock_level']) * 10 
                    for node in G.nodes()]
        
        # Increase spacing between nodes
        pos = nx.spring_layout(G, k=1)
        
        # Calculate edge properties
        edge_props = []
        for _, _, edge_data in G.edges(data=True):
            edge_props.append((edge_data['max_flow_rate'], edge_data['potential']))
        
        max_flow_max = max(ep[0] for ep in edge_props) if edge_props else 1
        potential_max = max(ep[1] for ep in edge_props) if edge_props else 1
        
        edge_widths = [1.0 + (ep[0] / max_flow_max) * 5 for ep in edge_props]
        edge_alphas = [0.2 + (ep[1] / potential_max) * 0.8 for ep in edge_props]
        
        # Draw edges with larger margins and more curve
        for i, (from_node, to_node) in enumerate(G.edges()):
            nx.draw_networkx_edges(G, pos,
                                edgelist=[(from_node, to_node)],
                                width=edge_widths[i],
                                alpha=edge_alphas[i],
                                edge_color='red',
                                arrows=True,
                                arrowsize=35,  # Larger arrows
                                arrowstyle='-|>',
                                connectionstyle='arc3,rad=0.1',  # More curve
                                min_source_margin=30,  # Larger margin from source
                                min_target_margin=30)  # Larger margin from target
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, 
                            node_color=node_colors,
                            node_size=node_sizes)
        
        if with_labels:
            # Node labels
            node_labels = {node: f"{node}\n(Stock: {G.nodes[node]['stock_level']:.0f})"
                        for node in G.nodes()}
            nx.draw_networkx_labels(G, pos, node_labels, font_size=8)
            
            # Edge labels with offset
            edge_labels = {(from_node, to_node): 
                        f"flow: {data['max_flow_rate']}\npot: {data['potential']:.2f}"
                        for (from_node, to_node), data in self.edges.items()}
            nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8)
        
        # Legend
        legend_elements = [
            plt.Line2D([0], [0], color='red', alpha=0.2, linewidth=2, label='Low Potential'),
            plt.Line2D([0], [0], color='red', alpha=1.0, linewidth=2, label='High Potential'),
            plt.Line2D([0], [0], color='red', linewidth=1, label='Low Max Flow'),
            plt.Line2D([0], [0], color='red', linewidth=6, label='High Max Flow'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue',
                    markersize=10, label='Warehouse'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen',
                    markersize=10, label='Store')
        ]
        
        plt.legend(handles=legend_elements, 
                loc='center left', 
                bbox_to_anchor=(1, 0.5),
                frameon=True)
        
        plt.title("Supply Chain Network Visualization\nEdge width = max flow rate, Edge color intensity = potential")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def simulate_flow(self):
        timesteps = next(iter(self.nodes.values())).timestamps
        num_steps = len(timesteps)
        
        kinetic_energy_history = []
        time_deltas = []
        initial_time = timesteps[0]
        
        for t in range(num_steps):
            if t > 0:
                # Copy previous state
                for node in self.nodes.values():
                    prev_stock = node.results.iloc[t-1]['stock_level']
                    node.results.iloc[t, node.results.columns.get_loc('stock_level')] = prev_stock
            
            if self.verbose:
                print(f"\nTimestep {t}: {timesteps[t]}")
            total_kinetic_energy = 0
            
            time_delta = (timesteps[t] - initial_time).total_seconds() / (24 * 3600)
            time_deltas.append(time_delta)
            
            # Process pending orders for all nodes
            for node in self.nodes.values():
                node.process_pending_orders(t)
                node.process_pending_flows(t)
            
            # Process actual sales for all nodes
            for node_name, node in self.nodes.items():
                actual_sales = node.actual_sales[t]
                if actual_sales > 0:
                    node.update_inventory(incoming=0, outgoing=actual_sales, timestamp_idx=t)
            
            # Handle flows based on mode
            if self.flow_mode == 'projected_stock':
                # Only use reorder point system
                for node in self.nodes.values():
                    node.check_and_place_order(t)
                    
            elif self.flow_mode == 'diffusion':
                # First handle warehouse reordering to ensure supply
                for node in self.nodes.values():
                    if node.is_warehouse:
                        node.check_and_place_order(t)
                
                # Then calculate diffusion flows
                flows_to_apply = []  # Store all flows first
                for (from_node, to_node), edge_data in self.edges.items():
                    if not self.nodes[to_node].is_warehouse:  # Only flow to stores
                        edge_potential = self.calculate_edge_potential(from_node, to_node, t)
                        max_flow_rate = edge_data['max_flow_rate']
                        
                        from_stock = self.nodes[from_node].results.iloc[t]['stock_level']
                        to_stock = self.nodes[to_node].results.iloc[t]['stock_level']
                        
                        inventory_difference = from_stock - to_stock
                        base_flow = edge_potential * inventory_difference
                        diffusion_flow = self.calculate_diffusion_flow(from_node, to_node, t)
                        total_flow = np.clip(base_flow + diffusion_flow, -max_flow_rate, max_flow_rate)
                        
                        if abs(total_flow) > 0.01:
                            lead_time = self.nodes[to_node].replenishment_lead_time
                            arrival_time = t + lead_time + 1
                            flows_to_apply.append((from_node, to_node, total_flow, arrival_time))
                
                # Apply all flows at once
                for from_node, to_node, flow, potential in flows_to_apply:
                    self.nodes[from_node].update_inventory(0, flow, t)
                    lead_time = self.nodes[to_node].replenishment_lead_time
                    arrival_time = t + lead_time + 1
                    self.nodes[to_node].update_inventory(flow, 0, t)
                    self.nodes[to_node].add_pending_flow(flow, arrival_time)
                    total_kinetic_energy += -flow * potential
                    
            elif self.flow_mode == 'hybrid':
                # First handle warehouse reordering
                for node in self.nodes.values():
                    if node.is_warehouse:
                        node.check_and_place_order(t)
                
                # Then calculate and store all flows
                flows_to_apply = []
                for (from_node, to_node), edge_data in self.edges.items():
                    to_node_obj = self.nodes[to_node]
                    if not to_node_obj.is_warehouse and to_node_obj.results.iloc[t]['stock_level'] < to_node_obj.target_stock:
                        edge_potential = self.calculate_edge_potential(from_node, to_node, t)
                        max_flow_rate = edge_data['max_flow_rate'] * 0.5  # Reduced flow rate in hybrid mode
                        
                        from_stock = self.nodes[from_node].results.iloc[t]['stock_level']
                        to_stock = to_node_obj.results.iloc[t]['stock_level']
                        
                        inventory_difference = from_stock - to_stock
                        base_flow = edge_potential * inventory_difference
                        diffusion_flow = self.calculate_diffusion_flow(from_node, to_node, t)
                        total_flow = np.clip(base_flow + diffusion_flow, -max_flow_rate, max_flow_rate)
                        
                        if abs(total_flow) > 0.01:
                            lead_time = self.nodes[to_node].replenishment_lead_time
                            arrival_time = t + lead_time + 1
                            flows_to_apply.append((from_node, to_node, total_flow, arrival_time))
                
                # Apply all flows at once
                for from_node, to_node, flow, potential in flows_to_apply:
                    self.nodes[from_node].update_inventory(0, flow, t)
                    lead_time = self.nodes[to_node].replenishment_lead_time
                    arrival_time = t + lead_time + 1
                    self.nodes[to_node].update_inventory(flow, 0, t)
                    self.nodes[to_node].add_pending_flow(flow, arrival_time)
                    total_kinetic_energy += -flow * potential
            
            # Update network-wide metrics
            self.network_metrics.iloc[t] = [
                sum(node.results.iloc[t]['forecast_demand'] for node in self.nodes.values()),
                sum(node.results.iloc[t]['actual_sales'] for node in self.nodes.values()),
                sum(node.results.iloc[t]['unused_potential'] for node in self.nodes.values()),
                total_kinetic_energy
            ]
            
            kinetic_energy_history.append(total_kinetic_energy)
        
        self.plot_results(time_deltas, kinetic_energy_history)
        

    def plot_results(self, time_deltas, kinetic_energy_history, style: str = "dark_background"):
        import seaborn as sns
        """Plot comprehensive results"""
        plt.style.use(style)
        plt.figure(figsize=(15, 18))
        ROWS = 6
        # Plot 1: System Energy
        plt.subplot(ROWS, 1, 1)
        plt.plot(time_deltas, kinetic_energy_history, label="Kinetic Energy", color='r')
        plt.xlabel('Days from Start')
        plt.ylabel('Energy')
        plt.title('System Energy Over Time')
        plt.legend()
        plt.grid(False)
        
        # Plot 2: Demand and Sales
        plt.subplot(ROWS, 1, 2)
        plt.plot(time_deltas, self.network_metrics['total_forecast_demand'], 
                label='Total Forecast Demand', color='b')
        plt.plot(time_deltas, self.network_metrics['total_actual_sales'], 
                label='Total Actual Sales', color='g')
        plt.plot(time_deltas, self.network_metrics['total_unused_potential'], 
                label='Total Unused Potential', color='r', linestyle='--')
        plt.xlabel('Days from Start')
        plt.ylabel('Units')
        plt.title('System-wide Demand and Sales')
        plt.legend()
        plt.grid(False)
        
        # Plot 3: Stock Levels
        plt.subplot(ROWS, 1, 3)
        for node_name, node in self.nodes.items():
            if node_name.startswith("Warehouse"):
                plt.plot(time_deltas, node.results['stock_level'], 
                        label=f"{node_name} Stock Level")
        plt.xlabel('Days from Start')
        plt.ylabel('Stock Level')
        plt.title('Stock Levels Over Time')
        plt.legend()
        plt.grid(False)

        plt.subplot(ROWS, 1, 4)
        for node_name, node in self.nodes.items():
            if node_name.startswith("Store"):
                plt.plot(time_deltas, node.results['stock_level'], 
                        label=f"{node_name} Stock Level")
                for idx in np.argwhere(node.results["stockout"].values > 0).reshape(-1,).tolist():
                    plt.axvline(time_deltas[idx], color="r", linestyle="--")
                    
        plt.xlabel('Days from Start')
        plt.ylabel('Stock Level')
        plt.title('Stock Levels Over Time with red dashed lines for stockouts')
        # plt.legend()
        plt.grid(False)


        plt.subplot(ROWS, 1, 5)
        for node_name, node in self.nodes.items():
            if node_name.startswith("Warehouse"):
                sns.kdeplot(node.results["stock_level"], fill=True, alpha=.3,
                            label=f"{node_name} Stock")
        plt.xlabel('Stock Level')
        plt.title('Stock Levels Density')
        plt.legend()
        plt.grid(False)

        plt.subplot(ROWS, 1, 6)
        for node_name, node in self.nodes.items():
            if node_name.startswith("Store"):
                sns.kdeplot(node.results["stock_level"], fill=True, alpha=.3,
                            label=f"{node_name} Stock")
        plt.xlabel('Stock Level')
        plt.title('Stock Levels Density')
        plt.legend()
        plt.grid(False)
        
        plt.tight_layout()
        plt.show()