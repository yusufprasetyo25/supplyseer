from typing import Dict, List, Set, Tuple, Optional, Union
import numpy as np
from pydantic import BaseModel, Field, validator
from enum import Enum
from gameconfig import (Player, PlayerType, Coalition, Partition)
from itertools import combinations

class SupplyChainGame:
    def __init__(self, players: List[Player]):
        self.players = set(players)
        self.suppliers = {p for p in players if p.type == PlayerType.SUPPLIER}
        self.manufacturers = {p for p in players if p.type == PlayerType.MANUFACTURER}
        self.retailers = {p for p in players if p.type == PlayerType.RETAILER}
        self.history: List[Dict] = []


    def _get_all_potential_coalitions(self) -> List[Coalition]:
        potential_coalitions = []
        for i in range(1, len(self.players) + 1):
            for combo in combinations(self.players, i):
                potential_coalitions.append(Coalition(members=set(combo)))
        return potential_coalitions

    def _are_coalitions_disjoint(self, coalitions: Set[Coalition]) -> bool:
        all_players = set()
        for coalition in coalitions:
            member_ids = {p.id for p in coalition.members}
            if any(pid in all_players for pid in member_ids):
                return False
            all_players.update(member_ids)
        return True
    
    def _calculate_gc_value(self, coalition: Coalition, base_demand: float, alpha: float,
                        supplier_costs: float, manufacturer_costs: float,
                        fixed_costs: float) -> float:
        """
        Calculate grand coalition value based on equations (11)-(12)
        """
        # Get suppliers and manufacturers from coalition
        suppliers = {p for p in coalition.members if p.type == PlayerType.SUPPLIER}
        manufacturers = {p for p in coalition.members if p.type == PlayerType.MANUFACTURER}
        
        # Divide costs by capacity to get per-unit costs
        capacity = min(sum(s.capacity for s in suppliers),
                    sum(m.capacity for m in manufacturers))
        unit_costs = (supplier_costs + manufacturer_costs) / capacity if capacity > 0 else 0
        
        # Optimal price from equation (11)
        p_star = (base_demand/(2*alpha)) + (unit_costs/2)
        
        # Calculate value using equation (12) 
        value = ((base_demand - alpha*p_star)*(p_star - unit_costs)) - fixed_costs
        return value

    def _calculate_vc_value(self, coalition: Coalition, base_demand: float, alpha: float,
                            supplier_costs: float, manufacturer_costs: float,
                            fixed_costs: float, partition: Partition) -> float:
        """
        Calculate vertical cooperation value based on equations (24)-(25)
        U_Vi = (D_Mi(p_i - C_Mi - C_S) + F_M-i D_M-i(q - C_S) - O_S - O_Mi)F_Vi
        """
        # Get suppliers and manufacturers from coalition
        suppliers = {p for p in coalition.members if p.type == PlayerType.SUPPLIER}
        manufacturers = {p for p in coalition.members if p.type == PlayerType.MANUFACTURER}
        
        capacity = min(sum(s.capacity for s in suppliers),
                    sum(m.capacity for m in manufacturers))
        unit_costs = (supplier_costs + manufacturer_costs) / capacity if capacity > 0 else 0

        other_coalitions = [c for c in partition.coalitions if c != coalition]
        competing_manuf = next(
            (p for c in other_coalitions for p in c.members
            if p.type == PlayerType.MANUFACTURER),
            None
        )

        if competing_manuf:
            # Handle competition case with equations (51)
            p_star = (base_demand + alpha*unit_costs)/(2*alpha)
            q_star = (base_demand + alpha*unit_costs)/(4*alpha)
            
            # Calculate demands
            D_Mi = base_demand - alpha*p_star
            D_M_other = base_demand - alpha*q_star
            
            # Value includes both direct sales and supplier revenue from competitor
            value = (D_Mi*(p_star - unit_costs) + 
                    D_M_other*(q_star - supplier_costs/capacity)) - fixed_costs
        else:
            # Without competition
            p_star = base_demand/(2*alpha) + unit_costs/2
            q_star = base_demand/(4*alpha) + unit_costs/2
            
            # Only direct sales in value calculation
            value = (base_demand - alpha*p_star)*(p_star - unit_costs) - fixed_costs

        return value

    def _calculate_alc_value(self, coalition: Coalition, base_demand: float, alpha: float,
                            supplier_costs: float, manufacturer_costs: float, 
                            fixed_costs: float, partition: Partition) -> float:
        """
        Calculate ALC (All Alone Case) value based on equations (3)-(4)
        U_Mi = ((D_Mi(a_M)(p_i - C_Mi - q)F_S - O_Mi)F_Mi  # For manufacturer
        U_S = (ΣD_Mi(a_M)F_Mi)(q - C_S) - O_S)F_S  # For supplier
        """
        suppliers = {p for p in coalition.members if p.type == PlayerType.SUPPLIER}
        manufacturers = {p for p in coalition.members if p.type == PlayerType.MANUFACTURER}
        
        capacity = min(sum(s.capacity for s in suppliers), 
                    sum(m.capacity for m in manufacturers))
        unit_costs = (supplier_costs + manufacturer_costs) / capacity if capacity > 0 else 0

        # From Theorem 1
        p_star = base_demand/(2*alpha) + unit_costs
        q_star = base_demand/(4*alpha) + unit_costs/2

        # Calculate demand
        D_M = base_demand - alpha*p_star

        if any(p.type == PlayerType.SUPPLIER for p in coalition.members):
            # Supplier's utility from equation (4)
            value = D_M*(q_star - supplier_costs/capacity) - fixed_costs
        else:
            # Manufacturer's utility from equation (3)
            value = D_M*(p_star - manufacturer_costs/capacity - q_star) - fixed_costs

        return value
    
    # TODO: investigate why this produces very large values
    # def calculate_coalition_value(self, coalition: Coalition, partition: Partition) -> float:
    #     """Calculate coalition value combining practical and theoretical calculations"""
    #     if not coalition.members:
    #         return 0.0

    #     suppliers = {p for p in coalition.members if p.type == PlayerType.SUPPLIER}
    #     manufacturers = {p for p in coalition.members if p.type == PlayerType.MANUFACTURER}

    #     # Base capacity checks
    #     supplier_capacity = sum(s.capacity for s in suppliers) if suppliers else 0.0
    #     manufacturer_capacity = sum(m.capacity for m in manufacturers) if manufacturers else 0.0

    #     if supplier_capacity == 0.0 or manufacturer_capacity == 0.0:
    #         return 0.0

    #     capacity = min(supplier_capacity, manufacturer_capacity)

    #     # Market power calculations
    #     market_power = max((p.market_power for p in coalition.members), default=0.0)
    #     market_power = max(0.0, min(1.0, market_power))

    #     # Competing power calculation
    #     competing_coalitions = [c for c in partition.coalitions if c != coalition and c.members]
    #     if competing_coalitions:
    #         competing_power = max(
    #             max((p.market_power for p in c.members), default=0.0)
    #             for c in competing_coalitions
    #         )
    #         competing_power = max(0.0, min(1.0, competing_power))
    #     else:
    #         competing_power = 0.0

    #     # Calculate base parameters for theoretical calculations
    #     base_demand = capacity * market_power * 100  # Scale demand by capacity and market power
    #     alpha = 0.7  # Demand sensitivity parameter (can be tuned)
    #     supplier_costs = sum(s.production_cost for s in suppliers)
    #     manufacturer_costs = sum(m.production_cost for m in manufacturers)
    #     fixed_costs = sum(p.setup_cost for p in coalition.members)

    #     # Calculate theoretical value based on coalition type
    #     if len(partition.coalitions) == 1:  # Grand coalition
    #         print("Coalition type is Grand Coalition (GC)")
    #         theoretical_value = self._calculate_gc_value(coalition, base_demand, alpha,
    #                                                 supplier_costs, manufacturer_costs,
    #                                                 fixed_costs)
    #     elif len(suppliers) == 1 and len(manufacturers) == 1:  # Vertical cooperation
    #         print("Coalition type is Vertical Cooperation (VC)")
    #         theoretical_value = self._calculate_vc_value(coalition, base_demand, alpha,
    #                                                 supplier_costs, manufacturer_costs,
    #                                                 fixed_costs, partition)
    #     else:  # All alone case
    #         print("Coalition type is All Alone Case (ALC)")
    #         theoretical_value = self._calculate_alc_value(coalition, base_demand, alpha,
    #                                                     supplier_costs, manufacturer_costs,
    #                                                     fixed_costs, partition)

    #     # Calculate practical value
    #     practical_value = capacity * market_power * (1.0 - 0.5 * competing_power)

    #     # Combine theoretical and practical values with scaling
    #     THEORY_WEIGHT = 0.3  # Can be tuned
    #     SCALING_FACTOR = 0.001  # Scale theoretical value to practical range

    #     print(f"Practical value: {practical_value}")
    #     combined_value = (
    #         (1 - THEORY_WEIGHT) * practical_value + 
    #         THEORY_WEIGHT * theoretical_value * SCALING_FACTOR
    #     ) - fixed_costs
    #     print(f"Combined value: {combined_value}")

    #     return max(0.0, combined_value)

    
    def calculate_coalition_value(self, coalition: Coalition, partition: Partition) -> float:
        """
        Calculate value of coalition given partition configuration with improved stability
        "Partition-form Cooperative Games" - Wadhwa et al, in section 2 they discuss:
            - Coalition formation requires both suppliers and manufacturers
            - Capacity is limited by minimum of supplier and manufacturer capacities

            Section 5:
            - Production costs are summed across coalition members
            - Market power influences coalition worth
        """
        if not coalition.members:
            return 0.0

        suppliers = {p for p in coalition.members if p.type == PlayerType.SUPPLIER}
        manufacturers = {p for p in coalition.members if p.type == PlayerType.MANUFACTURER}

        # Base production capacity
        supplier_capacity = sum(s.capacity for s in suppliers) if suppliers else 0.0
        manufacturer_capacity = sum(m.capacity for m in manufacturers) if manufacturers else 0.0

        if supplier_capacity == 0.0 or manufacturer_capacity == 0.0:
            return 0.0  # No value without both suppliers and manufacturers

        capacity = min(supplier_capacity, manufacturer_capacity)

        # Production costs
        prod_cost = sum(s.production_cost for s in suppliers)
        prod_cost += sum(m.production_cost for m in manufacturers)

        # Market power factor (with safety checks)
        market_power = max((p.market_power for p in coalition.members), default=0.0)
        market_power = max(0.0, min(1.0, market_power))  # Ensure between 0 and 1

        # Competing power calculation with safety checks
        competing_coalitions = [c for c in partition.coalitions if c != coalition and c.members]
        if competing_coalitions:
            competing_power = max(
                max((p.market_power for p in c.members), default=0.0)
                for c in competing_coalitions
            )
            competing_power = max(0.0, min(1.0, competing_power))
        else:
            competing_power = 0.0

        # Final value calculation
        multiplier = (1.0 - 0.5 * competing_power)
        value = (capacity * market_power * multiplier) - prod_cost

        return max(0.0, value)

    def calculate_shapley_value(self, player: Player, coalition: Coalition) -> float:
        """
        Calculate Shapley value for player in coalition with improved numerical stability
        """
        if not coalition.members or player not in coalition.members:
            return 0.0

        n = len(coalition.members)
        if n == 1:
            # If player is alone in coalition, they get the full value
            return self.calculate_coalition_value(coalition, 
                Partition(coalitions={coalition}))

        shapley_value = 0.0
        other_players = list(coalition.members - {player})

        # Calculate contribution for each possible ordering
        num_permutations = 0
        for subset_size in range(len(other_players) + 1):
            for subset in combinations(other_players, subset_size):
                subset = set(subset)

                # Calculate value without and with the player
                coalition_without = Coalition(members=subset)
                coalition_with = Coalition(members=subset | {player})

                # Create simple partitions for value calculation
                remaining_players = coalition.members - subset - {player}
                if remaining_players:
                    partition_without = Partition(coalitions={
                        coalition_without, 
                        Coalition(members=remaining_players)
                    })
                    partition_with = Partition(coalitions={
                        coalition_with,
                        Coalition(members=remaining_players - {player})
                    })
                else:
                    partition_without = Partition(coalitions={coalition_without})
                    partition_with = Partition(coalitions={coalition_with})

                # Calculate marginal contribution
                value_without = self.calculate_coalition_value(coalition_without, partition_without)
                value_with = self.calculate_coalition_value(coalition_with, partition_with)
                marginal_contribution = value_with - value_without

                # Calculate weight more safely
                subset_size = len(subset)
                if n <= 1:
                    weight = 1.0
                else:
                    # Use a more stable weight calculation
                    weight = 1.0 / n

                shapley_value += weight * marginal_contribution
                num_permutations += 1

        if num_permutations > 0:
            shapley_value = shapley_value / num_permutations

        return max(0.0, shapley_value)
    

    def is_core_stable(self, partition: Partition) -> bool:
        total_value = sum(c.value for c in partition.coalitions)
        
        for potential_coalition in self._get_all_potential_coalitions():
            if potential_coalition in partition.coalitions:
                continue
                
            new_coalitions = partition.coalitions | {potential_coalition}
            partition_with_potential = (Partition(coalitions=new_coalitions) 
                            if self._are_coalitions_disjoint(new_coalitions) 
                            else partition)
                            
            potential_value = self.calculate_coalition_value(
                potential_coalition,
                partition_with_potential
            )
            
            current_value = sum(
                c.value for c in partition.coalitions
                if any(p in potential_coalition.members for p in c.members)
            )
            
            if potential_value > current_value:
                return False
                
        return True

    def find_stable_partitions(self) -> List[Partition]:
        """Find all stable partition configurations"""
        stable_partitions = []
        
        for partition in self._generate_all_partitions():
            # Calculate values
            for coalition in partition.coalitions:
                coalition.value = self.calculate_coalition_value(coalition, partition)
            
            # Check stability
            if self.is_core_stable(partition):
                stable_partitions.append(partition)
                
        return stable_partitions

    def simulate_period(self, partition: Partition) -> Dict[str, float]:
        """Simulate one time period with given partition structure"""
        metrics = {}

        for coalition in partition.coalitions:
            # Calculate actual contributions
            for player in coalition.members:
                contribution = player.capacity * np.random.normal(1, 0.1)
                coalition.contributions[player] = contribution

                # Update market power based on contribution
                expected = player.capacity
                player.market_power = 0.9 * player.market_power + 0.1 * (contribution / expected)

            # Update coalition stability
            shapley_values = {
                p: self.calculate_shapley_value(p, coalition)
                for p in coalition.members
            }

            # Modified stability calculation to handle zero Shapley values
            stability_terms = []
            for p in coalition.members:
                if shapley_values[p] == 0:
                    if coalition.contributions[p] == 0:
                        # Both Shapley value and contribution are 0
                        stability_terms.append(1.0)
                    else:
                        # Shapley value is 0 but contribution isn't
                        stability_terms.append(0.0)
                else:
                    # Normal case
                    stability_terms.append(
                        1 - abs(shapley_values[p] - coalition.contributions[p])/shapley_values[p]
                    )

            stability = np.mean(stability_terms)
            coalition.stability_index = stability

        self.history.append({
            'partition': partition,
            'total_value': sum(c.value for c in partition.coalitions),
            'avg_stability': np.mean([c.stability_index for c in partition.coalitions])
        })

        return metrics

    def _get_subsets(self, players: Set[Player]) -> List[Set[Player]]:
        """Generate all possible subsets of players"""
        subsets = []
        for i in range(2**len(players)):
            subset = set()
            for j, player in enumerate(players):
                if i & (1 << j):
                    subset.add(player)
            subsets.append(subset)
        return subsets

    def _get_partition_with_subset(self, subset: Set[Player]) -> Partition:
        """Get partition configuration with given subset as coalition"""
        remaining = self.players - subset
        return Partition(coalitions={
            Coalition(members=subset),
            Coalition(members=remaining)
        })

    def _generate_all_partitions(self) -> List[Partition]:
        """Generate all possible partition configurations"""
        def generate_partition(remaining_players: Set[Player], current_partition: Set[Coalition]) -> List[Partition]:
            if not remaining_players:
                return [Partition(coalitions=current_partition)]
            
            partitions = []
            player = next(iter(remaining_players))
            
            # Add to existing coalition
            for coalition in current_partition:
                new_partition = current_partition - {coalition}
                new_coalition = Coalition(members=coalition.members | {player})
                new_partition.add(new_coalition)
                partitions.extend(
                    generate_partition(remaining_players - {player}, new_partition)
                )
                
            # Create new coalition
            new_partition = current_partition | {Coalition(members={player})}
            partitions.extend(
                generate_partition(remaining_players - {player}, new_partition)
            )
                
            return partitions
            
        return generate_partition(self.players, set())

class SupplyChainAnalyzer:
    def __init__(self, game: SupplyChainGame):
        self.game = game
        
    def analyze_partition(self, partition: Partition) -> Dict:
        """Analyze partition configuration stability and efficiency"""
        analysis = {
            'is_stable': self.game.is_core_stable(partition),
            'total_value': sum(c.value for c in partition.coalitions),
            'coalition_stats': []
        }
        
        for coalition in partition.coalitions:
            stats = {
                'size': len(coalition.members),
                'value': coalition.value,
                'stability': coalition.stability_index,
                'composition': {
                    'suppliers': len([p for p in coalition.members if p.type == PlayerType.SUPPLIER]),
                    'manufacturers': len([p for p in coalition.members if p.type == PlayerType.MANUFACTURER]),
                    'retailers': len([p for p in coalition.members if p.type == PlayerType.RETAILER])
                },
                'shapley_values': {
                    p.id: self.game.calculate_shapley_value(p, coalition)
                    for p in coalition.members
                }
            }
            analysis['coalition_stats'].append(stats)
            
        return analysis
    
    def check_nash_equilibrium(self, partition: Partition) -> Tuple[bool, Dict[str, float]]:
        """
        Check if partition is a Nash equilibrium and return improvement potentials
        """
        improvement_potentials = {}
        is_nash = True

        for coalition in partition.coalitions:
            for player in coalition.members:
                # Calculate current payoff
                current_payoff = self.game.calculate_shapley_value(player, coalition)

                # Check all alternative coalitions
                max_alternative_payoff = current_payoff

                for alt_coalition in partition.coalitions:
                    if alt_coalition == coalition:
                        continue

                    # Create temporary coalition with player switched
                    new_coalition = Coalition(
                        members=alt_coalition.members | {player}
                    )

                    # Create new partition configuration
                    new_coalitions = set()
                    for c in partition.coalitions:
                        if c == coalition:
                            if len(c.members) > 1:
                                new_coalitions.add(Coalition(members=c.members - {player}))
                        elif c == alt_coalition:
                            new_coalitions.add(new_coalition)
                        else:
                            new_coalitions.add(c)

                    new_partition = Partition(coalitions=new_coalitions)

                    # Calculate potential payoff in new configuration
                    potential_payoff = self.game.calculate_shapley_value(player, new_coalition)

                    if potential_payoff > max_alternative_payoff:
                        max_alternative_payoff = potential_payoff

                # Calculate improvement potential
                improvement = max_alternative_payoff - current_payoff
                improvement_potentials[player.id] = improvement

                if improvement > 0:
                    is_nash = False

        return is_nash, improvement_potentials

    def analyze_equilibrium(self, partition: Partition) -> Dict:
        """
        Comprehensive equilibrium analysis
        """
        is_nash, improvements = self.check_nash_equilibrium(partition)

        analysis = {
            'is_nash_equilibrium': is_nash,
            'improvement_potentials': improvements,
            'coalition_payoffs': {},
            'individual_payoffs': {},
            'stability_metrics': {
                'average_improvement_potential': np.mean(list(improvements.values())),
                'max_improvement_potential': max(improvements.values()),
                'players_with_incentive_to_deviate': sum(1 for v in improvements.values() if v > 0)
            }
        }

        # Calculate payoffs for each coalition
        for coalition in partition.coalitions:
            coalition_value = self.game.calculate_coalition_value(coalition, partition)
            analysis['coalition_payoffs'][str(sorted([p.id for p in coalition.members]))] = coalition_value

            # Calculate individual payoffs using Shapley value
            for player in coalition.members:
                shapley = self.game.calculate_shapley_value(player, coalition)
                analysis['individual_payoffs'][player.id] = shapley

        return analysis
        

    def find_optimal_partition(self) -> Tuple[Partition, Dict]:
        """Find partition configuration maximizing total value while maintaining stability"""
        stable_partitions = self.game.find_stable_partitions()
        
        if not stable_partitions:
            return None, {}
            
        best_partition = max(
            stable_partitions,
            key=lambda p: sum(c.value for c in p.coalitions)
        )
        
        return best_partition, self.analyze_partition(best_partition)

    def analyze_market_power_distribution(self) -> Dict:
        """Analyze distribution of market power across players and coalitions"""
        analysis = {
            'by_type': {
                PlayerType.SUPPLIER: [],
                PlayerType.MANUFACTURER: [],
                PlayerType.RETAILER: []
            },
            'by_coalition': []
        }
        
        for player in self.game.players:
            analysis['by_type'][player.type].append(player.market_power)
            
        stable_partitions = self.game.find_stable_partitions()
        if stable_partitions:
            partition = stable_partitions[0]
            for coalition in partition.coalitions:
                analysis['by_coalition'].append({
                    'size': len(coalition.members),
                    'total_power': sum(p.market_power for p in coalition.members),
                    'avg_power': np.mean([p.market_power for p in coalition.members])
                })
                
        return analysis

    def simulate_evolution(self, num_periods: int) -> Dict:
        """Simulate evolution of coalition structures over time"""
        results = {
            'total_value': [],
            'stability': [],
            'market_power': []
        }
        
        partition = self.game.find_stable_partitions()[0]
        
        for _ in range(num_periods):
            metrics = self.game.simulate_period(partition)
            
            results['total_value'].append(
                sum(c.value for c in partition.coalitions)
            )
            results['stability'].append(
                np.mean([c.stability_index for c in partition.coalitions])
            )
            results['market_power'].append(
                {p.id: p.market_power for p in self.game.players}
            )
            
        return results