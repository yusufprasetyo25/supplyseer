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
        
    def calculate_coalition_value(self, coalition: Coalition, partition: Partition) -> float:
        """Calculate value of coalition given partition configuration with improved stability"""
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
            
            stability = np.mean([
                1 - abs(shapley_values[p] - coalition.contributions[p])/shapley_values[p]
                for p in coalition.members
            ])
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