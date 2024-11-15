from typing import Dict, List, Set, Tuple, Optional, Union
import numpy as np
from pydantic import BaseModel, Field, validator
from enum import Enum
from supplyseer.gametheory.gameconfig import (Player, PlayerType, Coalition, Partition)

class SupplyChainGame:
    def __init__(self, players: List[Player]):
        self.players = set(players)
        self.suppliers = {p for p in players if p.type == PlayerType.SUPPLIER}
        self.manufacturers = {p for p in players if p.type == PlayerType.MANUFACTURER}
        self.retailers = {p for p in players if p.type == PlayerType.RETAILER}
        self.history: List[Dict] = []
        
    def calculate_coalition_value(self, coalition: Coalition, partition: Partition) -> float:
        """Calculate value of coalition given partition configuration"""
        suppliers = {p for p in coalition.members if p.type == PlayerType.SUPPLIER}
        manufacturers = {p for p in coalition.members if p.type == PlayerType.MANUFACTURER}
        
        # Base production capacity
        capacity = min(
            sum(s.capacity for s in suppliers) if suppliers else float('inf'),
            sum(m.capacity for m in manufacturers) if manufacturers else float('inf')
        )
        
        # Production costs
        prod_cost = sum(s.production_cost for s in suppliers)
        prod_cost += sum(m.production_cost for m in manufacturers)
        
        # Market power factor
        market_power = max((p.market_power for p in coalition.members), default=0)
        
        # Impact of partition structure
        competing_power = max(
            (c.market_power for c in partition.coalitions if c != coalition),
            default=0
        )
        
        # Final value calculation incorporating partition effects
        value = (capacity * market_power * (1 - 0.5 * competing_power) - prod_cost)
        return max(value, 0)

    def calculate_shapley_value(self, player: Player, coalition: Coalition) -> float:
        """Calculate Shapley value for player in coalition"""
        n = len(coalition.members)
        shapley_value = 0.0
        
        for subset in self._get_subsets(coalition.members - {player}):
            subset_value = self.calculate_coalition_value(
                Coalition(members=subset),
                self._get_partition_with_subset(subset)
            )
            subset_with_player = self.calculate_coalition_value(
                Coalition(members=subset | {player}),
                self._get_partition_with_subset(subset | {player})
            )
            
            weight = (len(subset) * np.math.factorial(n-len(subset)-1) * 
                     np.math.factorial(len(subset))) / np.math.factorial(n)
                     
            shapley_value += weight * (subset_with_player - subset_value)
            
        return shapley_value

    def is_core_stable(self, partition: Partition) -> bool:
        """Check if partition configuration is core stable"""
        total_value = sum(c.value for c in partition.coalitions)
        
        for potential_coalition in self._get_all_potential_coalitions():
            if potential_coalition in partition.coalitions:
                continue
                
            potential_value = self.calculate_coalition_value(
                potential_coalition,
                Partition(coalitions=partition.coalitions | {potential_coalition})
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