import pytest
import sys
print(sys.path)
from supplyseer.gametheory.gameconfig import (Player, PlayerType, Coalition, Partition)
from supplyseer.gametheory.coopgame import (SupplyChainGame, SupplyChainAnalyzer)


@pytest.fixture
def sample_players():
    """Fixture providing a standard set of players for testing"""
    return [
        Player(
            id="S1",
            type=PlayerType.SUPPLIER,
            capacity=100.0,
            production_cost=10.0,
            holding_cost=2.0,
            setup_cost=1000.0,
            market_power=0.7
        ),
        Player(
            id="S2",
            type=PlayerType.SUPPLIER,
            capacity=80.0,
            production_cost=12.0,
            holding_cost=1.8,
            setup_cost=800.0,
            market_power=0.5
        ),
        Player(
            id="M1",
            type=PlayerType.MANUFACTURER,
            capacity=90.0,
            production_cost=15.0,
            holding_cost=3.0,
            setup_cost=2000.0,
            market_power=0.6
        ),
        Player(
            id="M2",
            type=PlayerType.MANUFACTURER,
            capacity=70.0,
            production_cost=14.0,
            holding_cost=2.8,
            setup_cost=1800.0,
            market_power=0.4
        )
    ]

@pytest.fixture
def game(sample_players):
    """Fixture providing a game instance"""
    return SupplyChainGame(sample_players)

@pytest.fixture
def analyzer(game):
    """Fixture providing an analyzer instance"""
    return SupplyChainAnalyzer(game)

class TestPlayer:
    def test_player_creation(self):
        player = Player(
            id="S1",
            type=PlayerType.SUPPLIER,
            capacity=100.0,
            production_cost=10.0,
            holding_cost=2.0,
            setup_cost=1000.0,
            market_power=0.7
        )
        assert player.id == "S1"
        assert player.type == PlayerType.SUPPLIER
        assert player.capacity == 100.0

    def test_invalid_market_power(self):
        with pytest.raises(ValueError):
            Player(
                id="S1",
                type=PlayerType.SUPPLIER,
                capacity=100.0,
                production_cost=10.0,
                holding_cost=2.0,
                setup_cost=1000.0,
                market_power=1.5  # Invalid: > 1
            )

class TestCoalition:
    def test_coalition_creation(self, sample_players):
        coalition = Coalition(members={sample_players[0], sample_players[1]})
        assert len(coalition.members) == 2
        assert coalition.value == 0.0  # Default value

    def test_coalition_equality(self, sample_players):
        c1 = Coalition(members={sample_players[0], sample_players[1]})
        c2 = Coalition(members={sample_players[1], sample_players[0]})
        assert c1 == c2  # Order shouldn't matter

class TestPartition:
    def test_partition_creation(self, sample_players):
        c1 = Coalition(members={sample_players[0], sample_players[1]})
        c2 = Coalition(members={sample_players[2], sample_players[3]})
        partition = Partition(coalitions={c1, c2})
        assert len(partition.coalitions) == 2

    def test_invalid_partition(self, sample_players):
        # Try to create partition with overlapping coalitions
        c1 = Coalition(members={sample_players[0], sample_players[1]})
        c2 = Coalition(members={sample_players[1], sample_players[2]})  # Overlapping
        with pytest.raises(ValueError):
            Partition(coalitions={c1, c2})

class TestSupplyChainGame:
    def test_game_initialization(self, game):
        assert len(game.players) == 4
        assert len(game.suppliers) == 2
        assert len(game.manufacturers) == 2

    def test_coalition_value_calculation(self, game):
        coalition = Coalition(members={next(iter(game.suppliers)), 
                                     next(iter(game.manufacturers))})
        partition = Partition(coalitions={coalition})
        value = game.calculate_coalition_value(coalition, partition)
        assert value >= 0  # Value should be non-negative

    def test_shapley_value_calculation(self, game):
        player = next(iter(game.players))
        coalition = Coalition(members=game.players)
        value = game.calculate_shapley_value(player, coalition)
        assert isinstance(value, float)
        assert value >= 0

class TestSupplyChainAnalyzer:
    def test_analyzer_initialization(self, analyzer):
        assert analyzer.game is not None

    def test_find_optimal_partition(self, analyzer):
        partition, analysis = analyzer.find_optimal_partition()
        assert partition is not None
        assert isinstance(analysis, dict)
        assert 'is_stable' in analysis

    def test_analyze_equilibrium(self, analyzer):
        # Create a simple partition for testing
        coalition = Coalition(members={next(iter(analyzer.game.suppliers)), 
                                     next(iter(analyzer.game.manufacturers))})
        partition = Partition(coalitions={coalition})

        analysis = analyzer.analyze_equilibrium(partition)
        assert 'is_nash_equilibrium' in analysis
        assert 'improvement_potentials' in analysis

    def test_simulation(self, analyzer):
        results = analyzer.simulate_evolution(num_periods=5)
        assert 'total_value' in results
        assert 'stability' in results
        assert len(results['total_value']) == 5
        assert len(results['stability']) == 5