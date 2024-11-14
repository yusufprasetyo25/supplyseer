import pytest
import numpy as np
from typing import Dict
from pydantic import ValidationError
import matplotlib.pyplot as plt

from supplyseer.models.statistical.stochasticprocess import DemandSimConfig, StochasticDemandProcess, DemandPathResult

@pytest.fixture
def default_config() -> Dict:
    """Fixture providing default configuration parameters"""
    return {
        "initial_demand": 600000,
        "drift": 0.05,
        "volatility": 0.1,
        "decay_rate": 0.05,
        "time_horizon": 2.0,
        "n_paths": 1000,
        "n_steps": 50,
        "random_seed": 42
    }

@pytest.fixture
def demand_process(default_config) -> StochasticDemandProcess:
    """Fixture providing a configured StochasticDemandProcess instance"""
    config = DemandSimConfig(**default_config)
    return StochasticDemandProcess(config)

class TestDemandSimConfig:
    """Tests for DemandSimConfig validation and initialization"""
    
    def test_valid_config(self, default_config):
        """Test that valid configuration parameters are accepted"""
        config = DemandSimConfig(**default_config)
        assert config.initial_demand == default_config["initial_demand"]
        assert config.drift == default_config["drift"]
        assert config.volatility == default_config["volatility"]

    @pytest.mark.parametrize("field,invalid_value", [
        ("initial_demand", -100),  # Must be positive
        ("drift", -0.05),         # Must be non-negative
        ("volatility", 0),        # Must be positive
        ("decay_rate", -0.1),     # Must be non-negative
        ("time_horizon", 0),      # Must be positive
        ("n_paths", 0),           # Must be positive
        ("n_steps", 0)            # Must be positive
    ])
    def test_invalid_config_values(self, default_config, field, invalid_value):
        """Test that invalid configuration values raise ValidationError"""
        invalid_config = default_config.copy()
        invalid_config[field] = invalid_value
        with pytest.raises(ValidationError):
            DemandSimConfig(**invalid_config)

    def test_time_steps_validation(self, default_config):
        """Test the custom validator for time steps"""
        invalid_config = default_config.copy()
        invalid_config["n_steps"] = 1000  # More than days in time horizon
        with pytest.raises(ValidationError):
            DemandSimConfig(**invalid_config)

class TestStochasticDemandProcess:
    """Tests for StochasticDemandProcess functionality"""

    def test_initialization(self, demand_process, default_config):
        """Test proper initialization of the process"""
        assert demand_process.config.initial_demand == default_config["initial_demand"]
        assert demand_process.dt == default_config["time_horizon"] / default_config["n_steps"]
        assert len(demand_process.time_grid) == default_config["n_steps"]

    def test_simulation_shape(self, demand_process, default_config):
        """Test that simulation produces correctly shaped output"""
        results = demand_process.simulate()
        assert results.paths.shape == (default_config["n_paths"], default_config["n_steps"])
        assert len(results.mean_path) == default_config["n_steps"]
        assert isinstance(results.final_distribution, dict)

    def test_simulation_boundaries(self, demand_process):
        """Test that simulation results stay within reasonable bounds"""
        results = demand_process.simulate()
        assert np.all(results.paths > 0)  # Demand should always be positive
        assert np.all(np.isfinite(results.paths))  # No infinities or NaNs

    def test_mean_path_calculation(self, demand_process):
        """Test that mean path calculation is correct"""
        results = demand_process.simulate()
        calculated_mean = np.mean(results.paths, axis=0)
        np.testing.assert_array_almost_equal(results.mean_path, calculated_mean)

    def test_final_distribution_statistics(self, demand_process):
        """Test that final distribution statistics are calculated correctly"""
        results = demand_process.simulate()
        final_values = results.paths[:, -1]
        
        assert results.final_distribution['mean'] == pytest.approx(np.mean(final_values))
        assert results.final_distribution['median'] == pytest.approx(np.median(final_values))
        assert results.final_distribution['std'] == pytest.approx(np.std(final_values))

    def test_reproducibility(self, default_config):
            """Test that simulations are reproducible with same seed"""
            # Reset numpy's global random state
            np.random.seed(default_config['random_seed'])
            
            config = DemandSimConfig(**default_config)
            process1 = StochasticDemandProcess(config)
            results1 = process1.simulate()
            
            # Reset random state again
            np.random.seed(default_config['random_seed'])
            
            process2 = StochasticDemandProcess(config)
            results2 = process2.simulate()
            
            np.testing.assert_array_equal(results1.paths, results2.paths)

    def test_quantile_paths(self, demand_process):
        """Test quantile path calculations"""
        demand_process.simulate()
        quantiles = [0.05, 0.95]
        quantile_paths = demand_process.get_quantile_paths(quantiles)
        
        assert set(quantile_paths.keys()) == set(quantiles)
        assert all(len(path) == demand_process.config.n_steps 
                  for path in quantile_paths.values())

    def test_threshold_probability(self, demand_process):
        """Test threshold probability calculations"""
        results = demand_process.simulate()
        threshold = results.final_distribution['mean']
        probs = demand_process.get_demand_threshold_probability(threshold)
        
        assert len(probs) == demand_process.config.n_steps
        assert np.all((probs >= 0) & (probs <= 1))

    @pytest.mark.mpl
    def test_plot_results(self, demand_process):
        """Test plotting functionality"""
        demand_process.simulate()
        fig, (ax1, ax2) = demand_process.plot_results()
        
        # Check that the plot was created with correct dimensions
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 2
        
        # Check basic plot properties
        assert ax1.get_xlabel() == 'Time'
        assert ax1.get_ylabel() == 'Demand Level'
        
        plt.close(fig)  # Cleanup

class TestIntegration:
    """Integration tests for the complete workflow"""

    def test_complete_workflow(self, default_config):
        """Test the complete workflow from configuration to analysis"""
        # Setup
        config = DemandSimConfig(**default_config)
        process = StochasticDemandProcess(config)
        
        # Simulation
        results = process.simulate()
        assert results is not None
        
        # Analysis
        quantile_paths = process.get_quantile_paths()
        assert len(quantile_paths) == 5  # Default quantiles
        
        threshold_probs = process.get_demand_threshold_probability(
            threshold=results.final_distribution['median']
        )
        assert len(threshold_probs) == config.n_steps
        
        # Visualization
        fig, axes = process.plot_results()
        plt.close(fig)  # Cleanup

    @pytest.mark.parametrize("n_paths", [100, 500, 2000])
    def test_simulation_scaling(self, default_config, n_paths):
        """Test how the simulation behaves with different numbers of paths"""
        config = default_config.copy()
        config["n_paths"] = n_paths
        process = StochasticDemandProcess(DemandSimConfig(**config))
        
        results = process.simulate()
        assert results.paths.shape == (n_paths, config["n_steps"])