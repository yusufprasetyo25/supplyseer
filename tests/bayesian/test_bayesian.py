import numpy as np
from supplyseer.bayesian.bayesian_eoq import (
    BayesianDistribution,
    bayesian_eoq_full
)

def test_that_bayesian_eoq_and_computation_works():
    """
    Test the complete Bayesian EOQ computation workflow including probability calculations,
    posterior computations, and final EOQ calculations.
    """
    # Test parameters
    d = 100
    a = 10
    h = 1
    d_min = 50
    d_max = 250
    a_min = 5
    a_max = 25
    h_min = 0.5
    h_max = 2
    initial_d = 100
    initial_a = 10
    initial_h = 1
    n_param_values = 100
    parameter_space = "full"
    n_simulations = 1000


    # Test normal PDF calculation
    d_range = np.linspace(d_min, d_max, n_param_values)
    normal_probability_density = BayesianDistribution.normal_pdf(d_range, d, d*.1)
    assert len(normal_probability_density) == 100, "Length of normal_probability_density should be 100"
    assert not np.isnan(normal_probability_density).any(), "normal_probability_density should not contain NaN values"

    # Test bayesian distribution
    bayesian_calc = BayesianDistribution(
        empirical=d,
        prior=initial_d,
        min=d_min,
        max=d_max,
        num_points=n_param_values,
    )

    # Check posterior distributions
    assert len(bayesian_calc.calculate_parameter_ranges()) == 100, "Length of posterior_d should be 100"
    assert not np.isnan(bayesian_calc.calculate_parameter_ranges()).any(), "posterior_d should not contain NaN values"

    # Test full EOQ calculation
    eoq = bayesian_eoq_full(
        d=d,
        a=a,
        h=h,
        min_d=d_min,
        max_d=d_max,
        min_a=a_min,
        max_a=a_max,
        min_h=h_min,
        max_h=h_max,
        initial_d=initial_d,
        initial_a=initial_a,
        initial_h=initial_h,
        n_param_values=n_param_values,
        parameter_space=parameter_space,
        n_simulations=n_simulations
    )
    
    # Check EOQ result structure
    assert eoq is not None, "eoq should not be None"
    assert isinstance(eoq, dict), "eoq should be a dictionary"
    
    # Check required keys
    required_keys = [
        'bayesian_eoq_most_probable',
        'bayesian_eoq_min_least_probable',
        'bayesian_eoq_max_least_probable',
        'eoq_distribution',
        'eoq_credible_interval'
    ]
    for key in required_keys:
        assert key in eoq, f"eoq should contain '{key}'"
    
    # Check types of result components
    assert isinstance(eoq['bayesian_eoq_most_probable'], dict), \
        "eoq['bayesian_eoq_most_probable'] should be a dictionary"
    assert isinstance(eoq['bayesian_eoq_min_least_probable'], dict), \
        "eoq['bayesian_eoq_min_least_probable'] should be a dictionary"
    assert isinstance(eoq['bayesian_eoq_max_least_probable'], dict), \
        "eoq['bayesian_eoq_max_least_probable'] should be a dictionary"
    assert isinstance(eoq['eoq_distribution'], list), \
        "eoq['eoq_distribution'] should be a list"
    assert isinstance(eoq['eoq_credible_interval'], list), \
        "eoq['eoq_credible_interval'] should be a list"
    
    # Check distribution is not empty
    assert len(eoq['eoq_distribution']) > 0, \
        "eoq['eoq_distribution'] should not be empty"
    
    # Check values are reasonable
    assert all(v > 0 for v in eoq['eoq_distribution']), \
        "All EOQ values should be positive"
    assert all(v > 0 for v in eoq['eoq_credible_interval']), \
        "All credible interval values should be positive"
    assert eoq['bayesian_eoq_most_probable']['eoq'] > 0, \
        "Most probable EOQ should be positive"
