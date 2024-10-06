import numpy as np
from src.supplyseer.bayesian.bayesian_eoq import bayesian_eoq_full, bayesian_computation, normal_pdf

def test_that_bayesian_eoq_and_computation_works():
    

    d = 100
    a = 10
    h = 1
    d_range = np.linspace(50, 250, 100)
    a_range = np.linspace(5, 25, 100)
    h_range = np.linspace(0.5, 2, 100)
    initial_d = 100
    initial_a = 10
    initial_h = 1
    n_param_values = 100
    parameter_space = "full"
    n_simulations = 1000

    normal_probability_density = normal_pdf(d_range, d, d*.1)
    assert len(normal_probability_density) == 100, "Length of normal_probability_density should be 100"
    assert np.isnan(normal_probability_density).any() == False, "normal_probability_density should not contain NaN values"

    posterior_d, posterior_a, posterior_h = bayesian_computation(d, a, h, d_range, a_range, h_range, initial_d, initial_a, initial_h)
    
    assert len(posterior_d) == 100, "Length of posterior_d should be 100"
    assert len(posterior_a) == 100, "Length of posterior_a should be 100"
    assert len(posterior_h) == 100, "Length of posterior_h should be 100"
    assert np.isnan(posterior_d).any() == False, "posterior_d should not contain NaN values"
    assert np.isnan(posterior_a).any() == False, "posterior_a should not contain NaN values"
    assert np.isnan(posterior_h).any() == False, "posterior_h should not contain NaN values"

    eoq = bayesian_eoq_full(d=d, a=a, h=h, min_d=min(d_range), max_d=max(d_range), 
                            min_a=min(a_range), max_a=max(a_range), min_h=min(h_range), max_h=max(h_range), 
                            initial_d=initial_d, initial_a=initial_a, initial_h=initial_h, 
                            n_param_values=n_param_values, parameter_space=parameter_space, n_simulations=n_simulations)
    
    assert eoq is not None, "eoq should not be None"
    assert isinstance(eoq, dict), "eoq should be a dictionary"
    assert 'bayesian_eoq_most_probable' in eoq, "eoq should contain 'bayesian_eoq_most_probable'"
    assert 'bayesian_eoq_min_least_probable' in eoq, "eoq should contain 'bayesian_eoq_min_least_probable'"
    assert 'bayesian_eoq_max_least_probable' in eoq, "eoq should contain 'bayesian_eoq_max_least_probable'"
    assert 'eoq_distribution' in eoq, "eoq should contain 'eoq_distribution'"
    assert 'eoq_credible_interval' in eoq, "eoq should contain 'eoq_credible_interval'"
    assert isinstance(eoq['bayesian_eoq_most_probable'], dict), "eoq['bayesian_eoq_most_probable'] should be a dictionary"
    assert isinstance(eoq['bayesian_eoq_min_least_probable'], dict), "eoq['bayesian_eoq_min_least_probable'] should be a dictionary"
    assert isinstance(eoq['bayesian_eoq_max_least_probable'], dict), "eoq['bayesian_eoq_max_least_probable'] should be a dictionary"
    assert isinstance(eoq['eoq_distribution'], list), "eoq['eoq_distribution'] should be a list"
    assert isinstance(eoq['eoq_credible_interval'], list), "eoq['eoq_credible_interval'] should be a list"
    assert len(eoq['eoq_distribution']) > 0, "eoq['eoq_distribution'] should not be empty"





    