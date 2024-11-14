import numpy as np
from supplyseer.eoq import eoq
from typing import List, Union, Optional, Dict


def normal_pdf(x, mean, std):
    """
    Normal probability density function

    Args:
        x: data
        mean: mean of the distribution
        std: standard deviation of the distribution

    Returns:
        float
    """
    return (1/np.sqrt(2 * np.pi * std**2)) * np.exp(-.5 * ((x - mean)**2 ) / std**2)



def bayesian_computation(d, a, h,
                         d_range, a_range, h_range,
                         belief_d, belief_a, belief_h):
    """
    Computes the posterior distribution of the EOQ parameters using a Bayesian approach.
    Naive assumptions are made for the prior and likelihood functions assuming normal distributions.

    Args:
        d: demand
        a: ordering cost
        h: holding cost
        d_range: range of demand values
        a_range: range of ordering cost values
        h_range: range of holding cost values
        belief_d: initial guess for demand
        belief_a: initial guess for ordering cost
        belief_h: initial guess for holding cost

    Returns:
        tuple: posterior of demand, posterior of order cost, posterior of holding cost
    """
    def bayesian_posterior(x, parameter_range, initial_guess):
        """
        Calculates the posterior distribution of a parameter using a Bayesian approach.
        """
        prior = normal_pdf(parameter_range, x, x*.1)
        likelihood_pdf = normal_pdf(initial_guess, parameter_range, x*.1)
        unnormalized_posterior = prior * likelihood_pdf
        marginal_likelihood = np.trapz(unnormalized_posterior, parameter_range)
        posterior = unnormalized_posterior / marginal_likelihood

        return posterior
    
    posterior_d = bayesian_posterior(d, d_range, belief_d) # Normalized posterior
    posterior_a = bayesian_posterior(a, a_range, belief_a) # Normalized posterior
    posterior_h = bayesian_posterior(h, h_range, belief_h) # Normalized posterior

    return posterior_d, posterior_a, posterior_h



def eoq_credible_interval(posterior_d, posterior_a, posterior_h, d_range, a_range, h_range):
    """
    Computes the credible interval of the EOQ using the posterior distribution of the parameters.
    """

    quantiles = [0.025, 0.975]

    indices_d = [np.argmin(np.abs(posterior_d - np.quantile(posterior_d, q))) for q in quantiles]
    indices_a = [np.argmin(np.abs(posterior_a - np.quantile(posterior_a, q))) for q in quantiles]
    indices_h = [np.argmin(np.abs(posterior_h - np.quantile(posterior_h, q))) for q in quantiles]

    interval_d = [d_range[i] for i in indices_d]
    interval_a = [a_range[i] for i in indices_a]
    interval_h = [h_range[i] for i in indices_h]

    eoq_credible_interval = [eoq(interval_d[i], interval_a[j], interval_h[k]) for i in range(len(interval_d)) for j in range(len(interval_a)) for k in range(len(interval_h))]
    return eoq_credible_interval


def bayesian_eoq_full(d: Union[int, float], a: Union[int, float], h: Union[int, float],
                 min_d: Union[int, float], max_d: Union[int, float], min_a: Union[int, float], 
                 max_a: Union[int, float], min_h: Union[int, float], max_h: Union[int, float],
                 initial_d: Union[int, float], initial_a: Union[int, float], initial_h: Union[int, float],
                 n_param_values: int, parameter_space: Optional[str] = 'full', n_simulations: Optional[int] = 1):
    
    """
    This function computes the EOQ distribution and the Bayesian credible interval of the EOQ using a Bayesian approach.
    The EOQ distribution is using a combinatorial approach to calculate the EOQ for all possible combinations of the parameters.
    The Bayesian credible interval is calculated using the posterior distribution of the parameters.

    Args:
        d: demand
        a: ordering cost
        h: holding cost
        min_d: minimum demand
        max_d: maximum demand
        min_a: minimum ordering cost
        max_a: maximum ordering cost
        min_h: minimum holding cost
        max_h: maximum holding cost
        initial_d: initial guess for demand
        initial_a: initial guess for ordering cost
        initial_h: initial guess for holding cost
        n_param_values: number of parameter values to consider
    """
    d_range = np.linspace(min_d, max_d, n_param_values)
    a_range = np.linspace(min_a, max_a, n_param_values)
    h_range = np.linspace(min_h, max_h, n_param_values)

    posterior_d, posterior_a, posterior_h = bayesian_computation(d, a, h, 
                                                                 d_range, a_range, h_range, 
                                                                 initial_d, initial_a, initial_h)


    eoq_map = eoq(d_range[np.argmax(posterior_d)],
                      a_range[np.argmax(posterior_a)],
                      h_range[np.argmax(posterior_h)])
    
    eoq_min_lower = eoq(min_d,
                        min_a,
                        min_h)

    eoq_min_upper = eoq(max_d,
                        max_a,
                        max_h)
    
    eoq_distribution = [eoq(d_range[i], a_range[j], h_range[k]) for i in range(len(d_range)) for j in range(len(a_range)) for k in range(len(h_range))]

    if parameter_space == 'full':
        eoq_distribution = [eoq(d_range[i], a_range[j], h_range[k]) for i in range(len(d_range)) for j in range(len(a_range)) for k in range(len(h_range))]
    elif parameter_space == 'montecarlo':
        eoq_montecarlo = eoq(d_range, a_range, h_range)
        eoq_distribution = np.random.choice(eoq_montecarlo, size=n_simulations, replace=True)
    else:
        eoq_distribution = []
    

    eoq_credible = eoq_credible_interval(posterior_d, posterior_a, posterior_h, d_range, a_range, h_range)

    output = {
        'bayesian_eoq_most_probable': {
            'eoq': eoq_map,
            'd': d_range[np.argmax(posterior_d)],
            'a': a_range[np.argmax(posterior_a)],
            'h': h_range[np.argmax(posterior_h)]
        },
        'bayesian_eoq_min_least_probable': {
            'eoq': eoq_min_lower,
            'd': min_d,
            'a': min_a,
            'h': min_h
        },
        'bayesian_eoq_max_least_probable': {
            'eoq': eoq_min_upper,
            'd': max_d,
            'a': max_a,
            'h': max_h
        },
        'eoq_distribution': eoq_distribution,
        'eoq_credible_interval': eoq_credible}

    return output