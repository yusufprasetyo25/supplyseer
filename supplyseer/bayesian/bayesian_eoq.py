from dataclasses import dataclass, field
from typing import (
    Dict, List, Literal, Optional, Sequence, Union
)
import numpy as np
from supplyseer.eoq import eoq

@dataclass
class BayesianDistribution:
    """
    Data class to hold physical parameters.

    Attributes
    ----------
    empirical : float
        The empirical value of parameter
    prior : float
        The value of parameter from initial belief
    min : float
        The minimum value of parameter from initial belief
    max : float
        The maximum value of parameter from initial belief
    num_points : float
        The grid resolution of parameter
    coefficient_of_variance : float, optional
        The std divided by mean (approximated by empirical mean). Defaults to 0.1

    Methods
    -------
    normal_pdf():
        Returns normal distribution at a point x from defining mean and std
    calculate_parameter_ranges():
        Returns parameter ranges from min, max, and num_points
    calculate_standard_deviation():
        Returns parameter standard deviation from empirical_mean and coefficient_of_variance
    calculate_posterior():
        Returns parameter posterior distribution from prior and likelihood information
    """
    empirical: float
    prior: float
    min: float
    max: float
    num_points: int
    coefficient_of_variance: float = field(default=0.1)

    @staticmethod
    def normal_pdf(x: np.ndarray, mean: float, std: float) -> np.ndarray:
        """
        Calculate normal probability density function.

        Parameters
        ----------
        x : np.ndarray
            The location for evaluation of probability density function
        mean : float
            The mean value of the probability density function
        std : float
            The standard deviation of probability density function

        Returns
        -------
        np.ndarray
            Probability density function value at location x
        """
        return (1/np.sqrt(2 * np.pi * std**2)) * np.exp(-.5 * ((x - mean)**2) / std**2)

    def calculate_parameter_ranges(self):
        """
        Calculate parameter range points for a parameter.

        Returns
        -------
        np.ndarray
            Linearly-spaced from min to max with length num_points
        """
        return np.linspace(self.min, self.max, self.num_points)
    
    def calculate_standard_deviation(self):
        """
        Calculate standard deviation approximated by empirical mean.

        Returns
        -------
        float
            The standard deviation of the parameter
        """
        return self.empirical * self.coefficient_of_variance

    def calculate_posterior(self):
        """
        Calculate posterior distribution for a parameter.

        This uses range from `calculate_parameter_ranges`
        and uses std from `calculate_standard_deviation`
        to calculate prior and likelihood before multiplying
        results in posterior distribution.

        Notes
        -----
        This method uses
        self.calculate_parameter_ranges()
        self.calculate_standard_deviation()

        Returns
        -------
        np.ndarray
            Posterior distribution function value at parameter ranges
        """
        parameter_ranges = self.calculate_parameter_ranges()
        standard_deviation = self.calculate_standard_deviation()
        prior = self.normal_pdf(parameter_ranges, self.prior, standard_deviation)
        likelihood = self.normal_pdf(parameter_ranges, self.empirical, standard_deviation)
        unnormalized_posterior = prior * likelihood
        marginal_likelihood = np.trapezoid(unnormalized_posterior, parameter_ranges)
        return unnormalized_posterior / marginal_likelihood

@dataclass
class SimulationParameters:
    """
    Data class to hold simulation parameters.
    
    Parameters
    ----------
    parameter_grid : Literal['full', 'montecarlo']
        Parameters used to calculate distribution: 'full' or 'montecarlo'
    credible_inteval_alpha : float
        Credible interval alpha. Defaults to 0.05
    num_monte_carlo_simulation : int, optional
        If uses parameter_grid = 'montecarlo', number of simulation. Defaults to 1
    seed_monte_carlo_simulations : Union[None, int, Sequence[int], np.random.SeedSequence], optional
        If uses parameter_grid = 'montecarlo', seed value. Defaults to None
    """
    parameter_grid: Literal['full', 'montecarlo'] = field(default='full')
    credible_interval_alpha: float = field(default=0.05)
    num_monte_carlo_simulations: int = field(default=1)
    seed_monte_carlo_simulations: Union[None, int, Sequence[int], np.random.SeedSequence] = None

@dataclass
class BayesianEOQ:
    """
    Data class to hold EOQ parameters and their ranges.
    
    Attributes
    ----------
    demand : BayesianDistribution
        Bayesian distribution information of demand parameter
    order_cost : BayesianDistribution
        Bayesian distribution information of order_cost parameter
    holding_cost : BayesianDistribution
        Bayesian distribution information of holding_cost parameter
    simulation_parameters : SimulationParameters
        Simulation parameters in calculation of economic order quantity

    Methods
    -------
    _calculate_credible_quantile():
        Returns credible quantile for credible interval calculation
    calculate_credible_interval():
        Returns credible interval for the posterior parameters combination
    calculate_distribution():
        Returns distribution of economic order quantity for parameters in range
    get_most_probable_values():
        Returns most probable economic order quantity values
    get_least_probable_values():
        Returns extreme but least probable values for economic order quantity
    compute_full_analysis():
        Returns all calculation result
    """
    demand: BayesianDistribution
    order_cost: BayesianDistribution
    holding_cost: BayesianDistribution
    simulation_parameters: SimulationParameters

    def _calculate_credible_quantile(self) -> List[float]:
        """
        Calculates credible quantile for credible interval

        Returns
        -------
        List[float]
            List consists of lower and upper bound of quantiles
        """
        if (self.simulation_parameters.credible_interval_alpha < 0
            or self.simulation_parameters.credible_interval_alpha >= 1):
            raise ValueError("Alpha must be between 0 (inclusive) and 1 (exclusive)")
        return [self.simulation_parameters.credible_interval_alpha / 2,
                1 - self.simulation_parameters.credible_interval_alpha / 2]

    def calculate_credible_interval(self) -> List[float]:
        """
        Calculate credible interval for EOQ.
        
        This uses credible interval from `_calculate_credible_quantile`

        Notes
        -----
        This method uses self._calculate_credible_quantile()

        Returns
        -------
        List[float]
            Credible economic order quantity in parameters quantiles
        """
        quantiles = self._calculate_credible_quantile()
        def get_interval_values(posterior: np.ndarray, param_range: np.ndarray) -> List[float]:
            indices = [np.argmin(np.abs(posterior - np.quantile(posterior, q))) for q in quantiles]
            return [param_range[i] for i in indices]

        interval_d = get_interval_values(
            self.demand.calculate_posterior(),
            self.demand.calculate_parameter_ranges()
        )
        interval_a = get_interval_values(
            self.order_cost.calculate_posterior(),
            self.order_cost.calculate_parameter_ranges()
        )
        interval_h = get_interval_values(
            self.holding_cost.calculate_posterior(),
            self.holding_cost.calculate_parameter_ranges()
        )

        return [eoq(d, a, h)
                for d in interval_d
                for a in interval_a
                for h in interval_h]

    def calculate_distribution(self) -> List[float]:
        """
        Calculate EOQ distribution based on parameter ranges.
        
        Notes
        -----
        Uses 'full' parameter grid by default,
        can be changed to 'montecarlo' grid
        and optionally may define other parameters
        - `num_monte_carlo_simulations`
        - `seed_monte_carlo_simulations`

        Returns
        -------
        List[float]
            List of economic order quantity in pre-defined grids
        """
        if self.simulation_parameters.parameter_grid == 'full':
            d_range = self.demand.calculate_parameter_ranges()
            a_range = self.order_cost.calculate_parameter_ranges()
            h_range = self.holding_cost.calculate_parameter_ranges()
            return [eoq(d, a, h)
                    for d in d_range
                    for a in a_range
                    for h in h_range]
        if self.simulation_parameters.parameter_grid == 'montecarlo':
            eoq_montecarlo = eoq(d_range, a_range, h_range)
            rng = np.random.default_rng(
                self.simulation_parameters.seed_monte_carlo_simulations)
            return rng.choice(eoq_montecarlo,
                              size=self.simulation_parameters.num_monte_carlo_simulations,
                              replace=True).tolist()
        return []

    def get_most_probable_values(self) -> Dict[str, float]:
        """
        Get most probable values for EOQ parameters.
        
        Notes
        -----
        Uses ranges from `BayesianDistribution.calculate_parameter_ranges`
        and posterior from `BayesianDistribution.calculate_posterior`

        Returns
        -------
        Dict[str, float]
            Dictionary of most probable eoq with respecting d, a, and h
        """
        d_range = self.demand.calculate_parameter_ranges()
        a_range = self.order_cost.calculate_parameter_ranges()
        h_range = self.holding_cost.calculate_parameter_ranges()
        posterior_d = self.demand.calculate_posterior()
        posterior_a = self.demand.calculate_posterior()
        posterior_h = self.demand.calculate_posterior()
        return {
            'eoq': eoq(
                d_range[np.argmax(posterior_d)],
                a_range[np.argmax(posterior_a)],
                h_range[np.argmax(posterior_h)]
            ),
            'd': d_range[np.argmax(posterior_d)],
            'a': a_range[np.argmax(posterior_a)],
            'h': h_range[np.argmax(posterior_h)]
        }

    def get_least_probable_values(self) -> Dict[str, Dict[str, float]]:
        """
        Get least probable with most extreme values for EOQ parameters.

        Returns
        -------
        Dict[str, Dict[str, float]]
            Dictionary of least probable eoq and extreme values with respecting d, a, and h
        """
        return {
            'min': {
                'eoq': eoq(self.demand.min, self.order_cost.min, self.holding_cost.max),
                'd': self.demand.min,
                'a': self.order_cost.min,
                'h': self.holding_cost.max  # max since holding_cost in denominator
            },
            'max': {
                'eoq': eoq(self.demand.max, self.order_cost.max, self.holding_cost.min),
                'd': self.demand.max,
                'a': self.order_cost.max,
                'h': self.holding_cost.min  # min since holding_cost in denominator
            }
        }

    def compute_full_analysis(self) -> Dict:
        """
        Compute complete Bayesian EOQ analysis.
        
        Notes
        -----
        Builds on top of another methods
        - self.get_most_probable_values()
        - self.get_least_probable_values()
        - self.calculate_distribution()
        - self.calculate_credible_interval()

        Returns
        -------
        Dict
            Dictionary of all calculation available
        """
        return {
            'bayesian_eoq_most_probable': self.get_most_probable_values(),
            'bayesian_eoq_min_least_probable': self.get_least_probable_values()['min'],
            'bayesian_eoq_max_least_probable': self.get_least_probable_values()['max'],
            'eoq_distribution': self.calculate_distribution(),
            'eoq_credible_interval': self.calculate_credible_interval()
        }

# Example usage:
def bayesian_eoq_full(d: float, a: float, h: float,
                      min_d: float, max_d: float,
                      min_a: float, max_a: float,
                      min_h: float, max_h: float,
                      initial_d: float, initial_a: float, initial_h: float,
                      n_param_values: int,
                      parameter_space: Optional[str] = 'full',
                      n_simulations: Optional[int] = 1) -> Dict:
    """Wrapper function for backward compatibility."""
    data_demand = BayesianDistribution(
        empirical=d, prior=initial_d, min=min_d, max=max_d, num_points=n_param_values
    )
    data_order_cost = BayesianDistribution(
        empirical=a, prior=initial_a, min=min_a, max=max_a, num_points=n_param_values
    )
    data_holding_cost = BayesianDistribution(
        empirical=h, prior=initial_h, min=min_h, max=max_h, num_points=n_param_values
    )
    data_simulation_parameters = SimulationParameters(
        parameter_grid=parameter_space,
        num_monte_carlo_simulations=n_simulations
    )
    bayesian_eoq = BayesianEOQ(data_demand, data_order_cost,
                               data_holding_cost, data_simulation_parameters)
    return bayesian_eoq.compute_full_analysis()
