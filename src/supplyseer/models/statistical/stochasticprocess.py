import numpy as np
from typing import Optional, Dict, List, Tuple
from pydantic import BaseModel, Field, model_validator
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns
from typing_extensions import Annotated

class DemandSimConfig(BaseModel):
    """Configuration for demand simulation parameters"""
    initial_demand: Annotated[float, Field(gt=0)] = Field(
        ..., 
        description="Initial demand level"
    )
    drift: Annotated[float, Field(ge=0)] = Field(
        0.05, 
        description="Drift parameter (mu)"
    )
    volatility: Annotated[float, Field(gt=0)] = Field(
        0.1, 
        description="Volatility parameter (sigma)"
    )
    decay_rate: Annotated[float, Field(ge=0)] = Field(
        0.05, 
        description="Natural demand decay rate"
    )
    time_horizon: Annotated[float, Field(gt=0)] = Field(
        2.0, 
        description="Time horizon for simulation"
    )
    n_paths: Annotated[int, Field(gt=0)] = Field(
        1000, 
        description="Number of simulation paths"
    )
    n_steps: Annotated[int, Field(gt=0)] = Field(
        50, 
        description="Number of time steps"
    )
    random_seed: Optional[int] = Field(
        None, 
        description="Random seed for reproducibility"
    )

    @model_validator(mode='after')
    def validate_time_steps(self) -> 'DemandSimConfig':
        if self.n_steps >= self.time_horizon * 365:  # Assuming time_horizon is in years
            raise ValueError("Number of steps should be less than days in simulation horizon")
        return self

class DemandPathResult(BaseModel):
    """Results from a demand simulation"""
    paths: np.ndarray = Field(
        ..., 
        description="Array of all simulated paths"
    )
    mean_path: np.ndarray = Field(
        ..., 
        description="Mean path across all simulations"
    )
    final_distribution: Dict[str, float] = Field(
        ..., 
        description="Statistics of final values"
    )
    time_grid: np.ndarray = Field(
        ..., 
        description="Time points of simulation"
    )
    
    model_config = {
        "arbitrary_types_allowed": True  # Allow numpy arrays
    }

class StochasticDemandProcess:
    """
    Simulate demand evolution using Geometric Brownian Motion with decay.
    
    This class implements a stochastic process model for demand that accounts for:
    - Natural market decay/obsolescence
    - Market volatility
    - Trend/drift
    - Multiple simulation paths
    """
    
    def __init__(self, config: DemandSimConfig):
        """
        Initialize the stochastic demand process.

        Args:
            config: Configuration parameters for the simulation
        """
        self.config = config
        if config.random_seed is not None:
            np.random.seed(config.random_seed)
        
        self.time_grid = np.linspace(0, config.time_horizon, config.n_steps)
        self.dt = config.time_horizon / config.n_steps
        self._results = None

    def simulate(self) -> DemandPathResult:
        """
        Simulate demand paths using Geometric Brownian Motion with decay.
        
        Returns:
            DemandPathResult containing simulation results
        """
        # Initialize paths array
        paths = np.zeros((self.config.n_paths, self.config.n_steps))
        paths[:, 0] = self.config.initial_demand
        
        # Generate paths
        for i in range(1, self.config.n_steps):
            dW = np.random.normal(0, np.sqrt(self.dt), self.config.n_paths)
            paths[:, i] = paths[:, i-1] * np.exp(
                (-self.config.drift - self.config.decay_rate - 0.5 * self.config.volatility**2) * self.dt + 
                self.config.volatility * dW
            )
        
        # Calculate statistics
        mean_path = np.mean(paths, axis=0)
        final_values = paths[:, -1]
        
        self._results = DemandPathResult(
            paths=paths,
            mean_path=mean_path,
            final_distribution={
                'mean': float(np.mean(final_values)),
                'std': float(np.std(final_values)),
                'median': float(np.median(final_values)),
                'q25': float(np.percentile(final_values, 25)),
                'q75': float(np.percentile(final_values, 75))
            },
            time_grid=self.time_grid
        )
        
        return self._results

    def plot_results(self, style: str = "dark_background") -> Tuple[plt.Figure, plt.Axes]:
            """
            Create visualization of simulation results.
            
            Args:
                style: matplotlib style to use for plotting
                
            Returns:
                tuple containing the figure and axes objects
            """
            if self._results is None:
                raise ValueError("Must run simulate() before plotting")
            
            # Clear any existing plots and set style
            plt.clf()
            plt.close('all')
                
            # Set style globally
            plt.style.use(style)
            
            # Configure seaborn
            if style == "dark_background":
                sns.set_theme(style="darkgrid", rc={"text.color": "white",
                                                "axes.labelcolor": "white",
                                                "xtick.color": "white",
                                                "ytick.color": "white"})
            else:
                sns.set_theme(style="whitegrid")
            
            # Create figure with gridspec
            fig = plt.figure(figsize=(18, 8))
            gs = fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.05)

            # Set figure background color and text colors based on style
            text_color = 'white' if style == "dark_background" else 'black'
            fig.patch.set_facecolor('black' if style == "dark_background" else 'white')
            
            # Left plot - paths
            ax1 = fig.add_subplot(gs[0])
            ax1.set_facecolor('black' if style == "dark_background" else 'white')
            
            # Set text colors for ax1
            ax1.xaxis.label.set_color(text_color)
            ax1.yaxis.label.set_color(text_color)
            ax1.title.set_color(text_color)
            ax1.tick_params(colors=text_color)
            for spine in ax1.spines.values():
                spine.set_color(text_color)
            
            # Sort paths and plot
            sort_idx = np.argsort(self._results.paths[:, -1])
            sorted_paths = self._results.paths[sort_idx]
            positions = np.linspace(0, 1, len(sorted_paths))
            
            cmap = plt.cm.viridis
            for path, pos in zip(sorted_paths, positions):
                ax1.plot(self.time_grid, path, '-', color=cmap(pos), linewidth=1, alpha=0.6)
            
            ax1.plot(self.time_grid, self._results.mean_path, '--', color='red', 
                    label='E[Demand_t]', linewidth=2)
            
            # Right plot - final distribution
            ax2 = fig.add_subplot(gs[1])
            ax2.set_facecolor('black' if style == "dark_background" else 'white')
            
            # Set text colors for ax2
            ax2.xaxis.label.set_color(text_color)
            ax2.yaxis.label.set_color(text_color)
            ax2.title.set_color(text_color)
            ax2.tick_params(colors=text_color)
            for spine in ax2.spines.values():
                spine.set_color(text_color)
            
            # Create histogram
            hist, bins = np.histogram(self._results.paths[:, -1], bins=20)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            hist_colors = cmap(np.linspace(0, 1, len(bins)-1))
            
            for count, color, bottom, top in zip(hist, hist_colors, bins[:-1], bins[1:]):
                ax2.barh(y=(bottom + top)/2, width=count, height=top-bottom,
                        color=color, alpha=0.8, align='center')
            
            ax2.axhline(self._results.mean_path[-1], linestyle="--", color='red', 
                    label="E[Demand_t]")
            
            x_range = np.linspace(ax2.get_ylim()[0], ax2.get_ylim()[1], 100)
            y_pdf = norm.pdf(x_range, self._results.final_distribution['mean'], 
                            self._results.final_distribution['std'])
            y_pdf = y_pdf / np.max(y_pdf) * ax2.get_xlim()[1] * 0.9
            ax2.plot(y_pdf, x_range, text_color, label='Demand_T pdf', linewidth=2)
            
            # Formatting
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Demand Level')
            ax1.set_title(f'Demand Evolution Simulation\nInitial demand: {self.config.initial_demand:,.0f}')
            ax1.grid(True, alpha=0.3)
            
            # Set legend colors
            legend1 = ax1.legend()
            for text in legend1.get_texts():
                text.set_color(text_color)
                
            ax2.set_title('Final Distribution')
            ax2.set_xlabel('')
            ax2.set_xticklabels([])
            
            # Set legend colors for ax2
            legend2 = ax2.legend()
            for text in legend2.get_texts():
                text.set_color(text_color)
            
            # Match y-axis limits
            y_min = min(ax1.get_ylim()[0], ax2.get_ylim()[0])
            y_max = max(ax1.get_ylim()[1], ax2.get_ylim()[1])
            ax1.set_ylim(y_min, y_max)
            ax2.set_ylim(y_min, y_max)
            
            # Adjust layout
            plt.tight_layout()
            
            return fig, (ax1, ax2)

    def get_quantile_paths(self, quantiles: List[float] = [0.05, 0.25, 0.5, 0.75, 0.95]) -> Dict[float, np.ndarray]:
        """
        Calculate quantile paths across all simulations.
        
        Args:
            quantiles: List of quantiles to calculate
            
        Returns:
            Dictionary mapping quantile levels to paths
        """
        if self._results is None:
            raise ValueError("Must run simulate() before calculating quantiles")
            
        return {
            q: np.quantile(self._results.paths, q, axis=0)
            for q in quantiles
        }

    def get_demand_threshold_probability(self, threshold: float) -> np.ndarray:
        """
        Calculate probability of demand falling below threshold at each time point.
        
        Args:
            threshold: Demand level threshold
            
        Returns:
            Array of probabilities for each time point
        """
        if self._results is None:
            raise ValueError("Must run simulate() before calculating threshold probability")
            
        return np.mean(self._results.paths < threshold, axis=0)