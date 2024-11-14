import numpy as np
import warnings



def manipulative_index(input_data, window_size=10):
    """
    Calculate the Manipulability Index used in Robotics & Control.

    Args:
        input_data: an n-dimensional array (univariate or multivariate)
        window_size: the size of the window sliding over the Jacobian for calculation
    
    Returns:
        np.array: Manipulability Index for each time step except the first window_size-1 step
    """
    n_steps = input_data.shape[0]
    manipulability = np.zeros(n_steps - window_size + 1)
    X = np.arange(window_size)
    X_matrix = np.vstack([X, np.ones(window_size)]).T
    
    for t in range(window_size - 1, n_steps):
        window = input_data[t-window_size+1:t+1]
        beta = np.linalg.lstsq(X_matrix, window, rcond=None)[0]
        J = beta[0]
        
        # Ensure J is 2D
        if J.ndim == 1:
            J = J.reshape(1, -1)
        
        JJT = np.dot(J, J.T)
        
        # Handle potential singularity
        if JJT.size == 1:  # Scalar case
            manipulability[t - window_size + 1] = abs(JJT.item())
        else:
            try:
                manipulability[t - window_size + 1] = np.sqrt(np.linalg.det(JJT))
            except np.linalg.LinAlgError:
                # In case of singularity, use the product of diagonal elements
                print("Detected error, will do product of diagonals")
                manipulability[t - window_size + 1] = np.sqrt(np.prod(np.diag(JJT)))
    
    return manipulability

def shannon_diversity():
    pass

def pagels_lambda():
    pass

def kl_divergence():
    pass

def kolmogorov_sinai():
    pass

def lempel_ziv():
    pass

def hurst_exp_rs(input_data, min_window=10, max_window=None):
    """
    Calculate the Hurst exponent using the R/S method.
    
    Parameters:
    input_data (array-like): Input time series
    min_window (int, optional): Minimum window size to consider. Default is 10.
    max_window (int, optional): Maximum window size to consider. Default is len(input_data) / 2.
    
    Returns:
    float: Estimated Hurst exponent
    """
    input_data = np.asarray(input_data)
    
    if len(input_data) < 100:
        warnings.warn("Time series is short. Results may be unreliable.", UserWarning)
    
    if max_window is None:
        max_window = len(input_data) // 2
    
    max_window = min(max_window, len(input_data) // 2)
    
    if max_window < min_window:
        raise ValueError("max_window must be larger than min_window")
    
    rs_values = []
    window_sizes = range(min_window, max_window)

    for window_size in window_sizes:
        num_segments = len(input_data) // window_size
        rescaled_ranges = []

        for segment in range(num_segments):
            segment_data = input_data[segment * window_size: (segment + 1) * window_size]
            
            mean_adjusted_series = segment_data - np.mean(segment_data)
            
            cumulative_deviation = np.cumsum(mean_adjusted_series)
            
            R = np.max(cumulative_deviation) - np.min(cumulative_deviation)
            
            S = np.std(segment_data)
            
            if S > 0:
                rescaled_ranges.append(R / S)

        if rescaled_ranges:
            rs_values.append(np.mean(rescaled_ranges))

    log_rs_values = np.log(rs_values)
    log_window_sizes = np.log(window_sizes)
    
    slope, _ = np.polyfit(log_window_sizes, log_rs_values, 1)

    hurst = slope
    
    return hurst