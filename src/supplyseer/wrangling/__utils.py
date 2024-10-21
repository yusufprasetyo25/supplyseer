import numpy as np
from sklearn.utils import column_or_1d
from sklearn.metrics import mutual_info_score
from sklearn.neighbors import NearestNeighbors
from joblib import Parallel, delayed

def time_delay_embedding(
    time_series,
    time_delay=1,
    embedding_dimension=2,
    stride=1,
    flatten=False,
    ensure_last_value=True
):
    """
    Perform time-delay embedding on a univariate time series.

    Parameters
    ----------
    time_series : np.ndarray or list of np.ndarray
        Input time series data. Can be a single univariate time series or a list of such series.

    time_delay : int, default=1
        Number of time steps between successive coordinates in the embedding.

    embedding_dimension : int, default=2
        Number of delayed coordinates (embedding dimension).

    stride : int, default=1
        Step size between consecutive embedded vectors.

    flatten : bool, default=False
        If True and the input has more than one feature, flatten the embedded vectors.

    ensure_last_value : bool, default=True
        If True, ensure that the last value of the time series is included in the embedding.

    Returns
    -------
    embedded_series : np.ndarray or list of np.ndarray
        Embedded time series as a multi-dimensional array or list of arrays.
        - If input is a single series: shape (n_points, embedding_dimension)
        - If input is a list: list of arrays with shape (n_points_i, embedding_dimension)

    Raises
    ------
    ValueError
        If the time series is too short to create at least one embedded vector.
    TypeError
        If the input is neither a numpy.ndarray nor a list of numpy.ndarray.
    """
    def embed_single_series(series):
        n_time_steps = len(series)
        required_time_steps = time_delay * (embedding_dimension - 1) + 1

        if n_time_steps < required_time_steps:
            raise ValueError(
                f"Time series is too short for embedding: required at least {required_time_steps} time steps, "
                f"got {n_time_steps}."
            )

        # Calculate the number of embedding vectors
        max_start_index = n_time_steps - required_time_steps
        n_vectors = (max_start_index // stride) + 1

        # Compute the starting indices for each embedding vector
        start_indices = np.arange(n_vectors) * stride

        if ensure_last_value:
            # Adjust the last start index to include the last time step
            start_indices[-1] = n_time_steps - required_time_steps

        # Generate the indices for each embedding dimension
        dim_offsets = np.arange(embedding_dimension) * time_delay
        embedding_indices = start_indices[:, np.newaxis] + dim_offsets[np.newaxis, :]

        # Extract the embedded vectors
        embedded = series[embedding_indices]

        # Flatten if required
        if flatten and series.ndim > 1:
            # Transpose to bring non-time dimensions forward
            embedded = embedded.transpose(0, *range(1, series.ndim), series.ndim)
            # Calculate new shape
            embedded = embedded.reshape(len(series), -1, embedding_dimension * np.prod(series.shape[1:-1]))

        return embedded

    if isinstance(time_series, np.ndarray):
        # Single time series
        embedded = embed_single_series(time_series)
    elif isinstance(time_series, list):
        # List of time series
        embedded = [embed_single_series(series) for series in time_series]
    else:
        raise TypeError("Input time_series must be a numpy.ndarray or a list of numpy.ndarray.")

    return embedded


def takens_embedding_optimal_parameters(
    X,
    max_time_delay,
    max_dimension,
    stride=1,
    n_jobs=1,
    validate=True
):
    """
    Compute the optimal parameters for a Takens (time-delay) embedding of a univariate time series.

    Parameters
    ----------
    X : ndarray of shape (n_samples,) or (n_samples, 1)
        Input data representing a single univariate time series.

    max_time_delay : int
        Maximum time delay between two consecutive values for constructing one embedded point.

    max_dimension : int
        Maximum embedding dimension that will be considered in the optimization.

    stride : int, optional, default=1
        Stride duration between two consecutive embedded points.

    n_jobs : int or None, optional, default=1
        The number of jobs to run in parallel for the computation.
        `-1` means using all processors.

    validate : bool, optional, default=True
        Whether the input and hyperparameters should be validated.

    Returns
    -------
    time_delay : int
        The optimal time delay less than or equal to `max_time_delay`, as determined
        by minimizing the time-delayed mutual information.

    dimension : int
        The optimal embedding dimension less than or equal to `max_dimension`, as
        determined by a false nearest neighbors heuristic once `time_delay` is computed.

    Raises
    ------
    ValueError
        If the input time series is not one-dimensional or too short for embedding.

    TypeError
        If the input types are incorrect.
    """
    if validate:
        # Define validation criteria for hyperparameters
        validation_references = {
            'time_delay': {'type': int, 'other': lambda x: x > 0},
            'dimension': {'type': int, 'other': lambda x: x > 1},
            'stride': {'type': int, 'other': lambda x: x > 0},
            'n_jobs': {'type': (int, type(None))}
        }

        # Validate hyperparameters
        hyperparameters = {
            'time_delay': max_time_delay,
            'dimension': max_dimension,
            'stride': stride,
            'n_jobs': n_jobs
        }
        validate_params(hyperparameters, validation_references)

        # Ensure X is one-dimensional
        X = column_or_1d(X, warn=True)
        if X.ndim != 1:
            raise ValueError("Input time series X must be one-dimensional.")

    # Step 1: Compute Mutual Information for time delays from 1 to max_time_delay
    mutual_info_list = Parallel(n_jobs=n_jobs)(
        delayed(_mutual_information)(X, time_delay=delay, n_bins=100)
        for delay in range(1, max_time_delay + 1)
    )

    # Find the time delay that minimizes mutual information
    optimal_time_delay = np.argmin(mutual_info_list) + 1  # +1 because delays start at 1
    print(f"Optimal Time Delay: {optimal_time_delay}")

    # Step 2: Compute False Nearest Neighbors for embedding dimensions from 2 to max_dimension
    embedding_dimensions = range(2, max_dimension + 1)
    fnn_counts = Parallel(n_jobs=n_jobs)(
        delayed(false_nearest_neighbors)(
            time_series=X,
            time_delay=optimal_time_delay,
            embedding_dimension=dim,
            stride=stride,
            epsilon_multiplier=2.0,
            tolerance_ratio=10.0
        )
        for dim in embedding_dimensions
    )

    # Step 3: Calculate Variation of FNN Counts to Determine Optimal Embedding Dimension
    # Using Kennel et al.'s heuristic: find the embedding dimension where variation is minimized
    variation_list = []
    for idx in range(1, len(fnn_counts) - 1):
        current_fnn = fnn_counts[idx]
        prev_fnn = fnn_counts[idx - 1]
        next_fnn = fnn_counts[idx + 1]
        variation = abs(current_fnn - 2 * prev_fnn + next_fnn) / (prev_fnn + 1) / embedding_dimensions[idx]
        variation_list.append(variation)

    # Find the embedding dimension with the minimum variation
    if variation_list:
        optimal_dimension_index = np.argmin(variation_list) + 1  # +1 to align with embedding_dimensions
        optimal_embedding_dimension = embedding_dimensions[optimal_dimension_index]
    else:
        # If max_dimension is too low, default to 2
        optimal_embedding_dimension = 2

    print(f"Optimal Embedding Dimension: {optimal_embedding_dimension}")

    return optimal_time_delay, optimal_embedding_dimension



def compute_mutual_information(time_series, max_lag, n_bins=100):
    """
    Compute mutual information between the time series and its lagged version.

    Parameters
    ----------
    time_series : np.ndarray
        Input univariate time series.

    max_lag : int
        Maximum lag to compute mutual information.

    n_bins : int, default=100
        Number of bins for histogram estimation.

    Returns
    -------
    mutual_info : np.ndarray
        Mutual information values for lags from 1 to max_lag.
    """
    mutual_info = np.array([
        _mutual_information(time_series, time_delay=lag, n_bins=n_bins)
        for lag in range(1, max_lag + 1)
    ])
    return mutual_info

def _mutual_information(X, time_delay, n_bins=100):
    """
    Helper function to compute mutual information for a specific lag.

    Parameters
    ----------
    X : np.ndarray
        One-dimensional time series data.

    time_delay : int
        Time delay.

    n_bins : int
        Number of bins for histogram estimation.

    Returns
    -------
    mutual_info : float
        Estimated mutual information.
    """
    if time_delay < 1 or time_delay >= len(X):
        return 0.0

    X_delayed = X[time_delay:]
    X_truncated = X[:-time_delay]

    # Discretize the time series
    X_truncated_binned = np.digitize(X_truncated, bins=np.histogram_bin_edges(X_truncated, bins=n_bins)) - 1
    X_delayed_binned = np.digitize(X_delayed, bins=np.histogram_bin_edges(X_delayed, bins=n_bins)) - 1

    mutual_info = mutual_info_score(X_truncated_binned, X_delayed_binned)
    return mutual_info


def false_nearest_neighbors(
    time_series,
    time_delay,
    embedding_dimension,
    stride=1,
    epsilon_multiplier=2.0,
    tolerance_ratio=10.0
):
    """
    Calculate the number of false nearest neighbors in a specified embedding dimension.

    Parameters
    ----------
    time_series : np.ndarray
        Input univariate time series data.

    time_delay : int
        Time delay used in the embedding.

    embedding_dimension : int
        Embedding dimension.

    stride : int, default=1
        Step size between consecutive embedded vectors.

    epsilon_multiplier : float, default=2.0
        Multiplier for the standard deviation of the time series to define the epsilon threshold.

    tolerance_ratio : float, default=10.0
        Ratio threshold to determine if a neighbor is considered false.

    Returns
    -------
    n_false_neighbors : int
        Number of false nearest neighbors detected.

    Raises
    ------
    ValueError
        If the time series is too short to create at least one embedded vector.

    TypeError
        If the input `time_series` is not a one-dimensional numpy array.
    """
    # Validate input
    if not isinstance(time_series, np.ndarray):
        raise TypeError("Input `time_series` must be a numpy.ndarray.")
    if time_series.ndim != 1:
        raise ValueError("Input `time_series` must be one-dimensional.")

    # Perform time-delay embedding
    embedded = time_delay_embedding(
        time_series=time_series,
        time_delay=time_delay,
        embedding_dimension=embedding_dimension,
        stride=stride,
        flatten=False,
        ensure_last_value=True
    )

    # Initialize NearestNeighbors model to find the nearest neighbor (excluding self)
    neighbor_model = NearestNeighbors(n_neighbors=2, algorithm='auto')
    neighbor_model.fit(embedded)

    # Find the nearest neighbors
    distances, indices = neighbor_model.kneighbors(embedded)

    # Extract the distance to the nearest neighbor (excluding self)
    nearest_neighbor_distances = distances[:, 1]

    # Extract the indices of the nearest neighbors
    nearest_neighbor_indices = indices[:, 1]

    # Retrieve the last value of each embedded vector
    embedded_last_values = embedded[:, -1]

    # Retrieve the last value of the nearest neighbor's embedded vector
    nearest_neighbors_last_values = embedded[nearest_neighbor_indices, -1]

    # Calculate epsilon threshold based on the standard deviation of the time series
    epsilon = epsilon_multiplier * np.std(time_series)

    # Calculate the absolute difference between the embedded vectors and their nearest neighbors
    absolute_differences = np.abs(embedded_last_values - nearest_neighbors_last_values)

    # Calculate the ratio of the absolute difference to the nearest neighbor distance
    # Handle division by zero by setting the ratio to zero where distance is zero
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.divide(
            absolute_differences,
            nearest_neighbor_distances,
            out=np.zeros_like(absolute_differences),
            where=nearest_neighbor_distances != 0
        )

    # Identify false neighbors based on the ratio and distance thresholds
    is_false_neighbor = (ratio > tolerance_ratio) & (nearest_neighbor_distances < epsilon)

    # Count the number of false neighbors
    n_false_neighbors = np.sum(is_false_neighbor)

    return n_false_neighbors

def validate_params(parameters, references, exclude=None):
    """
    Validate hyperparameters against predefined references.

    Parameters
    ----------
    parameters : dict
        Dictionary where keys are parameter names (strings) and values are parameter values.
        Parameters listed in `exclude` are omitted from validation.

    references : dict
        Dictionary where keys are parameter names (strings) and values are dictionaries
        specifying validation criteria:
            - 'type': A class or tuple of classes that the parameter should be an instance of.
            - 'in': A collection of acceptable values for the parameter.
            - 'of': A nested dictionary for validating parameters that are collections.
            - 'other': A callable for custom validation logic.

    exclude : list of str, optional
        List of parameter names to exclude from validation. Defaults to an empty list.

    Raises
    ------
    ValueError
        If a parameter does not meet the specified validation criteria.

    TypeError
        If a parameter's type is incorrect.
    """
    # Initialize exclude list if None
    if exclude is None:
        exclude = []

    # Filter out excluded parameters
    parameters_to_validate = {
        key: value for key, value in parameters.items()
        if key not in exclude
    }

    # Iterate over parameters to validate
    for param_name, param_value in parameters_to_validate.items():
        if param_name not in references:
            raise ValueError(f"Unexpected parameter '{param_name}' provided.")

        reference = references[param_name]

        # Validate 'type'
        if 'type' in reference:
            expected_types = reference['type']
            if not isinstance(param_value, expected_types):
                raise TypeError(
                    f"Parameter '{param_name}' must be of type {expected_types}, "
                    f"but got type {type(param_value)}."
                )

        # Validate 'in'
        if 'in' in reference:
            acceptable_values = reference['in']
            if param_value not in acceptable_values:
                raise ValueError(
                    f"Parameter '{param_name}' must be one of {acceptable_values}, "
                    f"but got {param_value}."
                )

        # Validate 'of' for collection types
        if 'of' in reference:
            collection_reference = reference['of']
            if isinstance(param_value, dict):
                # Recursive validation for nested dictionaries
                validate_params(param_value, collection_reference)
            elif isinstance(param_value, (list, tuple, np.ndarray)):
                # Validate each item in the collection
                for item in param_value:
                    # Assuming 'of' contains a 'type' or other validation rules
                    if 'type' in collection_reference:
                        expected_item_types = collection_reference['type']
                        if not isinstance(item, expected_item_types):
                            raise TypeError(
                                f"Items in parameter '{param_name}' must be of type {expected_item_types}, "
                                f"but got type {type(item)}."
                            )
                    if 'in' in collection_reference:
                        acceptable_item_values = collection_reference['in']
                        if item not in acceptable_item_values:
                            raise ValueError(
                                f"Items in parameter '{param_name}' must be one of {acceptable_item_values}, "
                                f"but got {item}."
                            )
                    if 'other' in collection_reference:
                        # Custom validation callable
                        collection_reference['other'](item)

        # Validate 'other' with a custom callable
        if 'other' in reference:
            custom_validator = reference['other']
            if not callable(custom_validator):
                raise TypeError(
                    f"The 'other' reference for parameter '{param_name}' must be callable."
                )
            custom_validator(param_value)

