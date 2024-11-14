import pandas as pd
import polars as pl
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, column_or_1d
from supplyseer.wrangling.__utils import (
    validate_params, takens_embedding_optimal_parameters, time_delay_embedding
)

def fill_missing_dates(input_data, time_column, period_fill, group_by_columns, keep_order):
    """
    Function impute missing dates in a time series data with a forward fill strategy.
    Will automatically fill the group by columns with the same value as the previous row.
    Other columns will not be handled by this function. 

    Runs on Polars backend and is an extension of polars.DataFrame.upsample() method.
    Visit for more: https://docs.pola.rs/api/python/dev/reference/dataframe/api/polars.DataFrame.upsample.html

    Args:
        input_data: a pandas or polars dataframe
        time_column: the column with the time series data
        period_fill: the frequency to fill the missing dates
        group_by_columns: the columns to group by
        keep_order: whether to maintain the order of the data

    Returns:
        pd.DataFrame or pl.DataFrame: a dataframe with the missing dates filled
    """

    # Check the type first, if input is a pandas dataframe, convert it to polars and a dummy helper
    dummy_pandas, dummy_polars = False, False

    if isinstance(input_data, pd.DataFrame):
        input_data = pl.from_pandas(input_data).clone()
        dummy_pandas = True

    if isinstance(input_data, pl.DataFrame):
        input_data = input_data.clone()
        dummy_polars = True

        if input_data[time_column].dtype == pl.String:
            input_data = input_data.with_columns(time_column = pl.col(time_column).str.to_date())
    

    if input_data[time_column].dtype not in [pl.Date, pl.Datetime]:
        raise TypeError(f"The '{time_column}' column must be of type Date or Datetime")
    
    # Sort the dataframe by the time column
    input_data = input_data.set_sorted(time_column)

    # Upsample with a filling strategy
    input_data = input_data.upsample(
        time_column=time_column, every=period_fill, group_by=group_by_columns, maintain_order=keep_order
    )

    # Fill the missing values by group_by_columns with a forward fill strategy
    # but it must be an iterative process where it depends on the number of columns given as argument
    # such that it adds only 1 column if input is only 1 column and so on
    input_data = input_data.with_columns(
        [pl.col(col).forward_fill() for col in group_by_columns]
    )

    if dummy_pandas:
        return input_data.to_pandas()
    
    return input_data


# TODO: Add more functionality with higher delays and higher dimensions
# Do not change the list comprehension as it is the most efficient way to do this without adding complexity
# Takes about 5 seconds to run on 1 million data points on a M1 Macbook Pro

class UnivariateTakens(BaseEstimator, TransformerMixin):
    """
    SingleTakensEmbedding transforms a univariate time series into a multi-dimensional
    phase space using Takens' Embedding Theorem.

    Parameters
    ----------
    parameters_type : {'fixed', 'search'}, default='search'
        Determines whether to use fixed embedding parameters or search for optimal values.
        - 'fixed': Use the provided `time_delay` and `embedding_dimension`.
        - 'search': Automatically find optimal `time_delay` and `embedding_dimension`.

    time_delay : int, default=1
        Time delay between consecutive points in the embedding.
        Used directly if `parameters_type='fixed'`.
        Acts as the maximum lag to consider if `parameters_type='search'`.

    embedding_dimension : int, default=5
        Number of delayed coordinates (embedding dimension).
        Used directly if `parameters_type='fixed'`.
        Acts as the maximum dimension to consider if `parameters_type='search'`.

    stride : int, default=1
        Stride between consecutive embedded vectors.

    n_jobs : int or None, default=1
        Number of parallel jobs to run for parameter searching.
        `-1` means using all processors.

    Attributes
    ----------
    time_delay_ : int
        Optimal time delay after fitting (only if `parameters_type='search'`).

    embedding_dimension_ : int
        Optimal embedding dimension after fitting (only if `parameters_type='search'`).
    """

    def __init__(
        self,
        parameters_type='fixed',
        time_delay=1,
        embedding_dimension=5,
        stride=1,
        n_jobs=1
    ):
        self.parameters_type = parameters_type
        self.time_delay = time_delay
        self.embedding_dimension = embedding_dimension
        self.stride = stride
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        """
        Fit the transformer by optionally searching for optimal time_delay and embedding_dimension.

        Parameters
        ----------
        X : array-like of shape (n_samples,)
            Univariate time series data.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : object
            Fitted transformer.
        """
        # Validate input
        X = column_or_1d(X, warn=True)
        if X.ndim != 1:
            raise ValueError("Input time series X must be one-dimensional.")

        if self.parameters_type not in ['fixed', 'search']:
            raise ValueError("`parameters_type` must be either 'fixed' or 'search'.")

        if self.parameters_type == 'search':
            # Define validation criteria for hyperparameters
            validation_references = {
                'time_delay': {'type': int, 'other': lambda x: x > 0},
                'dimension': {'type': int, 'other': lambda x: x > 1},
                'stride': {'type': int, 'other': lambda x: x > 0},
                'n_jobs': {'type': (int, type(None))}
            }

            # Validate hyperparameters
            hyperparameters = {
                'time_delay': self.time_delay,
                'dimension': self.embedding_dimension,
                'stride': self.stride,
                'n_jobs': self.n_jobs
            }
            validate_params(hyperparameters, validation_references)

            # Search for optimal parameters
            optimal_time_delay, optimal_embedding_dimension = takens_embedding_optimal_parameters(
                X=X,
                max_time_delay=self.time_delay,
                max_dimension=self.embedding_dimension,
                stride=self.stride,
                n_jobs=self.n_jobs,
                validate=True
            )
            self.time_delay_ = optimal_time_delay
            self.embedding_dimension_ = optimal_embedding_dimension
        else:
            # Use fixed parameters
            # Validate fixed parameters
            fixed_validation_references = {
                'time_delay': {'type': int, 'other': lambda x: x > 0},
                'embedding_dimension': {'type': int, 'other': lambda x: x > 1},
                'stride': {'type': int, 'other': lambda x: x > 0}
            }

            fixed_hyperparameters = {
                'time_delay': self.time_delay,
                'embedding_dimension': self.embedding_dimension,
                'stride': self.stride
            }
            validate_params(fixed_hyperparameters, fixed_validation_references)

            self.time_delay_ = self.time_delay
            self.embedding_dimension_ = self.embedding_dimension

        return self

    def transform(self, X, y=None):
        """
        Transform the input time series into its embedded phase space.

        Parameters
        ----------
        X : array-like of shape (n_samples,)
            Univariate time series data.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        embedded_X : np.ndarray of shape (n_vectors, embedding_dimension_)
            Embedded phase space representation of the input time series.
        """
        # Check if fit has been called
        check_is_fitted(self, ['time_delay_', 'embedding_dimension_'])

        # Validate input
        X = column_or_1d(X, warn=True)
        if X.ndim != 1:
            raise ValueError("Input time series X must be one-dimensional.")

        # Perform time-delay embedding
        embedded_X = time_delay_embedding(
            time_series=X,
            time_delay=self.time_delay_,
            embedding_dimension=self.embedding_dimension_,
            stride=self.stride,
            flatten=False,
            ensure_last_value=True
        )

        return embedded_X

    def fit_transform(self, X, y=None):
        """
        Fit the transformer and transform the input time series.

        Parameters
        ----------
        X : array-like of shape (n_samples,)
            Univariate time series data.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        embedded_X : np.ndarray of shape (n_vectors, embedding_dimension_)
            Embedded phase space representation of the input time series.
        """
        return self.fit(X, y).transform(X, y)

    def resample(self, y, X=None):
        """
        Resample the target variable to align with the embedded vectors.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target variable corresponding to the time series.

        X : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        resampled_y : np.ndarray of shape (n_vectors,)
            Resampled target variable aligned with the embedded vectors.
        """
        # Check if fit has been called
        check_is_fitted(self, ['time_delay_', 'embedding_dimension_'])

        # Validate input
        y = column_or_1d(y, warn=True)
        if y.ndim != 1:
            raise ValueError("Target variable y must be one-dimensional.")

        required_length = self.time_delay_ * (self.embedding_dimension_ - 1) + 1
        if len(y) < required_length:
            raise ValueError(
                f"Target variable y is too short for embedding: requires at least {required_length} samples, got {len(y)}."
            )

        # Calculate the number of embedded vectors
        max_start_index = len(y) - required_length
        n_vectors = (max_start_index // self.stride) + 1

        # Generate start indices
        start_indices = np.arange(n_vectors) * self.stride

        # Ensure the last embedding includes the last value
        if self.stride > 1 and n_vectors > 0:
            start_indices[-1] = len(y) - required_length

        # Resample y by selecting the value corresponding to the last point in each embedded vector
        resampled_y = y[start_indices + self.time_delay_ * (self.embedding_dimension_ - 1)]

        return resampled_y