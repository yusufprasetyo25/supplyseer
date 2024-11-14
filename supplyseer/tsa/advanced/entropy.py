from ordpy import complexity_entropy
from typing import Union, List, Optional
import numpy as np
import pandas as pd
import polars as pl


def shannon_entropy(input_data: Union[np.ndarray, List, pl.Series, pd.Series]) -> np.ndarray:
    """
    Calculate the naive Shannon Entropy with base 2.

    Args: 
        input_data: a univariate input of data
    
    Returns:
        np.ndarray: a scalar value of the shannon entropy in numpy array
    """

    sorted_data = np.sort(input_data)
    px = sorted_data / np.sum(sorted_data)
    plogx = np.log2(px)

    return -np.dot(px, plogx)


def permutation_complexity(input_data: Union[np.ndarray, List, pl.Series, pd.Series],
                           n_pairs: Optional[int] = 2) -> list:
    
    """
    Calculates the Permutation Entropy and Statistical Complexity of an array

    Args:
        input_data: a univariate or bivariate input of data
    
    Returns:
        tuple: a tuple of permutation entropy and statistical complexity
    """

    output = complexity_entropy(input_data, dx=n_pairs)

    return [output[0], output[1]]


