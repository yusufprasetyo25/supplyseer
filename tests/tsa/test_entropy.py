from ordpy import complexity_entropy
from typing import Union, List, Optional
import numpy as np
import pandas as pd
import polars as pl

from src.supplyseer.tsa.advanced.entropy import shannon_entropy, permutation_complexity



def test_entropy_metrics():

    x = np.linspace(1, 1000, 1000)
    xmax_expected, xmin_expected = 1000, 1

    assert x.max() == xmax_expected
    assert x.min() == xmin_expected


    shannon_entropy_output = shannon_entropy(x)
    permutation_entropy = permutation_complexity(x)[0]
    statistical_complexity = permutation_complexity(x)[1]


    assert np.round(shannon_entropy_output, 3) == np.float64(9.688)
    assert permutation_entropy == np.float64(0.0)
    assert statistical_complexity == np.float64(0.0)
