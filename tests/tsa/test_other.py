import numpy as np

from src.supplyseer.tsa.statistical.other import manipulative_index, hurst_exp_rs

def test_other_statistical_metrics():

    x = np.linspace(1, 1000, 1000)
    xmax_expected, xmin_expected = 1000, 1
    
    assert x.max() == xmax_expected
    assert x.min() == xmin_expected

    manipulative_index_output = manipulative_index(x, window_size=10)
    manipulative_index_output = np.round(manipulative_index_output.sum())

    hurst_exp_rs_output = hurst_exp_rs(x, min_window=10, max_window=None)
    hurst_exp_rs_output = np.round(hurst_exp_rs_output, 3)

    assert manipulative_index_output == np.float64(991.0)
    assert hurst_exp_rs_output == np.float64(1.0)