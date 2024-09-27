import numpy as np
import pandas as pd
import polars as pl
from typing import Union, List, Optional


def pca_embeddings(input_data: Union[np.ndarray, pd.Series, pl.Series],
                   n_embeddings: Optional[int] = 3,
                   auto_standardize: Optional[str] = True) -> np.ndarray:
    
    """
    Classical PCA calculation using NumPy library. 

    Args:
        input_data: the data you want to use for PCA
        n_embeddings: the number of components in your PCA model
        auto_standardize: this function handles the standardization of your data
    
    Returns:
        np.ndarray: a numpy array of shape (rows, n_embeddings)
    """




class PCA:
    def __init__(self, input_data):

        self.input_data = input_data


    def _convert_to_numpy(self):

        if type(self.input_data) == pd.DataFrame:
            return self.input_data.to_numpy()
        
        if type(self.input_data) == pl.DataFrame:
            return self.input_data.to_numpy()

    def _standardize_data(self):
        if type(self.input_data) != np.ndarray:
            self.input_data = self._convert_to_numpy()
        
        self.input_data = self.input_data - self.input_data.mean(axis=0)

        return self.input_data
    
    def fit_pca(self, k=3):

        x = self._standardize_data()
        cov = np.cov(x, rowvar=False)
        ev, eig = np.linalg.eig(cov)
        idx = np.argsort(ev)[::-1]
        eig = eig[:, idx]
        ev = ev[idx]

        explained_var = ev / np.sum(ev)

        reduced_data = np.matmul(x, eig[:,:k])

        output = {
            "pca_data": reduced_data,
            "explained_variance": explained_var,
            "n_embeddings": k,

        }

        return output



