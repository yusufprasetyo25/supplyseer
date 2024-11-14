import numpy as np
import pandas as pd
import polars as pl
from typing import Union, List, Optional

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



