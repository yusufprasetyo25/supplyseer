import numpy as np
import pyAgrum as gum
import pandas as pd
import polars as pl
from typing import Union

class ProbabilisticNet(gum.BayesNet):
    """
    This is a wrapper around the BayesNet class found in the Python library of
    pyAgrum and is providing all of the functionalities in it.
    """
    def __init__(self):
        super().__init__()

    def quick_BN(self, graph_structure: str):
        return gum.fastBN(graph_structure)
    
    def inference_hard_evidence(self,
                            model_input: gum.pyAgrum.BayesNet,
                            input_data: Union[pd.DataFrame, pl.DataFrame], 
                            input_features: list,
                            target: str) -> np.ndarray:
        
        """
        Function that does inference on a dataframe given that a BayesNet model has been
        constructed with both parameters and structure. 

        Args: 
            input_data: a pandas dataframe with discrete features
            input_features: a list of the predictive features you have data on
            target: the target feature you want to do inference on
        
        Returns:
            output: a numpy array of the posterior of the target feature

        """
        if isinstance(input_data, pd.DataFrame):
            inference_df = input_data[input_features].copy()
        elif isinstance(input_data, pl.DataFrame):
            inference_df = input_data.to_pandas()[input_features].copy()
        else:
            raise TypeError(f"Must provide a pandas or polars dataframe. You provided {type(input_data)}")


        ie = gum.LazyPropagation(model_input)
        
        input_feats = {}
        output = []
        for i in range(len(inference_df)):
            for key, val in zip(input_features, inference_df[input_features].iloc[i,:].values):
                input_feats[key] = int(val)
            
            ie.setEvidence(input_feats)
            ie.makeInference()
            output.append( ie.posterior(target)[:] )

        return np.asarray(output)
