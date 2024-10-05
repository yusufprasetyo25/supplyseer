import numpy as np

from src.supplyseer.models.statistical.arima import ARIMAModel

def test_arima_model():

    configs = {"arima_order": [1,0,1],
                 "sarima_order": [0,0,0, 0],
                 "approach": "standard",
                 "random_state": 0,
                 "n_jobs": -1,
                 "optimizer":  "lbfgs",
                 "suppress_warnings":  True,
                 "stationary_target": False,
                 "number_iter":  100,
                 "bayesian": False,
                 "auto_arima": True,
                 "auto_arima_control": "auto",
                 "stationary_target": True}
    
    

    first_range = np.arange(0, 100)
    first_term = np.sin(first_range*.3)+20

    model = ARIMAModel(configs)
    model.estimate_fit(first_term)

    assert model.model is not None, "No model was created"
    assert model.model.summary() is not None, "No model summary was created"