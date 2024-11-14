from sktime.forecasting.arima import ARIMA as SktimeARIMA, AutoARIMA as SktimeAutoARIMA
from sktime.forecasting.base import ForecastingHorizon
from typing import List, get_origin, get_args
from pydantic import BaseModel, Field
import polars as pl
import pandas as pd
from pprint import pformat
import numpy as np
from typing import Union, List, Optional
import warnings

class ARIMAConfig(BaseModel):
    arima_order: List[int] = Field(default=[1, 0, 1])
    sarima_order: List[int] = Field(default=[0, 0, 0])
    approach: str = Field(default="standard")
    random_state: int = Field(default=0)
    number_models_parallel: int = Field(default=-1)
    optimizer: Optional[str] = Field(default="lbfgs")
    suppress_warnings: Optional[bool] = Field(default=False)
    stationary_target: Optional[bool] = Field(default=False)
    number_iter: int = Field(default=100)
    bayesian: Optional[bool] = Field(default=False)
    auto_arima: Optional[bool] = Field(default=False)
    auto_arima_control: Optional[str] = Field(default="auto")
    seasonal_differencing: Optional[int] = Field(default=1)



class ARIMAModel:
    def __init__(self, config: dict = None):
        self.config = ARIMAConfig(**(config or {}))

        if self.config.auto_arima:
            self.model = SktimeAutoARIMA(
                start_p = 2 if self.config.auto_arima_control == "auto" else self.config.arima_order[0],
                d = None if self.config.auto_arima_control == "auto" else self.config.arima_order[1],
                start_q = 2 if self.config.auto_arima_control == "auto" else self.config.arima_order[2],
                
                start_P = 1 if self.config.auto_arima_control == "auto" else self.config.sarima_order[0],
                D = None if self.config.auto_arima_control == "auto" else self.config.sarima_order[1],
                start_Q = 1  if self.config.auto_arima_control == "auto" else self.config.sarima_order[2],

                sp = self.config.seasonal_differencing,
                stationary = self.config.stationary_target,

                n_jobs = self.config.number_models_parallel,
                maxiter = self.config.number_iter
            )
        
        # Make this explicit!
        else:
            self.model = SktimeARIMA(order=self.config.arima_order,
                                    seasonal_order=tuple(self.config.sarima_order),
                                    method = self.config.optimizer,
                                    suppress_warnings = self.config.suppress_warnings,
                                    maxiter = self.config.number_iter,
                                    mle_regression = True if self.config.approach == "standard" else False,
                                    )
            
        # Set attributes based on the Pydantic model
        for param in self.config.model_fields:
            setattr(self, f"_{param}", getattr(self.config, param))

    def get_params(self):
        return self.config.model_dump()

    @classmethod
    def get_default_config(cls):
        return ARIMAConfig().model_dump()
    

    def estimate_fit(self, data: Union[List, np.ndarray, pd.Series, pl.Series]):
        if isinstance(data, (list, np.ndarray, pl.Series)):
            data = pd.Series(data)
        
        print("Starting model fitting...")
        try:
            self.model.fit(data)
            print("Model fitting completed.")
            print(f"Model attributes after fitting: {dir(self.model)}")
            if self.config.auto_arima:
                print(f"AutoARIMA model_ attribute exists: {hasattr(self.model, 'model_')}")
            else:
                print(f"ARIMA _forecaster attribute exists: {hasattr(self.model, '_forecaster')}")
        except Exception as e:
            print(f"Error during fitting: {str(e)}")
        return self
    
    def get_fitted_params(self, print_format=False):

        if self.config.auto_arima:
            if not hasattr(self.model, 'model_'):
                msg = "AutoARIMA model was fitted but did not converge or store results. Can't pretty print, please use .model.get_fitted_params()"
                return warnings.warn(msg, Warning)

            results = self.model.model_.arima_res_
        else:
            if not hasattr(self.model, '_forecaster') or self.model._forecaster is None:
                return "ARIMA model was fitted but did not converge or store results."
            results = self.model._forecaster.arima_res_

        params = {
            'AIC': results.aic,
            'BIC': results.bic,
            'Order': results.order,
            'Seasonal Order': results.seasonal_order,
        }

        if not print_format:
            params['Coefficients'] = results.to_dict()
            params['P-values'] = results.to_dict()
            return params

        # Pretty print format
        output = ["ARIMA Model Fitted Parameters:", "=" * 30]
        output.extend(f"{k}: {v}" for k, v in params.items())
        output.extend(["", "Coefficients:", "-" * 20])

        coef_table = []
        for name, coef, pval in zip(results.params.index, results.params, results.pvalues):
            significance = ''
            if pval < 0.001:
                significance = '***'
            elif pval < 0.01:
                significance = '**'
            elif pval < 0.05:
                significance = '*'
            coef_table.append(f"{name:<20} {coef:>10.4f} {pval:>10.4f} {significance}")

        output.extend(coef_table)
        output.append("\nSignificance codes:  0 '***' 0.001 '**' 0.01 '*' 0.05")
        output_joined = '\n'.join(output)
        return output_joined
    
    def print_fitted_params(self):
        params = self.get_fitted_params()
        
        # Create a formatted string for the main parameters
        main_params = {k: v for k, v in params.items() if k not in ['Coefficients', 'P-values']}
        output = "ARIMA Model Fitted Parameters:\n" + "=" * 30 + "\n"
        output += pformat(main_params, indent=2, width=40)
        
        # Create a DataFrame for coefficients and p-values
        coef_df = pd.DataFrame({
            'Coefficient': params['Coefficients'],
            'P-value': params['P-values']
        })
        coef_df['Significance'] = coef_df['P-value'].apply(lambda p: 
            '***' if p < 0.001 else 
            '**' if p < 0.01 else 
            '*' if p < 0.05 else '')
        
        output += "\n\nCoefficients and P-values:\n" + "-" * 30 + "\n"
        output += coef_df.to_string(float_format=lambda x: f"{x:.4f}")
        output += "\n\nSignificance codes:  0 '***' 0.001 '**' 0.01 '*' 0.05"
        
        print(output)


    def summary(self):
        if self.config.auto_arima:
            if not hasattr(self.model, 'model_'):
                raise ValueError("AutoARIMA model has not been fitted yet.")
            return self.model.model_.arima_res_.summary()
        else:
            if not hasattr(self.model, '_forecaster') or self.model._forecaster is None:
                raise ValueError("ARIMA model has not been fitted yet.")
            return self.model._forecaster.arima_res_.summary()

    @property
    def aic(self):
        return self.get_fitted_params()['aic']

    @property
    def bic(self):
        return self.get_fitted_params()['bic']

    @property
    def residuals(self):
        if self.config.auto_arima:
            if not hasattr(self.model, 'model_'):
                raise ValueError("AutoARIMA model has not been fitted yet.")
            return self.model.model_.arima_res_.resid
        else:
            if not hasattr(self.model, '_forecaster') or self.model._forecaster is None:
                raise ValueError("ARIMA model has not been fitted yet.")
            return self.model._forecaster.arima_res_.resid

    def debug_info(self):
        if self.config.auto_arima:
            if not hasattr(self.model, 'model_'):
                return "AutoARIMA model has not been fitted yet."
            res = self.model.model_.arima_res_
        else:
            if not hasattr(self.model, '_forecaster') or self.model._forecaster is None:
                return "ARIMA model has not been fitted yet."
            res = self.model._forecaster.arima_res_
        
        info = {
            "Model type": "AutoARIMA" if self.config.auto_arima else "ARIMA",
            "Model attributes": dir(self.model),
            "Results attributes": dir(res),
            "Model order": res.model.order,
            "Seasonal order": res.model.seasonal_order,
            "AIC": res.aic,
            "BIC": res.bic
        }
        return info