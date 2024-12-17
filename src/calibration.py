# calibration.py

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from typing import Tuple

# def calibrate_w_model(df: pd.DataFrame) -> Tuple[float, float, float, float]:
#     """
#     Calibrate W model parameters using OLS regression.
    
#     Parameters:
#     - df: DataFrame containing 'W_lag' and 'r_u_lag' columns
    
#     Returns:
#     - alpha: Intercept
#     - rho: Coefficient for W_lag
#     - b1: Coefficient for r_u_lag
#     - sigma: Standard deviation of residuals
#     """
#     X = df[['W_lag', 'r_u_lag']].values
#     y = df['W'].values
    
#     model = LinearRegression()
#     model.fit(X, y)
    
#     alpha = model.intercept_
#     rho, neg_b1 = model.coef_
#     b1 = -neg_b1  # As per your original code
    
#     residuals = y - model.predict(X)
#     sigma = residuals.std()
    
#     return alpha, rho, b1, sigma



def calibrate_w_model(df: pd.DataFrame) -> Tuple[float, float, float, float]:
        """
        Calibrate W model parameters using OLS regression and display results.

        :param df: DataFrame containing 'W_lag' and 'r_u_lag' columns.
        """
        X = df[['W_lag', 'r_u_lag']]
        y = df['W']
        X = sm.add_constant(X)  # Add constant for OLS

        model = sm.OLS(y, X).fit()
        model.summary()

        alpha = model.params['const']
        rho = model.params['W_lag']
        b1 = -model.params['r_u_lag']  # As per your original code
        sigma = np.sqrt(model.mse_resid)

        return ((alpha, rho, b1, sigma), model.summary())
        