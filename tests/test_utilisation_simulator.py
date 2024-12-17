# import unittest
# import numpy as np
# import pandas as pd
# from scipy.stats import zscore
# import statsmodels.api as sm
# from src.models.utilisation_simulator import UtilizationSimulator
# from src.interest_rate_models.semilog_irm import SemiLogIRM
# import requests

# def retrieve_lending_market(chain: str ='ethereum', controller_address: str = '0xeda215b7666936ded834f76f3fbc6f323295110a'):
#     url = f'https://prices.curve.fi/v1/lending/markets/{chain}/{controller_address}/snapshots?fetch_on_chain=false&agg=day'
#     r = requests.get(url)
#     df =  pd.json_normalize(r.json()['data'])
#     df['timestamp'] = pd.to_datetime(df['timestamp'])  
#     return df 


# class TestUtilizationSimulator(unittest.TestCase):
    
#     @classmethod
#     def setUpClass(cls):
#         """
#         Setup class-level attributes for use in the tests.
#         """
#         last_x_month = 5 
#         # Retrieve empirical data
#         cls.df = retrieve_lending_market()
#         cls.df['utilisation_ratio'] = cls.df['total_debt_usd'] / cls.df['total_assets_usd']
#         cls.empirical_data = cls.df[cls.df['timestamp'] >= cls.df['timestamp'].max() - pd.DateOffset(months=last_x_month)]

#         # OLS Analysis for Parameters
#         cls.empirical_data['delta_utilisation'] = cls.empirical_data['utilisation_ratio'].diff()
#         cls.empirical_data = cls.empirical_data.dropna(subset=['delta_utilisation'])
#         cls.empirical_data = cls.empirical_data[
#             (np.abs(zscore(cls.empirical_data['delta_utilisation'])) < 2)
#         ]  # Exclude outliers

#         # Prepare regression data
#         regression_data = cls.empirical_data.dropna(subset=['delta_utilisation', 'rate'])
#         regression_data['lagged_avg_rate'] = regression_data['rate'].shift(1)
#         regression_data = regression_data.dropna(subset=['lagged_avg_rate'])

#         # Perform OLS regression
#         X = sm.add_constant(regression_data['lagged_avg_rate'])  # Add intercept
#         y = regression_data['delta_utilisation']
#         model = sm.OLS(y, X).fit()

#         # Annualize coefficients
#         cls.b0_annualized = model.params['const'] * 365
#         cls.b1_annualized = model.params['lagged_avg_rate'] * 365

#         # Calculate annualized volatility
#         cls.sigma_annualized = regression_data['delta_utilisation'].std() * np.sqrt(365)

#         # Print OLS Summary
#         print("### OLS Regression Summary ###")
#         print(model.summary())
#         print("\n### Derived Parameters ###")
#         print(f"Annualized Drift Parameter (b0): {cls.b0_annualized:.4f}")
#         print(f"Annualized Borrower Sensitivity (b1): {cls.b1_annualized:.4e}")
#         print(f"Annualized Volatility (σ): {cls.sigma_annualized:.4f}")

#         # Define simulation parameters
#         cls.semilog_irm = SemiLogIRM(rate_min=0.01, rate_max=10.0)
#         cls.params = {
#             "U0": 0.5,           # Initial utilization
#             "T": 1.0,            # Total simulation time (1 year)
#             "dt": 1 / 365,       # Daily time step
#             "rate_model": cls.semilog_irm.calculate_rate,
#             "A": cls.b1_annualized,
#             "C": cls.b0_annualized,
#             "r0": 0.21,# 0.05,
#             "r1": 0.17, # 0.01,
#             "sigma_const": cls.sigma_annualized,
#         }

#     def test_simulation_matches_empirical_data_stabilized(self):
#         """
#         Test if the simulation's metrics for the last 6 months
#         match empirical data within tolerable limits.
#         """
#         last_x_month = 5 

#         num_simulations = 1000
#         last_months_index = int((self.params["T"] / self.params["dt"]) * (1 - last_x_month / 12))
#         simulated_means = []
#         simulated_volatilities = []

#         # Run multiple simulations
#         for _ in range(num_simulations):
            
#             # simulator = UtilizationSimulator(
#             #     U0=self.params["U0"],
#             #     T=self.params["T"],
#             #     dt=self.params["dt"],
#             #     rate_model=self.params["rate_model"],
#             #     A=self.params["A"],
#             #     C=self.params["C"],
#             #     r0=self.params["r0"],
#             #     r1=self.params["r1"],
#             #     sigma_const=self.params["sigma_const"]
#             # )

#             simulator = UtilizationSimulator(
#                 U0=self.params["U0"],  # Initial utilization
#                 T=self.params["T"],   # 1 year
#                 dt=self.params["dt"],  # Daily steps
#                 rate_model=self.params["rate_model"],
#                 b0=self.params['C'],  # Derived from OLS
#                 b1=self.params["A"],  # Derived from OLS
#                 sigma=self.params["sigma_const"]  # Derived from OLS
#             )

#             utilization_path, _ = simulator.simulate()

#             # Evaluate metrics over the last 6 months
#             last_months = utilization_path[last_months_index:]
#             simulated_means.append(np.mean(last_months))
#             simulated_volatilities.append(np.std(last_months))

#         # Calculate aggregated metrics
#         simulated_mean = np.mean(simulated_means)
#         simulated_volatility = np.mean(simulated_volatilities) * np.sqrt(365)  

#         # Empirical metrics
#         empirical_mean = self.empirical_data['utilisation_ratio'].mean()
#         empirical_volatility = self.empirical_data['utilisation_ratio'].std() * np.sqrt(365)

#         # Tolerances for comparison
#         tolerance_mean = 0.05
#         tolerance_volatility = 0.05

#         # Assertions
#         self.assertAlmostEqual(simulated_mean, empirical_mean, delta=tolerance_mean,
#                                msg=f"Simulated mean {simulated_mean} deviates from empirical mean {empirical_mean}")
#         self.assertAlmostEqual(simulated_volatility, empirical_volatility, delta=tolerance_volatility,
#                                msg=f"Simulated volatility {simulated_volatility} deviates from empirical volatility {empirical_volatility}")


# if __name__ == "__main__":
#     unittest.main()


import unittest
import numpy as np
import pandas as pd
from scipy.stats import zscore
import statsmodels.api as sm
from src.models.utilisation_simulator import UtilizationSimulator
from src.interest_rate_models.semilog_irm import SemiLogIRM
import requests


def retrieve_lending_market(chain: str = 'ethereum', controller_address: str = '0xeda215b7666936ded834f76f3fbc6f323295110a'):
    url = f'https://prices.curve.fi/v1/lending/markets/{chain}/{controller_address}/snapshots?fetch_on_chain=false&agg=day'
    r = requests.get(url)
    df = pd.json_normalize(r.json()['data'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


class TestUtilizationSimulator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Setup class-level attributes for use in the tests.
        """
        last_x_months = 5
        cls.df = retrieve_lending_market()
        cls.df['utilisation_ratio'] = cls.df['total_debt_usd'] / cls.df['total_assets_usd']
        cls.empirical_data = cls.df[cls.df['timestamp'] >= cls.df['timestamp'].max() - pd.DateOffset(months=last_x_months)]

        # OLS Analysis for Parameters
        cls.empirical_data['delta_utilisation'] = cls.empirical_data['utilisation_ratio'].diff()
        cls.empirical_data = cls.empirical_data.dropna(subset=['delta_utilisation'])
        cls.empirical_data = cls.empirical_data[
            (np.abs(zscore(cls.empirical_data['delta_utilisation'])) < 2)
        ]

        regression_data = cls.empirical_data.dropna(subset=['delta_utilisation', 'borrow_apy'])
        regression_data['lagged_avg_rate'] = regression_data['borrow_apy'].shift(1)
        regression_data = regression_data.dropna(subset=['lagged_avg_rate'])

        # Perform OLS regression
        X = sm.add_constant(regression_data['lagged_avg_rate'])
        y = regression_data['delta_utilisation']
        model = sm.OLS(y, X).fit()

        # Annualized parameters
        cls.b0_annualized = model.params['const'] * 365
        cls.b1_annualized = model.params['lagged_avg_rate'] * 365
        cls.sigma_annualized = regression_data['delta_utilisation'].std() * np.sqrt(365)
        

        # Print OLS Summary
        print("### OLS Regression Summary ###")
        print(model.summary())
        print("\n### Derived Parameters ###")
        print(f"Annualized Drift Parameter (b0): {cls.b0_annualized:.4f}")
        print(f"Annualized Borrower Sensitivity (b1): {cls.b1_annualized:.4e}")
        print(f"Annualized Volatility (σ): {cls.sigma_annualized:.4f}")

        # Simulation parameters
        cls.semilog_irm = SemiLogIRM(rate_min=0.01, rate_max=10.0)
        cls.params = {
            "U0": cls.empirical_data['utilisation_ratio'].iloc[0],
            "T": 1.0,
            "dt": 1 / 365,
            "rate_model": cls.semilog_irm.calculate_rate,
            "b0": cls.b0_annualized,
            "b1": cls.b1_annualized,
            "sigma": cls.sigma_annualized
        }

    def test_simulation_matches_empirical_data_stabilized(self):
        """
        Test if the simulation's metrics for the last 5 months match empirical data.
        """
        last_x_months = 5
        num_simulations = 100
        last_months_index = int((self.params["T"] / self.params["dt"]) * (1 - last_x_months / 12))
        simulated_means = []
        simulated_volatilities = []

        for _ in range(num_simulations):
            simulator = UtilizationSimulator(**self.params)
            utilization_path, _ = simulator.simulate()

            # Metrics for the last 5 months
            last_months = utilization_path[last_months_index:]
            simulated_means.append(np.mean(last_months))
            simulated_volatilities.append(np.std(last_months))

        # Aggregate metrics
        simulated_mean = np.mean(simulated_means)
        simulated_volatility = np.mean(simulated_volatilities) * np.sqrt(365)

        # Empirical metrics
        empirical_mean = self.empirical_data['utilisation_ratio'].mean()
        empirical_volatility = self.empirical_data['utilisation_ratio'].std() * np.sqrt(365)

        # Tolerances
        tolerance_mean = 0.05
        tolerance_volatility = 0.05

        # Assertions
        self.assertAlmostEqual(simulated_mean, empirical_mean, delta=tolerance_mean,
                               msg=f"Simulated mean {simulated_mean} deviates from empirical mean {empirical_mean}")
        self.assertAlmostEqual(simulated_volatility, empirical_volatility, delta=tolerance_volatility,
                               msg=f"Simulated volatility {simulated_volatility} deviates from empirical volatility {empirical_volatility}")


if __name__ == "__main__":
    unittest.main()

# import unittest
# import numpy as np
# import pandas as pd
# from scipy.stats import zscore
# import statsmodels.api as sm
# from src.models.utilisation_simulator import UtilizationSimulator
# from src.interest_rate_models.semilog_irm import SemiLogIRM
# import requests

# def retrieve_lending_market(chain: str = 'ethereum', controller_address: str = '0xeda215b7666936ded834f76f3fbc6f323295110a'):
#     url = f'https://prices.curve.fi/v1/lending/markets/{chain}/{controller_address}/snapshots?fetch_on_chain=false&agg=day'
#     r = requests.get(url)
#     df = pd.json_normalize(r.json()['data'])
#     df['timestamp'] = pd.to_datetime(df['timestamp'])
#     return df

# class TestUtilizationSimulator(unittest.TestCase):

#     @classmethod
#     def setUpClass(cls):
#         """
#         Setup class-level attributes for use in the tests.
#         """
#         last_x_months = 5
#         cls.df = retrieve_lending_market()
#         cls.df['utilisation_ratio'] = cls.df['total_debt_usd'] / cls.df['total_assets_usd']
#         cls.empirical_data = cls.df[cls.df['timestamp'] >= cls.df['timestamp'].max() - pd.DateOffset(months=last_x_months)]

#         # Compute ΔU_t
#         cls.empirical_data['delta_utilisation'] = cls.empirical_data['utilisation_ratio'].diff()
#         cls.empirical_data = cls.empirical_data.dropna(subset=['delta_utilisation'])
#         cls.empirical_data = cls.empirical_data[(np.abs(zscore(cls.empirical_data['delta_utilisation'])) < 2)]

#         # Regression for A and C
#         r0 = 0.30  # Borrower alternative rate
#         r1 = 0.20  # Supplier alternative rate
#         cls.empirical_data['X1'] = cls.empirical_data['utilisation_ratio'] * (r0 - cls.empirical_data['borrow_apy']/100)
#         cls.empirical_data['X2'] = (1 - cls.empirical_data['utilisation_ratio']) * (cls.empirical_data['borrow_apy']/100 - r1)

#         # Prepare regression data
#         regression_data = cls.empirical_data[['delta_utilisation', 'X1', 'X2']].dropna()
#         X = sm.add_constant(regression_data[['X1', 'X2']])
#         y = regression_data['delta_utilisation']

#         # Perform OLS
#         model = sm.OLS(y, X).fit()

#         # Extract parameters
#         cls.b0_annualized = model.params['const'] * 365
#         cls.A = model.params['X1'] * 365
#         cls.C = -model.params['X2'] * 365  # Negate to match the equation
#         cls.sigma_annualized = regression_data['delta_utilisation'].std() * np.sqrt(365)

#         # Debug: Print OLS results
#         print("### OLS Regression Summary ###")
#         print(model.summary())
#         print("\n### Derived Parameters ###")
#         print(f"Annualized Drift (b0): {cls.b0_annualized:.4f}")
#         print(f"Borrower Sensitivity (A): {cls.A:.4f}")
#         print(f"Supplier Sensitivity (C): {cls.C:.4f}")
#         print(f"Volatility (σ): {cls.sigma_annualized:.4f}")

#         # Simulation parameters
#         cls.semilog_irm = SemiLogIRM(rate_min=0.01, rate_max=10.0)
#         cls.params = {
#             "U0": cls.empirical_data['utilisation_ratio'].iloc[0],  # Initial utilization
#             "T": 1.0,
#             "dt": 1 / 365,  # Daily time step
#             "rate_model": cls.semilog_irm.calculate_rate,
#             "b0": cls.b0_annualized,
#             "A": cls.A,
#             "C": cls.C,
#             "r0": r0,
#             "r1": r1,
#             "sigma_const": cls.sigma_annualized,
#         }

#     def test_simulation_matches_empirical_data_stabilized(self):
#         """
#         Test if the simulation's metrics for the last 6 months match empirical data.
#         """
#         last_x_months = 5
#         num_simulations = 1000
#         last_months_index = int((self.params["T"] / self.params["dt"]) * (1 - last_x_months / 12))
#         simulated_means = []
#         simulated_volatilities = []

#         for _ in range(num_simulations):
#             simulator = UtilizationSimulator(
#                 U0=self.params["U0"],
#                 T=self.params["T"],
#                 dt=self.params["dt"],
#                 rate_model=self.params["rate_model"],
#                 # b0=self.params['b0'],
#                 A=self.params["A"],
#                 C=self.params["C"],
#                 r0=self.params["r0"],
#                 r1=self.params["r1"],
#                 sigma_const=self.params["sigma_const"]
#             )
#             utilization_path, _ = simulator.simulate()

#             # Metrics for the last 5 months
#             last_months = utilization_path[last_months_index:]
#             simulated_means.append(np.mean(last_months))
#             simulated_volatilities.append(np.std(last_months))

#         # Aggregate metrics
#         simulated_mean = np.mean(simulated_means)
#         simulated_volatility = np.mean(simulated_volatilities) * np.sqrt(365)

#         # Empirical metrics
#         empirical_mean = self.empirical_data['utilisation_ratio'].mean()
#         empirical_volatility = self.empirical_data['utilisation_ratio'].std() * np.sqrt(365)

#         # Debug: Print metrics
#         print(f"Simulated Mean: {simulated_mean}, Empirical Mean: {empirical_mean}")
#         print(f"Simulated Volatility: {simulated_volatility}, Empirical Volatility: {empirical_volatility}")

#         # Tolerances
#         tolerance_mean = 0.05
#         tolerance_volatility = 0.05

#         # Assertions
#         self.assertAlmostEqual(simulated_mean, empirical_mean, delta=tolerance_mean,
#                                msg=f"Simulated mean {simulated_mean} deviates from empirical mean {empirical_mean}")
#         self.assertAlmostEqual(simulated_volatility, empirical_volatility, delta=tolerance_volatility,
#                                msg=f"Simulated volatility {simulated_volatility} deviates from empirical volatility {empirical_volatility}")


# if __name__ == "__main__":
#     unittest.main()
