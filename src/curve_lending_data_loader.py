import requests
import pandas as pd
import numpy as np 
from src.interest_rate_models.semilog_irm import SemiLogIRM

class CurveLendingMarketDataLoader:
    """
    A class to retrieve and process daily lending market data from the Curve API.
    """
    def __init__(self, chain: str = 'ethereum', controller_address: str = '0xeda215b7666936ded834f76f3fbc6f323295110a'):
        """
        Initialize the DataLoader with chain and controller address.

        :param chain: Blockchain network (default: 'ethereum').
        :param controller_address: Address of the controller (default: '0xeda215b7666936ded834f76f3fbc6f323295110a').
        """
        self.chain = chain
        self.controller_address = controller_address
        self.base_url = "https://prices.curve.fi/v1/lending/markets"

    def retrieve_data(self) -> pd.DataFrame:
        """
        Fetch daily lending market data from the Curve API and return it as a DataFrame.

        :return: DataFrame containing the lending market data.
        """
        url = f"{self.base_url}/{self.chain}/{self.controller_address}/snapshots?fetch_on_chain=false&agg=day"
        response = requests.get(url)

        if response.status_code != 200:
            raise ValueError(f"Failed to fetch data: {response.status_code} - {response.text}")

        data = response.json().get('data', [])

        if not data:
            raise ValueError("No data found in the API response.")

        df = pd.json_normalize(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    
    def enrich_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enrich the DataFrame with additional calculations and transformations.

        :param df: Raw DataFrame to enrich.
        :return: Enriched DataFrame.
        """
        # Calculate Utilisation ratio
        df['utilisation_ratio'] = df['total_debt_usd'] / df['total_assets_usd']

        # Calculate borrow APR
        df['borrowAPR'] = (df['rate'] * 365 * 86400) / 1e18

        # ensure order 
        df = df.sort_values(by='timestamp').reset_index(drop=True)

        # Logit transform of utilization
        def logit(u):
            return np.log(u / (1 - u))

        def inv_logit(w):
            return 1 / (1 + np.exp(-w))

        df['W'] = df['utilisation_ratio'].apply(logit)

        # Set up SEMI-LOG IRM
        rate_model = SemiLogIRM(rate_min=0.0001, rate_max=10)  # Current market parameters
        rate_function = rate_model.calculate_rate

        # Estimate W model
        df['W_lag'] = df['W'].shift(1)
        df.dropna(inplace=True)

        # Compute r(U_{t-1})
        df['U_lag'] = df['W_lag'].apply(inv_logit)
        df['r_u_lag'] = df['U_lag'].apply(rate_function)

        return df

# # Example usage
# if __name__ == "__main__":
#     loader = CurveLendingMarketDataLoader()
#     df = loader.retrieve_data()
#     print(df.head())
