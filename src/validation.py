# validation.py

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from src.simulation import simulate_utilization_paths
from src.metrics import calculate_metrics
from src.interest_rate_models.semilog_irm import SemiLogIRM
from typing import Tuple

def validate_optimized_irm(
    optimized_params: Tuple,
    df: pd.DataFrame,
    alpha: float,
    rho: float,
    b1: float,
    sigma: float,
    irm_class: type = SemiLogIRM,
    num_steps: int = 180,
    num_paths: int = 5000,
    target_u: float = 0.9,
    threshold_u: float = 0.95
) -> pd.DataFrame:
    """
    Validate the optimized IRM parameters by running extensive simulations.
    
    Parameters:
    - optimized_params: Tuple containing (rate_min, rate_max)
    - df: DataFrame with calibrated parameters
    - alpha, rho, b1, sigma: W model parameters
    - irm_class: IRM class
    - num_steps: Number of simulation steps
    - num_paths: Number of simulation paths
    - target_u: Target utilization ratio
    - threshold_u: Threshold utilization ratio
    
    Returns:
    - metrics_df: DataFrame containing metrics for all simulations
    """
    irm = irm_class(*optimized_params)
    
    # Simulate paths
    sim_U_paths = simulate_utilization_paths(
        alpha=alpha,
        rho=rho,
        b1=b1,
        sigma=sigma,
        W0=df['W'].iloc[-1],
        num_steps=num_steps,
        num_paths=num_paths,
        controller_type=irm
    )
    
    # Calculate metrics for each path
    metrics = {
        'MSE': [],
        'Time_Above_Threshold': [],
        'Volatility_U': []
    }
    
    for path in sim_U_paths:
        m = calculate_metrics(path, target_u, threshold_u)
        metrics['MSE'].append(m['MSE'])
        metrics['Time_Above_Threshold'].append(m['Time_Above_Threshold'])
        metrics['Volatility_U'].append(m['Volatility_U'])
    
    metrics_df = pd.DataFrame(metrics)
    
    # # Summary Statistics
    # print("\nOptimized IRM Metrics Summary:")
    # print(metrics_df.describe())
    
    # # Visualizations
    # metrics_to_plot = ['MSE', 'Time_Above_Threshold', 'Volatility_U']
    # for metric in metrics_to_plot:
    #     plt.figure(figsize=(10, 6))
    #     sns.histplot(metrics_df[metric], bins=50, kde=True, color='green')
    #     plt.title(f'Distribution of {metric} - Optimized IRM')
    #     plt.xlabel(metric)
    #     plt.ylabel('Frequency')
    #     plt.show()
    
    return metrics_df
