# metrics.py

import numpy as np

def calculate_metrics(simulated_U: np.ndarray, target_u: float = 0.9, threshold_u: float = 0.95) -> dict:
    """
    Calculate key metrics for a simulated utilization path.
    
    Parameters:
    - simulated_U: Array of simulated utilization ratios
    - target_u: Target utilization ratio
    - threshold_u: Threshold utilization ratio
    
    Returns:
    - metrics: Dictionary containing MSE, Time Above Threshold, Volatility of U
    """
    mse = np.mean((simulated_U - target_u) ** 2)
    time_above_threshold = np.sum(simulated_U > threshold_u)
    volatility_u = np.std(simulated_U)
    
    return {
        'MSE': mse,
        'Time_Above_Threshold': time_above_threshold,
        'Volatility_U': volatility_u
    }

def composite_objective(metrics: dict, weights: dict = {'MSE': 1.0, 'Time_Above_Threshold': 1.0, 'Volatility_U': 1.0}) -> float:
    """
    Combine multiple metrics into a single loss value.
    
    Parameters:
    - metrics: Dictionary of metric values
    - weights: Dictionary of weights for each metric
    
    Returns:
    - loss: Weighted sum of metrics
    """
    loss = 0.0
    for key, value in metrics.items():
        loss += weights.get(key, 1.0) * value
    return loss
