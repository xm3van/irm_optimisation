from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from src.simulation import simulate_utilization_paths
from src.metrics import calculate_metrics, composite_objective
import pandas as pd
import numpy as np
from typing import Tuple, Type, List


# def optimize_irm_parameters(
#     df: pd.DataFrame,
#     irm_class: Type,
#     search_space: List[Real],
#     update_params: callable,
#     target_u: float = 0.9,
#     threshold_u: float = 0.95,
#     num_steps: int = 180,
#     num_paths: int = 100,
#     weights: dict = {'MSE': 1.0, 'Time_Above_Threshold': 1, 'Volatility_U': 1},
#     n_calls: int = 30,
#     model_input: set = (1, 1, 1, 1),
#     random_state: int = 42
# ) -> Tuple[float, ...]:
#     """
#     Generalized optimization function for various IRM models.
    
#     Parameters:
#     - df: DataFrame with calibrated parameters
#     - irm_class: IRM class to optimize
#     - search_space: List of skopt.Real defining the parameter search space
#     - update_params: Callable to update IRM parameters (specific to model)
#     - target_u: Target utilization ratio
#     - threshold_u: Threshold utilization ratio
#     - num_steps: Number of simulation steps
#     - num_paths: Number of simulation paths per parameter set
#     - weights: Weights for the objective function
#     - n_calls: Number of optimization iterations
#     - model_input: Parameters for the W model
#     - random_state: Seed for reproducibility
    
#     Returns:
#     - Optimized parameters as a tuple
#     """
#     alpha, rho, b1, sigma = model_input

#     # Initialize the IRM
#     irm = irm_class()
    
#     @use_named_args(search_space)
#     def objective(**kwargs):
#         # Update IRM parameters
#         update_params(irm, **kwargs)
        
#         # Simulate multiple paths
#         sim_U_paths = simulate_utilization_paths(
#             alpha=alpha,
#             rho=rho,
#             b1=b1,
#             sigma=sigma,
#             W0=df['W'].iloc[-1],
#             num_steps=num_steps,
#             num_paths=num_paths,
#             controller_type=irm
#         )
        
#         # Calculate metrics across all paths
#         mse_total = 0.0
#         time_above_threshold_total = 0.0
#         vol_u_total = 0.0
        
#         for path in sim_U_paths:
#             metrics = calculate_metrics(path, target_u, threshold_u)
#             mse_total += metrics['MSE']
#             time_above_threshold_total += metrics['Time_Above_Threshold']
#             vol_u_total += metrics['Volatility_U']
        
#         # Average metrics
#         avg_metrics = {
#             'MSE': mse_total / num_paths,
#             'Time_Above_Threshold': time_above_threshold_total / num_paths,
#             'Volatility_U': vol_u_total / num_paths
#         }
        
#         # Compute loss
#         loss = composite_objective(avg_metrics, weights)
#         return loss
    
#     # Perform Bayesian Optimization
#     res = gp_minimize(
#         func=objective,
#         dimensions=search_space,
#         n_calls=n_calls,
#         random_state=random_state,
#         verbose=True
#     )
    
#     print(f"Optimized parameters: {res.x}")
#     return res

from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from src.simulation import simulate_utilization_paths
from src.metrics import calculate_metrics, composite_objective
import pandas as pd
import numpy as np
from typing import Tuple, Type, List


def optimize_irm_parameters(
    df: pd.DataFrame,
    irm_class: Type,
    search_space: List[Real],
    update_params: callable,
    target_u: float = 0.9,
    threshold_u: float = 0.95,
    num_steps: int = 180,
    num_paths: int = 100,
    weights: dict = {'MSE': 1.0, 'Time_Above_Threshold': 1, 'Volatility_U': 1},
    n_calls: int = 30,
    model_input: set = (1, 1, 1, 1),
    random_state: int = 42
) -> Tuple[float, ...]:
    """
    Generalized optimization function for various IRM models.
    """
    alpha, rho, b1, sigma = model_input

    # Initialize the IRM
    irm = irm_class()
    
    @use_named_args(search_space)
    def objective(**kwargs):
        # Extract rate_min and delta_rate
        rate_min = kwargs['rate_min']
        delta_rate = kwargs['delta_rate']
        
        # Compute rate_max
        rate_max = rate_min + delta_rate
        
        # Enforce constraints
        if rate_min >= rate_max or rate_max > 10:
            return 1e6  # Large penalty for invalid configurations
        
        # Update IRM parameters
        update_params(irm, rate_min=rate_min, delta_rate=delta_rate)
        
        # Simulate multiple paths
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
        
        # Calculate metrics across all paths
        mse_total = 0.0
        time_above_threshold_total = 0.0
        vol_u_total = 0.0
        
        for path in sim_U_paths:
            metrics = calculate_metrics(path, target_u, threshold_u)
            mse_total += metrics['MSE']
            time_above_threshold_total += metrics['Time_Above_Threshold']
            vol_u_total += metrics['Volatility_U']
        
        # Average metrics
        avg_metrics = {
            'MSE': mse_total / num_paths,
            'Time_Above_Threshold': time_above_threshold_total / num_paths,
            'Volatility_U': vol_u_total / num_paths
        }
        
        # Compute loss
        loss = composite_objective(avg_metrics, weights)
        return loss
    
    # Perform Bayesian Optimization
    res = gp_minimize(
        func=objective,
        dimensions=search_space,
        n_calls=n_calls,
        random_state=random_state,
        verbose=True
    )
    
    print(f"Optimized parameters: {res.x}")
    return res


from scipy.optimize import differential_evolution

def differential_evolution_optimizer(
    df: pd.DataFrame,
    irm_class: Type,
    bounds: List[Tuple[float, float]],
    update_params: callable,
    target_u: float = 0.9,
    threshold_u: float = 0.95,
    num_steps: int = 180,
    num_paths: int = 100,
    weights: dict = {'MSE': 1.0, 'Time_Above_Threshold': 1, 'Volatility_U': 1},
    max_iter: int = 30,
    popsize: int = 15,
    model_input: set = (1, 1, 1, 1),
    seed: int = 42
):
    """
    Optimizes IRM parameters using Differential Evolution.
    """
    alpha, rho, b1, sigma = model_input
    
    # Initialize the IRM
    irm = irm_class()
    
    def objective(params):
        min_rate, max_rate = params

        update_params(irm, min_rate, max_rate)
        
        
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
        
        # Calculate metrics across paths
        mse_total = 0.0
        time_above_threshold_total = 0.0
        vol_u_total = 0.0
        
        for path in sim_U_paths:
            metrics = calculate_metrics(path, target_u, threshold_u)
            mse_total += metrics['MSE']
            time_above_threshold_total += metrics['Time_Above_Threshold']
            vol_u_total += metrics['Volatility_U']
        
        # Average metrics
        avg_metrics = {
            'MSE': mse_total / num_paths,
            'Time_Above_Threshold': time_above_threshold_total / num_paths,
            'Volatility_U': vol_u_total / num_paths
        }
        
        # Composite loss
        loss = composite_objective(avg_metrics, weights)
        return loss
    
    # Run Differential Evolution
    result = differential_evolution(
        func=objective,
        bounds=bounds,
        maxiter=max_iter,
        popsize=popsize,
        seed=seed
    )
    
    print(f"Optimized parameters: {result.x}")
    return result


import cma
import pandas as pd
import numpy as np
from typing import Tuple, Type, List

def cmaes_optimizer(
    df: pd.DataFrame,
    irm_class: Type,
    update_params: callable,
    bounds: List[Tuple[float, float]],
    target_u: float = 0.9,
    threshold_u: float = 0.95,
    num_steps: int = 180,
    num_paths: int = 100,
    weights: dict = {'MSE': 1.0, 'Time_Above_Threshold': 1, 'Volatility_U': 1},
    max_iter: int = 100,
    sigma: float = 0.5,
    random_state: int = 42,
    model_input: Tuple[float, float, float, float] = (1, 1, 1, 1)
):
    """
    Optimizes IRM parameters using CMA-ES.

    Parameters:
    - df: DataFrame used for calibration.
    - irm_class: IRM class to optimize.
    - update_params: Callable to update IRM parameters.
    - bounds: List of (min, max) bounds for the parameters.
    - target_u: Target utilization ratio.
    - threshold_u: Threshold utilization ratio.
    - num_steps: Number of simulation steps.
    - num_paths: Number of paths for simulations.
    - weights: Weights for the composite objective function.
    - max_iter: Maximum number of CMA-ES iterations.
    - sigma: Initial step size for CMA-ES.
    - random_state: Random seed for reproducibility.
    - model_input: Input model parameters (alpha, rho, b1, sigma).

    Returns:
    - best_params: Optimized parameters [rate_min, rate_max].
    - best_value: Optimal objective function value.
    """
    np.random.seed(random_state)
    alpha, rho, b1, sigma_input = model_input

    irm = irm_class()

    # Objective function for CMA-ES
    def objective(params):
        rate_min, rate_max = params
        
        # Enforce constraints
        if rate_min >= rate_max or rate_min < bounds[0][0] or rate_max > bounds[1][1]:
            return 1e6  # Large penalty for invalid configurations
        
        # Update IRM parameters
        update_params(irm, rate_min=rate_min, rate_max = rate_max)

        # Simulate utilization paths
        sim_U_paths = simulate_utilization_paths(
            alpha=alpha,
            rho=rho,
            b1=b1,
            sigma=sigma_input,
            W0=df['W'].iloc[-1],
            num_steps=num_steps,
            num_paths=num_paths,
            controller_type=irm
        )

        # Calculate metrics
        mse_total, time_above_threshold_total, vol_u_total = 0.0, 0.0, 0.0
        for path in sim_U_paths:
            metrics = calculate_metrics(path, target_u, threshold_u)
            mse_total += metrics['MSE']
            time_above_threshold_total += metrics['Time_Above_Threshold']
            vol_u_total += metrics['Volatility_U']

        avg_metrics = {
            'MSE': mse_total / num_paths,
            'Time_Above_Threshold': time_above_threshold_total / num_paths,
            'Volatility_U': vol_u_total / num_paths
        }

        return composite_objective(avg_metrics, weights)

    # Initial guess (center of bounds)
    x0 = [np.mean(bound) for bound in bounds]

    # Configure CMA-ES
    es = cma.CMAEvolutionStrategy(
        x0=x0,
        sigma0=sigma,
        inopts={
            'bounds': [list(bound[0] for bound in bounds), list(bound[1] for bound in bounds)],
            'maxiter': max_iter,
            'seed': random_state
        }
    )

    # Run optimization
    es.optimize(objective)

    # Results
    best_params = es.result.xbest
    best_value = es.result.fbest

    print(f"Optimized parameters: {best_params}, Objective value: {best_value}")
    return es
