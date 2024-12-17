# simulation.py

import numpy as np
from src.interest_rate_models.base_irm import BaseIRM

def inv_logit(w: float) -> float:
    return 1 / (1 + np.exp(-w))

# def simulate_utilization_paths(
#     alpha: float,
#     rho: float,
#     b1: float,
#     sigma: float,
#     W0: float,
#     num_steps: int,
#     num_paths: int,
#     controller_type: BaseIRM
# ) -> np.ndarray:
#     """
#     Simulate multiple utilization paths based on the chosen IRM.
    
#     Parameters:
#     - irm: Instance of an IRM (SemiLogIRM or PControllerIRM)
#     - alpha, rho, b1, sigma: Parameters from the W model
#     - W0: Initial W value
#     - num_steps: Number of simulation steps
#     - num_paths: Number of simulation paths
#     - target_u: Target utilization ratio (used for P-Controller)
#     - controller_type: Type of IRM ('semilog' or 'p_controller')
    
#     Returns:
#     - sim_U_paths: Simulated utilization paths (num_paths x num_steps)
#     """
#     sim_W_paths = np.zeros((num_paths, num_steps))
#     for i in range(num_paths):
#         W_sim = np.zeros(num_steps)
#         W_sim[0] = W0
#         for t in range(1, num_steps):
#             current_w = W_sim[t-1]
#             # Compute current U from W
#             current_u = inv_logit(current_w)
#             r_current = controller_type.calculate_rate(current_u)
#             # W_t = alpha + rho*W_{t-1} - b1*r(U_{t-1}) + sigma * Z
#             W_t = alpha + rho*current_w - b1*r_current + sigma*np.random.randn()
#             W_sim[t] = W_t
#         sim_W_paths[i,:] = W_sim

#     sim_paths = inv_logit(sim_W_paths)
    
#     return sim_paths


def simulate_utilization_paths(
    alpha: float,
    rho: float,
    b1: float,
    sigma: float,
    W0: float,
    num_steps: int,
    num_paths: int,
    controller_type: BaseIRM
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate multiple utilization paths and rate paths based on the chosen IRM.

    Parameters:
    - alpha, rho, b1, sigma: Parameters from the W model
    - W0: Initial W value
    - num_steps: Number of simulation steps
    - num_paths: Number of simulation paths
    - controller_type: Instance of an IRM (SemiLogIRM or PControllerIRM)

    Returns:
    - sim_U_paths: Simulated utilization paths (num_paths x num_steps)
    - sim_R_paths: Simulated rate paths (num_paths x num_steps)
    """
    sim_W_paths = np.zeros((num_paths, num_steps))
    sim_R_paths = np.zeros((num_paths, num_steps))

    for i in range(num_paths):
        W_sim = np.zeros(num_steps)
        R_sim = np.zeros(num_steps)
        W_sim[0] = W0

        for t in range(1, num_steps):
            current_w = W_sim[t - 1]
            # Compute current U from W
            current_u = inv_logit(current_w)
            # Compute the rate using the IRM controller
            r_current = controller_type.calculate_rate(current_u)
            R_sim[t] = r_current
            # Update W_t using the given stochastic model
            W_t = alpha + rho * current_w - b1 * r_current + sigma * np.random.randn()
            W_sim[t] = W_t

        sim_W_paths[i, :] = W_sim
        sim_R_paths[i, :] = R_sim

    sim_U_paths = inv_logit(sim_W_paths)

    return sim_U_paths, sim_R_paths
