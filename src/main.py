# main.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from src.interest_rate_models.semilog_irm import SemiLogIRM
from src.interest_rate_models.p_controller_irm import PControllerIRM
from src.calibration import calibrate_w_model
from src.metrics import calculate_metrics, composite_objective
from src.optimization import optimize_semi_log_irm
from src.validation import validate_optimized_irm
from src.simulation import simulate_utilization_paths
from src.curve_lending_data_loader import CurveLendingMarketDataLoader

# Step 1: Load and Prepare Data

loader = CurveLendingMarketDataLoader(chain='ethereum', controller_address='0xeda215b7666936ded834f76f3fbc6f323295110a')
df = loader.retrieve_data()

df['utilisation_ratio'] = df['total_debt_usd'] / df['total_assets_usd']


# Compute 'borrowAPR' as per your formula
df['borrowAPR'] = (df['rate'] * 365 * 86400) / 1e18  # Adjust based on actual formula

# Filter data
df = df[(df['utilisation_ratio'] > 0) & (df['utilisation_ratio'] < 1) & (df['borrowAPR'] > 0)]
df = df.sort_values(by='timestamp').reset_index(drop=True)

# Logit transform of utilization
def logit(u):
    return np.log(u / (1 - u))

df['W'] = df['utilisation_ratio'].apply(logit)



# Initialize IRMs
rate_model_semilog = SemiLogIRM(rate_min=0.0001, rate_max=0.5)
rate_model_p = PControllerIRM(phi_low=0.9, phi_high=1.1, threshold=0.05, target_u=0.9)

# Step 3: Estimate W Model Parameters
alpha, rho, b1, sigma = calibrate_w_model(df)
print("Estimated W model parameters:")
print(f"alpha = {alpha}")
print(f"rho = {rho}")
print(f"b1 = {b1}")
print(f"sigma = {sigma}")

# Step 4: Simulate Synthetic Paths with Semi-Log IRM
num_steps = len(df)
W0 = df['W'].iloc[0]
num_paths = 1000

sim_U_paths_semilog = simulate_utilization_paths(
    irm=rate_model_semilog,
    alpha=alpha,
    rho=rho,
    b1=b1,
    sigma=sigma,
    W0=W0,
    num_steps=num_steps,
    num_paths=num_paths,
    target_u=0.9,
    controller_type='semilog'
)

# Convert W to U
inv_logit = lambda w: 1 / (1 + np.exp(-w))
sim_U_paths_semilog = inv_logit(sim_U_paths_semilog)

# Plot average simulated U vs empirical U
avg_sim_U_semilog = sim_U_paths_semilog.mean(axis=0)
empirical_U = df['utilisation_ratio'].values

plt.figure(figsize=(12, 6))
plt.plot(empirical_U, label='Empirical U', alpha=0.7)
plt.plot(avg_sim_U_semilog, label='Simulated U (Semi-Log IRM)', alpha=0.7)
plt.title('Empirical vs Simulated Utilization Ratio')
plt.xlabel('Time (Days)')
plt.ylabel('Utilization Ratio')
plt.legend()
plt.show()

# Step 5: Optimize Semi-Log IRM Parameters
# Assuming the last 6 months of data for calibration
df_last6 = df.tail(180).reset_index(drop=True)
optimized_rate_min, optimized_rate_max = optimize_semi_log_irm(
    df=df_last6,
    irm_class=SemiLogIRM,
    target_u=0.9,
    threshold_u=0.95,
    num_steps=180,
    num_paths=100,
    weights={'MSE': 1.0, 'Time_Above_Threshold': 0.5, 'Volatility_U': 0.5},
    n_calls=30,
    random_state=42
)

# Step 6: Validate the Optimized Model
metrics_df = validate_optimized_irm(
    optimized_params=(optimized_rate_min, optimized_rate_max),
    df=df_last6,
    alpha=alpha,
    rho=rho,
    b1=b1,
    sigma=sigma,
    irm_class=SemiLogIRM,
    num_steps=180,
    num_paths=5000,
    target_u=0.9,
    threshold_u=0.95
)
