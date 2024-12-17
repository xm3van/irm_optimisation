# src/interest_rate_models/p_controller_irm.py

from src.interest_rate_models.base_irm import BaseIRM

class PControllerIRM(BaseIRM):
    def __init__(self, phi_low: float = 0.9, phi_high: float = 1.1, threshold: float = 0.05, target_u: float = 0.9):
        """
        Initialize the P-Controller IRM.
        
        Parameters:
        - phi_low: Factor to decrease rate (e.g., 0.9 for -10%)
        - phi_high: Factor to increase rate (e.g., 1.1 for +10%)
        - threshold: Deviation threshold from target utilization
        - target_u: Target utilization ratio
        """
        self.phi_low = phi_low
        self.phi_high = phi_high
        self.threshold = threshold
        self.target_u = target_u
    
    def calculate_rate(self, utilization: float, current_rate: float = None, prev_derivative: float = None) -> float:
        """
        Adjust the interest rate based on utilization deviation.
        
        Parameters:
        - utilization: Current utilization ratio (U_t)
        - current_rate: Current interest rate (r_t) [Required for P-Controller]
        
        Returns:
        - Adjusted interest rate (r_t')
        """
        if current_rate is None:
            raise ValueError("current_rate must be provided for P-Controller IRM.")
        
        deviation = utilization - self.target_u
        if deviation <= -self.threshold:
            phi = self.phi_low
        elif deviation >= self.threshold:
            phi = self.phi_high
        else:
            phi = 1.0  # No change
        
        new_rate = current_rate * phi
        return max(new_rate, 0.0001)  # Ensure rate stays positive
