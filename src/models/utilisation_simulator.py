# import numpy as np

# class UtilizationSimulator:
#     """
#     Simulates utilization dynamics using a stochastic differential equation (SDE).
    
#     Attributes:
#         U0 (float): Initial utilization ratio.
#         T (float): Total simulation time.
#         dt (float): Time step.
#         rate_model (function): Interest rate model function.
#         A (float): Borrower sensitivity.
#         C (float): Lender sensitivity.
#         r0 (float): Borrower alternative rate.
#         r1 (float): Lender alternative rate.
#         sigma_const (float): Volatility constant.
#         params (list): Parameters for the rate model.
#     """
    
#     def __init__(self, U0, T, dt, rate_model, A, C, r0, r1, sigma_const):
#         self.U0 = U0
#         self.T = T
#         self.dt = dt
#         self.rate_model = rate_model
#         self.A = A
#         self.C = C
#         self.r0 = r0
#         self.r1 = r1
#         self.sigma_const = sigma_const
    
#     @staticmethod
#     def drift(U, r, A, C, r0, r1):
#         """
#         Compute the drift term for utilization dynamics.
        
#         Parameters:
#             U (float): Current utilization ratio.
#             r (float): Current interest rate.
#             A (float): Borrower sensitivity.
#             C (float): Lender sensitivity.
#             r0 (float): Borrower alternative rate.
#             r1 (float): Lender alternative rate.
        
#         Returns:
#             float: Drift value for utilization.
#         """
#         return A * U * (r0 - r) - C * (1 - U) * (r - r1)

#     @staticmethod
#     def volatility(U, sigma_const):
#         """
#         Compute the volatility term for utilization dynamics.
        
#         Parameters:
#             U (float): Current utilization ratio.
#             sigma_const (float): Volatility constant.
        
#         Returns:
#             float: Volatility value for utilization.
#         """
#         return sigma_const * U

#     def simulate(self):
#         """
#         Simulate utilization dynamics using the specified parameters.
        
#         Returns:
#             tuple: Two np.arrays containing the utilization path and interest rate path over time.
#         """
#         n_steps = int(self.T / self.dt)
#         U = self.U0
#         r = self.rate_model(U)  # Calculate initial rate

#         path_U = [U]
#         path_r = [r]

#         for _ in range(n_steps):
#             dW = np.random.normal(0, np.sqrt(self.dt))  # Brownian motion increment
#             r = self.rate_model(U)  # Compute interest rate using the model
#             path_r.append(r)  # Record the interest rate
#             dU_drift = self.drift(U, r, self.A, self.C, self.r0, self.r1) * self.dt
#             dU_volatility = self.volatility(U, self.sigma_const) * dW
#             dU = dU_drift + dU_volatility
#             U = max(0, min(1, U + dU))  # Constrain U to [0, 1]
#             path_U.append(U)

#         return np.array(path_U), np.array(path_r)



import numpy as np

class UtilizationSimulator:
    """
    Simulates utilization dynamics using a stochastic differential equation (SDE).
    
    Attributes:
        U0 (float): Initial utilization ratio.
        T (float): Total simulation time.
        dt (float): Time step.
        rate_model (function): Interest rate model function.
        b0 (float): Baseline drift parameter (supplier sensitivity).
        b1 (float): Borrower sensitivity to interest rate.
        sigma (float): Volatility parameter.
    """
    
    def __init__(self, U0, T, dt, rate_model, b0, b1, sigma):
        self.U0 = U0
        self.T = T
        self.dt = dt
        self.rate_model = rate_model
        self.b0 = b0
        self.b1 = b1
        self.sigma = sigma

    def drift(self, r):
        """
        Compute the drift term for utilization dynamics.
        
        Parameters:
            r (float): Current interest rate.
        
        Returns:
            float: Drift value for utilization.
        """
        return -self.b0 - self.b1 * r

    def simulate(self):
        """
        Simulate utilization dynamics using the specified parameters.
        
        Returns:
            tuple: Two np.arrays containing the utilization path and interest rate path over time.
        """
        n_steps = int(self.T / self.dt)
        U = self.U0
        r = self.rate_model(U)  # Calculate initial rate

        path_U = [U]
        path_r = [r]

        for _ in range(n_steps):
            dW = np.random.normal(0, np.sqrt(self.dt))  # Brownian motion increment
            r = self.rate_model(U)  # Compute interest rate using the model
            path_r.append(r)  # Record the interest rate
            dU_drift = self.drift(r) * self.dt
            dU_volatility = self.sigma * dW
            dU = dU_drift + dU_volatility
            U = max(0, min(1, U + dU))  # Constrain U to [0, 1]
            path_U.append(U)

        return np.array(path_U), np.array(path_r)
