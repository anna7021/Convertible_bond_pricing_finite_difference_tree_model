import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class HazardRateBootstrap:
    """
    A class for bootstrapping piecewise-constant hazard rates from CDS spreads and
    calculating survival probabilities.
    """

    def __init__(self, tenors, spreads, discount_factors, recovery_rate):
        """
        Initialize the HazardRateBootstrap class.

        Parameters:
            tenors (array-like): The tenors defining the time intervals.
            spreads (array-like): The CDS spreads for each tenor (annualized, in decimal).
            discount_factors (array-like): Discount factors D(0, T_k) for each tenor.
            recovery_rate (float): The recovery rate (e.g., 0.40 for 40% recovery).
        """
        self.tenors = np.array(tenors, dtype=float)
        self.spreads = np.array(spreads, dtype=float)
        self.discount_factors = np.array(discount_factors, dtype=float)
        self.recovery_rate = recovery_rate

    def bootstrap_hazard_rates(self):
        """
        Bootstraps piecewise-constant hazard rates for each interval [T_{k-1}, T_k].

        Returns:
            tuple: A tuple containing:
                - hazard_rates (np.ndarray): The hazard rates [lambda_1, lambda_2, ..., lambda_n].
                - survival_probs (np.ndarray): The survival probabilities [P(T_0), P(T_1), ..., P(T_n)].
        """
        T = self.tenors
        S = self.spreads
        D = self.discount_factors
        R = self.recovery_rate

        n = len(T)
        lambdas = np.zeros(n)
        P_surv = np.zeros(n)
        G = np.zeros(n)
        H = np.zeros(n)

        P_surv[0] = 1.0
        H[1] = P_surv[0] * D[1]

        for k in range(1, n):
            dt = T[k] - T[k - 1]
            H[k] = P_surv[k - 1] * D[k]
            if k == 1:
                lambdas[k] = - (1.0 / dt) * np.log((1 - R - S[k] * dt) / (1 - R))
            else:
                sum_G = np.sum(G[1:k])
                sum_H = np.sum(H[1:k + 1])

                numerator = (P_surv[k - 1] * D[k] * (1 - R) + sum_G - S[k] * dt * sum_H)
                denominator = P_surv[k - 1] * D[k] * (1 - R)

                ratio = max(numerator / denominator, 1e-14)
                lambdas[k] = - (1.0 / dt) * np.log(ratio)

            P_surv[k] = P_surv[k - 1] * np.exp(-lambdas[k] * dt)
            G[k] = P_surv[k - 1] * (1.0 - np.exp(-lambdas[k] * dt)) * D[k] * (1 - R)

        self.hazard_rates = lambdas
        self.survival_probs = P_surv

        return lambdas, P_surv

    def survival_probability_at_t(self, T):
        """
        Calculate the survival probability at an arbitrary time T.

        Parameters:
            T (float): The time at which to calculate the survival probability.

        Returns:
            float: The survival probability at time T.
        """
        T = float(T)
        tenors = self.tenors
        hazard_rates = self.hazard_rates

        if T < tenors[0]:
            return 1.0

        j = np.searchsorted(tenors, T)

        cumulative_hazard = np.sum(hazard_rates[1:j] * np.diff(tenors[:j]))
        cumulative_hazard += hazard_rates[j] * (T - tenors[j - 1])

        return np.exp(-cumulative_hazard)

    def plot_hazard_rates(self):
        """
        Plot the piecewise-constant hazard rates.
        """
        time_steps = np.repeat(self.tenors, 2)[1:-1]
        hazard_rates_piecewise = np.repeat(self.hazard_rates[1:], 2)

        plt.figure(figsize=(10, 6))
        plt.step(time_steps, hazard_rates_piecewise, where='post', label='Hazard Rates')
        plt.title("Piecewise-Constant Hazard Rates vs. Tenors")
        plt.xlabel("Time (Years)")
        plt.ylabel("Hazard Rates")
        plt.grid()
        plt.legend()
        plt.show()

    def plot_survival_probabilities(self, T=None, survival_prob=None):
        """
        Plot the survival probabilities.

        Parameters:
            T (float, optional): The specific time for which survival probability is calculated.
            survival_prob (float, optional): The survival probability at time T.
        """
        plt.figure(figsize=(10, 6))
        plt.step(self.tenors, self.survival_probs, where='post',
                 label='Survival Probabilities', color='green', linewidth=0.8)

        if T is not None and survival_prob is not None:
            plt.scatter([T], [survival_prob], color='red', label=f'Survival Probability at T={T:.2f}', zorder=5)

        plt.title("Survival Probabilities vs. Tenors")
        plt.xlabel("Time (Years)")
        plt.ylabel("Survival Probabilities")
        plt.grid()
        plt.legend()
        plt.show()

# Example Usage
data = pd.read_excel("CDS_data.xlsx")
if __name__ == "__main__":
    tenors = data.iloc[:, 0]
    spreads = data.iloc[:, 1]
    discount_factors = data.iloc[:, 2]
    recovery = 0.40

    bootstrap = HazardRateBootstrap(tenors, spreads, discount_factors, recovery)
    hazard_rates, survival_probs = bootstrap.bootstrap_hazard_rates()

    T = 1.13
    survival_prob = bootstrap.survival_probability_at_t(T)
    print(f"Survival probability at T = {T}: {survival_prob:.6f}")

    bootstrap.plot_hazard_rates()
    bootstrap.plot_survival_probabilities(T, survival_prob)
