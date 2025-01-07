'''
This project is to use Monte Carlo simulation to price earn out.

The product information: From and after the receipt by the investors of aggregate proceeds in respect to the Class A units that exceed the product of
(i) 2.50 multiplied by (ii) the investors’ investment amount in the Class A units, at such time as there are proceeds actually
paid, the buyer shall promptly pay to the sellers an amount equal to 25.0% of the available proceed until such time as the
sellers have received an aggregate amount equal to $20,000,000. --cited from https://cdn.hl.com/pdf/2023/emerging-trends--earnout-structured-as-equity.pdf
'''
# parameters assumptions:
# Initial Investment: $300M
# Earn out period: after 5 years
# 12% annual discount rate

import numpy as np
import matplotlib.pyplot as plt

def simulate_exit_proceeds(n_sims, mu, sigma, random_seed=123):
    """
    Simulate future exit proceeds (X) using a lognormal distribution.

    Parameters:
    n_sims : int. Number of Monte Carlo simulations.
    mu : float. The log of the median of X (i.e., E[ln(X)]).
    sigma : float. Standard deviation of ln(X).
    random_seed : int.

    Returns:
    np.ndarray: Simulated draws from a lognormal distribution.
    """
    np.random.seed(random_seed)
    proceeds = np.random.lognormal(mean=mu, sigma=sigma, size=n_sims)
    return proceeds

def earnout(exit_value, threshold, earnout_rate=0.25, earnout_cap=20e6):
    """
    Compute the earnout payment for a single scenario.

    Earnout = earnout_rate * (exit_value - threshold) if exit_value > threshold,
    capped at earnout_cap, else 0.
    """
    if exit_value <= threshold:
        return 0.0
    else:
        earnout = min(earnout_rate * (exit_value - threshold), earnout_cap)
        return earnout

def discount_cash_flow(cash_flow, annual_discount_rate, years):
    return cash_flow / ((1.0 + annual_discount_rate) ** years)

def monte_carlo_valuation(n_sims, investment_amount, multiple, earnout_rate, earnout_cap, mu, sigma, annual_discount_rate, years, random_seed):
    threshold = multiple * investment_amount
    exits = simulate_exit_proceeds(n_sims=n_sims, mu=mu, sigma=sigma, random_seed=random_seed)
    earnouts = np.array([earnout(x, threshold, earnout_rate, earnout_cap) for x in exits])
    discounted_earnouts = discount_cash_flow(earnouts, annual_discount_rate, years)
    pv_earnout = np.mean(discounted_earnouts)
    return discounted_earnouts, pv_earnout

def main():
    n_sims = 100_000
    investment_amount = 300e6   # $300 million
    multiple = 2.50
    earnout_rate = 0.25
    earnout_cap = 20e6
    mu = 21.03
    sigma = 1.1     # lognormal volatility for a 5-year horizon
    annual_discount_rate = 0.12
    years = 5
    random_seed = 42
    # Monte Carlo Valuation
    discounted_earnouts, pv_earnout = monte_carlo_valuation(
        n_sims=n_sims,
        investment_amount=investment_amount,
        multiple=multiple,
        earnout_rate=earnout_rate,
        earnout_cap=earnout_cap,
        mu=mu,
        sigma=sigma,
        annual_discount_rate=annual_discount_rate,
        years=years,
        random_seed=random_seed
    )
    print(f"Estimated Present Value of the Earnout: ${pv_earnout:,.2f}")


if __name__ == "__main__":
    main()



