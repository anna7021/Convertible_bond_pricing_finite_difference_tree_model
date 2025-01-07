import math
import numpy as np
import matplotlib.pyplot as plt

class ConvertibleBondBinomial:
    """
    Class for pricing convertible bonds using binomial tree models.
    Supports:
    - Credit Spread Model
    - Reduced Form Credit Risk Model
    """
    def __init__(self, K_bond, ratio, r, div_yield, c, freq, sigma, T, N, lambda_=None, RR=None):
        """
        Initialize the parameters for the convertible bond pricing model.

        Parameters:
        K_bond (float): Face value of the bond.
        ratio (float): Conversion ratio of the bond.
        r (float): Risk-free interest rate.
        div_yield (float): Dividend yield of the stock.
        c (float): Coupon rate of the bond.
        freq (int): Coupon payment frequency (per year).
        sigma (float): Volatility of the stock.
        T (float): Time to maturity (in years).
        N (int): Number of steps in the binomial tree.
        lambda_ (float, optional): Hazard rate for reduced form model.
        RR (float, optional): Recovery rate for reduced form model.
        """
        self.K_bond = K_bond
        self.ratio = ratio
        self.r = r
        self.div_yield = div_yield
        self.c = c
        self.freq = freq
        self.sigma = sigma
        self.T = T
        self.N = N
        self.lambda_ = lambda_
        self.RR = RR

    def CB_credit_spread_model(self, S0, dd, calls=None, puts=None):
        """
        Price the convertible bond using the credit spread model.
        This model considers the impact of credit risk on the bond's value by
        incorporating a credit spread (dd) into the discount rate.
        It uses a risk-neutral probability framework and accounts for
        optionality through call and put schedules.

        Parameters:
        S0 (float): Current stock price.
        dd (float): Credit spread adjustment.
        calls (dict, optional): Call schedule (year: call price).
        puts (dict, optional): Put schedule (year: put price).

        Returns:
        float: Convertible bond price.
        """
        dt = self.T / self.N
        u = np.exp(self.sigma * np.sqrt(dt))
        d = 1 / u
        q = (np.exp((self.r - self.div_yield) * dt) - d) / (u - d)
        S = np.asarray([S0 * u ** j * d ** (self.N - j) for j in range(self.N + 1)])

        accrued_coupon_final = self.c * (self.T % (1.0 / self.freq)) * self.K_bond
        final_parity = self.ratio * S
        redemption_value = self.K_bond + accrued_coupon_final
        V = np.maximum(final_parity, redemption_value)
        p = np.zeros(self.N + 1)
        p[final_parity > redemption_value] = 1.0
        y = p * (self.r - self.div_yield) + (1 - p) * ((self.r - self.div_yield) + dd)

        for i in range(self.N - 1, -1, -1):
            j_vec = np.arange(i + 1)
            S = S0 * (u ** j_vec) * (d ** (i - j_vec))
            p = q * p[1:] + (1 - q) * p[:-1]
            accrued_coupon = self.c * (i * dt % (1.0 / self.freq)) * self.K_bond

            H = (
                q * np.exp(-y[1:] * dt) * (V[1:] + accrued_coupon) +
                (1 - q) * np.exp(-y[:-1] * dt) * (V[:-1] + accrued_coupon)
            )

            call_value = float("inf")
            if calls:
                for key, value in calls.items():
                    if key == math.floor(self.T - i * dt):
                        call_value = value + accrued_coupon
                        break
            put_value = 0.0
            if puts:
                for key, value in puts.items():
                    if key == math.floor(self.T - i * dt):
                        put_value = value + accrued_coupon
                        break

            temp = np.minimum(H, call_value)
            temp = np.maximum(temp, put_value)
            V = np.maximum(temp, S * self.ratio)
            p = np.where(V == S * self.ratio, 1.0, p)
            p = np.where(((V == put_value) | (V == call_value)), 0.0, p)
            y = p * (self.r - self.div_yield) + (1 - p) * ((self.r - self.div_yield) + dd)

        return V[0]

    def CB_reduced_form_credit_risk_model(self, S0, calls=None, puts=None):
        """
        Price the convertible bond using the reduced form credit risk model.
        This model uses a reduced-form approach to handle credit risk,
        incorporating a hazard rate (lambda_) to model the probability of default
        and a recovery rate (RR) to estimate the bond's value in the event of default.

        Parameters:
        S0 (float): Current stock price.
        calls (dict, optional): Call schedule (year: call price).
        puts (dict, optional): Put schedule (year: put price).

        Returns:
        float: Convertible bond price.
        """
        dt = self.T / self.N
        u = np.exp(self.sigma * np.sqrt(dt))
        d = 1 / u
        q = (np.exp((self.r + self.lambda_ - self.div_yield) * dt) - d) / (u - d)
        surv_prob = np.exp(-self.lambda_ * dt)
        def_prob = 1 - surv_prob

        stock_tree = [S0 * (u ** j) * (d ** (self.N - j)) for j in range(self.N + 1)]
        stock_tree = np.array(stock_tree, dtype=float)

        surv_to_T = np.exp(-self.lambda_ * self.T)
        default_by_T = 1 - surv_to_T
        accrued_coupon_final = self.c * (self.T % (1.0 / self.freq)) * self.K_bond
        bond_redemption = self.K_bond + accrued_coupon_final
        no_conversion_value = surv_to_T * bond_redemption + default_by_T * (self.RR * self.K_bond)
        conversion_value = surv_to_T * (self.ratio * stock_tree)
        V = np.maximum(no_conversion_value, conversion_value)

        for i in range(self.N - 1, -1, -1):
            accrued_coupon = self.c * (i * dt % (1.0 / self.freq)) * self.K_bond
            pv_continuation = np.exp(-(self.r + self.lambda_ - self.div_yield) * dt) * (q * V[1:] + (1 - q) * V[:-1])
            pv_coupon = np.exp(-self.lambda_ * dt) * accrued_coupon
            Vr = self.RR * self.K_bond * np.exp(-(self.r - self.div_yield) * (i * dt))
            pv_recovery = np.exp(-(self.r - self.div_yield) * dt) * (1 - np.exp(-self.lambda_ * dt)) * Vr
            H = pv_continuation + pv_coupon + pv_recovery

            j_vec = np.arange(i + 1)
            stock_step_i = S0 * (u ** j_vec) * (d ** (i - j_vec))

            call_value = float("inf")
            if calls:
                for key, value in calls.items():
                    if key == math.floor(self.T - i * dt):
                        call_value = value + accrued_coupon
                        break
            put_value = 0.0
            if puts:
                for key, value in puts.items():
                    if key == math.floor(self.T - i * dt):
                        put_value = value + accrued_coupon
                        break

            temp = np.minimum(H, call_value)
            temp = np.maximum(temp, put_value)
            V = np.maximum(temp, self.ratio * stock_step_i)

        return V[0]

# Implementation Example
if __name__ == "__main__":
    bond = ConvertibleBondBinomial(K_bond=100, ratio=1, r=0.02, div_yield=0.01, c=0.03, freq=2,
                                    sigma=0.2, T=5, N=50, lambda_=0.05, RR=0.4)
    stock_prices = np.linspace(0, 200, 100)

    bond_prices_cs = [bond.CB_credit_spread_model(S, dd=0.03) for S in stock_prices]
    bond_prices_rf = [bond.CB_reduced_form_credit_risk_model(S) for S in stock_prices]

    plt.figure(figsize=(10, 6))
    plt.plot(stock_prices, bond_prices_cs, label="Credit Spread Model")
    plt.plot(stock_prices, bond_prices_rf, label="Reduced Form Model", linestyle="--")
    plt.axhline(y=100, color="gray", linestyle="--", label="Face Value (Bond Floor)")
    plt.plot(stock_prices, stock_prices, linestyle=":", label="Equity Conversion Value (Parity)")
    plt.title("Convertible Bond Price vs. Equity Price")
    plt.xlabel("Equity Price")
    plt.ylabel("Convertible Bond Price")
    plt.legend()
    plt.grid()
    plt.show()
