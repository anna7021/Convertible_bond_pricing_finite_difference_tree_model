from math import floor
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
np.set_printoptions(suppress=True)

class FiniteDifferenceConvertibleBondPricing:
    def __init__(self, S, T, c, c_freq, sigma, r, p, eta, R, kappa, F, call_schedule, put_schedule, M, N, n):
        self.S = S    # Initial stock price
        self.T = T    # Time to maturity
        self.c = c    # Annual coupon rate
        self.c_freq = c_freq    # Coupon payment frequency per year
        self.sigma = sigma  # Volatility
        self.r = r    # Risk-free rate
        self.p = p   # Probability of default, hazard rate
        self.eta = eta  # Proportion of value lost during the recovery process
        self.R = R    # Recovery rate
        self.kappa = kappa  # Conversion ratio
        self.F = F    # Face value/ redemption value of the straight bond
        self.call_schedule = call_schedule    # Call schedule
        self.put_schedule = put_schedule    # Put schedule
        self.M = M    # Number of price steps
        self.N = N    # Number of time steps
        self.n = n    # Number of time steps between each coupon payment date

    def _generate_grid_params(self, price_multiplier=2):
        '''
        Grid creation:
            y-axis: Stock Price Grid. Divides the stock price range into M discrete steps.
            x-axis: Time Grid. Divides the time to maturity into N discrete steps, aligned with coupon payment dates.
        '''
        ds = price_multiplier * self.S / self.M    # S_max = multiplier * S
        self.S_idx = int(self.S / ds)
        dt = 1 / (self.n * self.c_freq)
        self.N = int(self.T / dt)    # Overwrite  N to fit coupon payment dates all fall on the grids
        return ds, dt

    def _generate_tridiagonal_matrix(self, M, aj, bj, cj):
        diag_main = np.array([bj(j) for j in range(1, M)])
        diag_lower = np.array([aj(j) for j in range(2, M)])
        diag_upper = np.array([cj(j) for j in range(1, M - 1)])
        coef_matrix = np.diag(diag_main) + np.diag(diag_lower, -1) + np.diag(diag_upper, 1)
        return coef_matrix

    def implicit_fd(self):
        ds, dt = self._generate_grid_params(price_multiplier=2)
        print(f"Generated grid: M = {self.M} price points, N = {self.N} time points")
        B_grid = np.zeros((self.M + 1, self.N + 1))
        C_grid = np.zeros((self.M + 1, self.N + 1))
        S_array = np.linspace(self.M * ds, 0, self.M + 1)
        T_array = np.linspace(0, self.N * dt, self.N + 1)

        # coupon schedule calculation
        T_c = np.linspace(0, self.N * dt, 1 + self.c_freq * self.T)
        c_payment = [self.c * self.F / self.c_freq] * len(T_c)
        c_payment[0] = 0   # coupon payment at t=0

        # Boundary conditions:
        #   At maturity (t=T): Straight bond value: B=F (face value).
        #   Conversion option value: C=max(κS−F,0).
        for i in range(self.N+1):
            t = T_array[i]
            remaining_coupons = T_c[T_c >= t]
            pv_coupons = sum(
                (self.c / self.c_freq) * self.F * np.exp(-(self.r + self.p) * (tc - t))
                for tc in remaining_coupons
            )
            pv_redemption = self.F * np.exp(-(self.r + self.p) * (self.T - t))
            # B_grid[:, i] = pv_redemption
            B_grid[:, i] = pv_coupons + pv_redemption
            C_grid[:, i] = np.maximum(self.kappa * S_array - B_grid[:, i], 0)
        # B_grid[:, 0] = self.F * np.exp(-(self.r + self.p) * self.T)
        B_grid[:, 0] = self.F * np.exp(-(self.r + self.p) * self.T) + sum(c_payment * np.exp(-(self.r + self.p) * T_c))
        C_grid[:, 0] = np.maximum(self.kappa * S_array - B_grid[:, 0], 0)
        B_grid[:, -1] = self.F
        C_grid[:, -1] = np.maximum(self.kappa * S_array - self.F, 0)
        print(B_grid)
        print(C_grid)

        # Coefficient functions for B
        ajB = lambda j: 0.5 * j * ((self.r + self.p * self.eta) - self.sigma**2 * j) * dt
        bjB = lambda j: 1 + (self.r + self.p*(1-self.R) + self.sigma**2 * j**2) * dt
        cjB = lambda j: 0.5 * j * (-(self.r + self.p * self.eta) - self.sigma**2 * j) * dt
        coef_matrixB = self._generate_tridiagonal_matrix(self.M, ajB, bjB, cjB)
        M_B_inverse = np.linalg.inv(coef_matrixB)
        # Coefficient functions for C
        ajC = lambda j: 0.5 * j * ((self.r + self.p * self.eta) - self.sigma**2 * j) * dt
        bjC = lambda j: 1 + (self.r + self.p + self.sigma**2 * j**2) * dt
        cjC = lambda j: 0.5 * j * (-(self.r + self.p * self.eta) - self.sigma**2 * j) * dt
        coef_matrixC = self._generate_tridiagonal_matrix(self.M, ajC, bjC, cjC)
        M_C_inverse = np.linalg.inv(coef_matrixC)


        for i in range(self.N - 1, -1, -1):
            # Solve the PDE for B_i first
            Z_B = np.zeros(self.M - 1)
            Z_B[0] = ajB(1) * B_grid[0, i]
            Z_B[-1] = cjB(self.M - 1) * B_grid[-1, i]
            B_grid[1:self.M, i] = M_B_inverse @ (B_grid[1:self.M, i + 1] - Z_B)

            # Solve the PDE for C_i
            const = self.p * np.maximum(self.kappa * (1 - self.eta) * S_array[1:self.M] - self.R * B_grid[1:self.M, i], 0)    # the intrinsic value of the holder's last option to convert into the residual value of the share
            Z_C = np.ones(self.M - 1) * const
            Z_C[0] += ajC(1) * C_grid[0, i]
            Z_C[-1] += cjC(self.M - 1) * C_grid[-1, i]
            C_grid[1:self.M, i] = M_C_inverse @ (C_grid[1:self.M, i + 1] - Z_C)

            # # Conditions check
            # # Check for put schedule
            # if floor(self.T - i * dt) in self.put_schedule.keys():
            #     pyear = floor(self.T - i * dt)
            #     Bp = self.put_schedule[pyear]
            #     # If Bp > κS, and continuation value (B + C) < Bp, B:= Bp-C
            #     indices = np.where((Bp > self.kappa * S_array[1:self.M])
            #                         & ((B_grid[1:self.M, i] + C_grid[1:self.M, i]) < Bp))
            #     B_grid[indices, i] = Bp - C_grid[indices, i]
            #     # If Bp <= κS, and continuation value (B + C) < κS, C:= kS-B
            #     indices = np.where((Bp <= self.kappa * S_array[1:self.M]) &
            #                        ((B_grid[1:self.M, i] + C_grid[1:self.M, i]) < (self.kappa * S_array[1:self.M])))
            #     C_grid[indices, i] = self.kappa * S_array[indices] - B_grid[indices, i]
            #
            # # Check for call schedule
            # if floor(self.T - i * dt) in self.call_schedule.keys():
            #     cyear = floor(self.T - i * dt)
            #     Bc = self.call_schedule[cyear]
            #     # If Bc < kS, C:= kS-B
            #     indices = np.where(Bc < self.kappa * S_array[1:self.M])
            #     C_grid[indices, i] = self.kappa * S_array[indices] - B_grid[indices, i]
            #     # If Bc >= kS, and (B+C) > Bc, C:= Bc-B
            #     indices = np.where((Bc >= self.kappa * S_array[1:self.M]) &
            #                        ((B_grid[1:self.M, i] + C_grid[1:self.M, i]) > Bc))
            #     C_grid[indices, i] = Bc - B_grid[indices, i]

            # Conditions check
            # Check for put schedule
            if floor(self.T - i * dt) in self.put_schedule.keys():
                pyear = floor(self.T - i * dt)
                Bp = self.put_schedule[pyear]
                # If Bp > κS, and continuation value (B + C) < Bp, B:= Bp-C
                indices = np.where((Bp > self.kappa * S_array)
                                   & ((B_grid[:, i] + C_grid[:, i]) < Bp))
                B_grid[indices, i] = Bp - C_grid[indices, i]
                # If Bp <= κS, and continuation value (B + C) < κS, C:= kS-B
                indices = np.where((Bp <= self.kappa * S_array) &
                                   ((B_grid[:, i] + C_grid[:, i]) < (self.kappa * S_array)))
                C_grid[indices, i] = self.kappa * S_array[indices] - B_grid[indices, i]

            # Check for call schedule
            if floor(self.T - i * dt) in self.call_schedule.keys():
                cyear = floor(self.T - i * dt)
                Bc = self.call_schedule[cyear]
                # If Bc < kS, C:= kS-B
                indices = np.where(Bc < self.kappa * S_array)
                C_grid[indices, i] = self.kappa * S_array[indices] - B_grid[indices, i]
                # If Bc >= kS, and (B+C) > Bc, C:= Bc-B
                indices = np.where((Bc >= self.kappa * S_array) &
                                   ((B_grid[:, i] + C_grid[:, i]) > Bc))
                C_grid[indices, i] = Bc - B_grid[indices, i]
            #
            # Coupon jumps incorporation
            if (i * dt) % (1.0 / self.c_freq) == 0:
                B_grid[1:self.M, i] += self.c / self.c_freq * self.F

            # print("C:", C_grid, "B: ", B_grid)

        return B_grid[self.S_idx, 0], B_grid[:, 0], C_grid[self.S_idx, 0], C_grid[:, 0], S_array



# Example usage
S = 50
T = 5
c = 0.08
c_freq = 2
sigma = 0.2
r = 0.05
p = 0.02    # hazard rate
eta = 0.03   # haircut
R = 0.5    # recovery rate
kappa = 2  # conversion ratio
F = 50
call_schedule = {3:55, 4:55, 5:55}
put_schedule = {3:55, 4:55, 5:55}
M = 50
N = 100 # will be overwritten in the function
n = 20


pricing_model = FiniteDifferenceConvertibleBondPricing(S, T, c, c_freq, sigma, r, p, eta, R, kappa, F, call_schedule, put_schedule, M, N, n)
B_value0, B_array0, C_value0, C_array0, S_array = pricing_model.implicit_fd()
print(f"Implicit Price: {B_value0 + C_value0}")

V_array0 = B_array0 + C_array0
plt.figure(figsize=(10, 6))
plt.plot(S_array, V_array0, label="Implicit Method")
plt.title("Convertible Bond Value vs Stock Price")
plt.xlabel("Stock Price (S)")
plt.ylabel("Convertible Bond Value")
plt.legend()
plt.grid(True)
plt.show()


