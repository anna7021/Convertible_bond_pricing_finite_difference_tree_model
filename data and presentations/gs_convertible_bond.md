## Assumption:
1. The distribution of future stock prices is lognormal with know volatility.
2. All future interest rates (rf), the stock loan rate and the issuer's credit spread, are known with certainty.
3. All the information we need about default risk is contained in the credit spread for the issuer's straight bonds. 
4. BS works as well, you can create continous riskless hedging. The expected value of the bond is the expected value over all future stock price scenarios.

## The Binomial Tree-- Cox-Ross-Rubinstein Model
Holding value H = PV(coupon paid over the next period) + PV(E($V_u$, $V_d$))
### Some possible provisions:
1. **No active call or put provisions**: The investor can either hold the bond for one more period or convert it to stock. **V = max(H, parity)**
2. **The convertible can be put at price P**: The investor can hold, convert, or put the bond for cash equal in value to P plus accrued interest. **V = max(H, parity, put value)**
3. **The convertible is both callable at price C and putable at price P**: The issuer will call the bond when the call value (= C + accured interest) is less than the holding value H. When the bond is called, The investor can still choose whether to put the bond for the put value, convert it to stock, or accept the issuer's call. **V = max(parity, put value, min(H, call value))**

## The Credit-Adjusted Distount Rate
What is the discount factor to calculate PV?-- Credit-adjusted discount rate, y\
- $p$: probability at a given node that the convertible will convert to stock in the future
- $1-p$: probability that it will remain a fixed-income bond
- $d = r_f + issuer's\ credit\ spread\ to\ r_f$, is the risky rate 
- $y = p\times r + (1-p) \times d$, the weighted mixture of the riskless and risky rates 

## Model Summary
1. Build a stock tree. The average stock price matches the stock's forward price.
2. At maturity, 
   - value of the convertible bond = max(fixed-income redemption value, conversion value)
   - p=1 at nodes where converts, p=0 elsewhere
3. Move backward one level at a time. At each node, 
   - conversion probability $p = average\ p\ at\ two\ connected\ nodes$ 
   - $y = p\times r + (1-p) \times d$
   - convertible value = max(H, call, put, conversion value)
   - set p=0 at nodes where the bond being put, p=1 where the node results from conversion.

Test senarios: 
1. Deep ITM: parity >> face value, y -> rf
2. Volatility: 1) makes the expected payoff from conversion greater; 2) increases the probability of default, lowers teh convertible's value

Model Drawbacks:
1. Lognormal Stock Price Assumption. Consider the convertible as a compound derivative claim on the assets, assume the distribution of future asset values is lognormal, the distribution of future stock prices can not be lognormal
2. Constant Credit Spread. Think o a convertible as a credit derivative, the value of the derivative depends upon the volatility of the credit spread 

## Credit Spread
1. Constant credit spread model
2. A reduced form model using an exogenous Poisson process to model the default intensity.

















