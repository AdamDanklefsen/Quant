from datetime import datetime
from Options import trading_days_between


def BS_Implied_Volatility(df, now = datetime.now()):
    import numpy as np
    import pandas as pd
    from scipy.stats import norm
    assert isinstance(df, pd.DataFrame)
    assert 'Strike' in df.columns and 'Price' in df.columns and 'Type' in df.columns and 'Expiry' in df.columns and 'Spot' in df.columns and 'Rate' in df.columns


    def C(S, K, T, r, sigma):
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    def Vega(S, K, T, r, sigma):
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return S * norm.pdf(d1) * np.sqrt(T)
    
    VOL = pd.Series(0.4, index=df.index)
    T = df['Expiry'].apply(lambda x: trading_days_between(now, x)/252).values

    for i in range(len(df)):
        if df.iloc[i]['Price'] <= np.maximum(0, df.iloc[i]['Spot'] - df.iloc[i]['Strike'] * np.exp(-df.iloc[i]['Rate'] * T[i])):
            VOL.iloc[i] = 1e-4
            continue
        diff = 10
        if i != 0:
            VOL.iloc[i] = VOL.iloc[i-1]
        # print(f"Calculating for index {i}, Strike: {df.iloc[i]['Strike']}, Price: {df.iloc[i]['Price']}, Initial VOL: {VOL.iloc[i]}, T: {T[i]}")
        iter = 0
        while np.abs(diff) > 1e-3 and iter < 50:
            OptionPrice = C(df.iloc[i]['Spot'], df.iloc[i]['Strike'], T[i], df.iloc[i]['Rate'], VOL.iloc[i])
            OptionVega = Vega(df.iloc[i]['Spot'], df.iloc[i]['Strike'], T[i], df.iloc[i]['Rate'], VOL.iloc[i])
            # print(f"OptionPrice: {OptionPrice}, OptionVega: {OptionVega}")
            diff = (df.iloc[i]['Price'] - OptionPrice)/np.clip(OptionVega, 1e-4, None)
            VOL.iloc[i] = VOL.iloc[i] + np.clip(diff, -.1, .1)
            # print(f"Calculating for index {i}, Price: {df.iloc[i]['Price']}, Calculated Price: {OptionPrice}, Vega: {OptionVega}, Diff: {diff}, VOL: {VOL.iloc[i]}")
            iter += 1
        if iter == 50:
            print(f"Warning: Max iterations reached for index {i}, Strike: {df.iloc[i]['Strike']}, Price: {df.iloc[i]['Price']}, Final VOL: {VOL.iloc[i]}, Final Diff: {diff}")
    VOL = np.clip(VOL, 1e-4, 5)
    return VOL
            
def fit_IV_slice(df, now=datetime.now()):
    import numpy as np
    import pandas as pd
    from scipy.optimize import minimize
    import matplotlib.pyplot as plt

    assert isinstance(df, pd.DataFrame)
    required_cols = {'Strike', 'Price', 'Type', 'Expiry', 'Spot', 'Rate', 'Implied Volatility'}
    assert required_cols.issubset(df.columns)

    # Filter for valid implied volatilities
    IV_raw = df['Implied Volatility'].values
    mask = IV_raw > 1e-4
    if not np.any(mask):
        raise ValueError("No valid implied volatilities above threshold.")

    K = df['Strike'].values[mask]
    IV = IV_raw[mask]
    S = df['Spot'].values[0]
    T = trading_days_between(now, df['Expiry'].values[0]) / 252
    r = df['Rate'].values[0]
    F = S * np.exp(r * T)  # forward price
    k = np.log(K / F)      # log-moneyness

    # SVI-JW total implied variance function
    def svi_jw(k, v, psi, chi, p, zeta):
        return v + 0.5 * psi * (chi * (k - p) + np.sqrt((k - p)**2 + zeta**2))

    # Initial guess and bounds
    p0 = [np.min(IV)**2 * T, 0.5, 0.0, k[np.argmin(IV)], 0.1]  # [v, psi, chi, p, zeta]
    bounds = [
        (1e-3, 2),     # v > 0
        (1e-3, 10),    # psi > 0
        (-1, 1),       # chi in [-1, 1]
        (-2, 2),       # p around 0
        (1e-3, 2)      # zeta > 0
    ]

    # Total variance and loss function
    total_var = IV**2 * T
    minIV_idx = np.argmin(IV)
    k0 = k[minIV_idx]
    print(f"Minimum IV at k={k0}, K = {K[minIV_idx]}, IV = {IV[minIV_idx]}")

    def w(x):
        sigma = 0.08  # controls weight spread
        return np.exp(-0.5 * (x - k0)**2 / sigma**2) / np.sqrt(2 * np.pi * sigma**2)

    L = lambda p: np.sum((total_var - svi_jw(k, *p))**2)
    print(f"Using {np.sum(mask)} valid points for fitting.")

    # Optimize
    from scipy.optimize import least_squares

    def residuals(p):
        return np.sqrt(w(k)) * (total_var - svi_jw(k, *p))

    result = least_squares(residuals, p0, bounds=tuple(np.array(bounds).T), loss='huber')

    # result = minimize(L, p0, bounds=bounds, tol=1e-9, options={'maxiter': 1000})

    print(f"Loss: {L(result.x)}")
    print(f"Initial guess: v={p0[0]:.6f}, psi={p0[1]:.6f}, chi={p0[2]:.6f}, p={p0[3]:.6f}, zeta={p0[4]:.6f}")
    print(f"Fitted SVI-JW parameters: v={result.x[0]:.6f}, psi={result.x[1]:.6f}, chi={result.x[2]:.6f}, p={result.x[3]:.6f}, zeta={result.x[4]:.6f}")
    print(result)

    # Interpolator function for strike
    def svi_k(K_input):
        k_input = np.log(K_input / F)
        return np.sqrt(svi_jw(k_input, *result.x) / T)

    # Return results
    fitted_IV = np.sqrt(svi_jw(np.log(df['Strike'] / F), *result.x) / T)
    return result.x, fitted_IV, svi_k, w(np.log(df['Strike'].values / F))

def Distribution_from_IV(df, svicurve, svi_k, now = datetime.now()):
    import numpy as np
    import pandas as pd
    from scipy.stats import norm
    assert isinstance(df, pd.DataFrame)
    assert 'Strike' in df.columns and 'Price' in df.columns and 'Type' in df.columns and 'Expiry' in df.columns and 'Spot' in df.columns and 'Rate' in df.columns
    K = df['Strike'].values
    S = df['Spot'].values[0]
    T = trading_days_between(now, df['Expiry'].values[0])/252
    r = df['Rate'].values[0]
    X = np.log(K / S) / np.sqrt(T)
    IV = svicurve
    d1 = (np.log(S / K) + (r + 0.5 * IV**2) * T) / (IV * np.sqrt(T))
    d2 = d1 - IV * np.sqrt(T)
    P = np.exp(-r * T) * norm.pdf(d2) / (K**2 * IV * np.sqrt(T))
    def P_k(K):
        d1 = (np.log(S / K) + (r + 0.5 * svi_k(K)**2) * T) / (svi_k(K) * np.sqrt(T))
        d2 = d1 - svi_k(K) * np.sqrt(T)
        return np.exp(-r * T) * norm.pdf(d2) / (K**2 * svi_k(K) * np.sqrt(T))
    return P, P_k
