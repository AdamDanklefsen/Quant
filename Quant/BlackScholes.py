from datetime import datetime
from Options import trading_days_between
def BS_Option_Price(df, svicurve, now=datetime.now()):
    import numpy as np
    import pandas as pd
    from scipy.stats import norm
    assert isinstance(df, pd.DataFrame)
    assert 'Strike' in df.columns and 'Price' in df.columns and 'Type' in df.columns and 'Expiry' in df.columns and 'Spot' in df.columns and 'Rate' in df.columns

    def C(S, K, T, r, sigma):
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    def P(S, K, T, r, sigma):
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    Prices = []
    for i in range(len(df)):
        T = trading_days_between(now, df.iloc[i]['Expiry'])/252
        if df.iloc[i]['Type'] == 'C':
            price = C(df.iloc[i]['Spot'], df.iloc[i]['Strike'], T, df.iloc[i]['Rate'], svicurve[i])
        else:
            price = P(df.iloc[i]['Spot'], df.iloc[i]['Strike'], T, df.iloc[i]['Rate'], svicurve[i])
        Prices.append(price)
    return np.array(Prices)