# All Functions in this file take in a pd.Series of prices and return a pd.Series of the same length
import pandas as pd

def moving_average(prices: pd.Series, window: int) -> pd.Series:
    return prices.rolling(window=window).mean()
def EWMA(prices: pd.Series, window: int) -> pd.Series:
    return prices.ewm(span=window, adjust=False).mean()
def stddev(prices: pd.Series, window: int) -> pd.Series:
    return prices.rolling(window=window).std()

def crossover_signal(prices: pd.Series, fast_window: int, slow_window: int) -> pd.Series:
    fast_ma = moving_average(prices, fast_window)
    slow_ma = moving_average(prices, slow_window)
    signal = (fast_ma < slow_ma).astype(int)
    return signal
def crossover_signal_edge(prices: pd.Series, fast_window: int, slow_window: int) -> pd.Series:
    fast_ma = moving_average(prices, fast_window)
    slow_ma = moving_average(prices, slow_window)
    signal = (fast_ma < slow_ma).astype(int)
    signal = (signal > signal.shift(1)).astype(int) - (signal < signal.shift(1)).astype(int)
    return signal.astype(int)
def RSI(prices: pd.Series, window: int = 14) -> pd.Series:
    U = prices.diff().clip(lower=0)
    D = -prices.diff().clip(upper=0)
    U_EMA = U.ewm(span=window, adjust=False).mean()
    D_EMA = D.ewm(span=window, adjust=False).mean()
    RS = U_EMA / D_EMA
    RSI = 100 - (100 / (1 + RS))
    return RSI
def RSI_normalized(prices: pd.Series, window: int = 14) -> pd.Series:
    rsi = RSI(prices, window)
    rsi_normalized = (rsi - 50) / rsi.std()
    return rsi_normalized
def Bollinger_Lower(prices: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.Series:
    ma = moving_average(prices, window)
    sd = stddev(prices, window)
    lower_band = ma - (num_std * sd)
    return lower_band
def Bollinger_Upper(prices: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.Series:
    ma = moving_average(prices, window)
    sd = stddev(prices, window)
    upper_band = ma + (num_std * sd)
    return upper_band