def augmented_dickey_fuller_test(S, ):
    import pandas as pd
    assert isinstance(S, pd.Series)
    from statsmodels.tsa.stattools import adfuller
    result = adfuller(S.dropna())
    adf_stat = result[0]
    p_value = result[1]
    coint_percentage = 1 - p_value

    return coint_percentage, adf_stat

def rolling_linear_regression(x,y, period=20):
    from sklearn.linear_model import LinearRegression
    import numpy as np
    import pandas as pd
    import datetime as dt
    assert isinstance(x, pd.Series) and isinstance(y, pd.Series)
    assert isinstance(period, int) and period > 1
    assert len(x) == len(y)
    delta = x.index[1]-x.index[0]
    period = dt.timedelta(days=period)
    HedgeRatio = pd.Series(np.nan, index=x.index)
    intercept = pd.Series(np.nan, index=x.index)

    startDay = x.index[0]
    while startDay < x.index[-1]-period:
        model = LinearRegression().fit(x[startDay:startDay + period].values.reshape(-1,1), y[startDay:startDay + period].values)
        HedgeRatio[startDay + period] = model.coef_[0]
        intercept[startDay + period] = model.intercept_
        startDay += delta
    return HedgeRatio, intercept




