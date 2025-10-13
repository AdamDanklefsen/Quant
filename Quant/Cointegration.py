import datetime as dt


def augmented_dickey_fuller_test(S1, S2):
    import pandas as pd
    assert isinstance(S1, pd.Series) and isinstance(S2, pd.Series)
    from statsmodels.tsa.stattools import adfuller
    result = adfuller(S1.dropna())
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

def coint_test(x,y):
    import pandas as pd
    import numpy as np
    from statsmodels.tsa.stattools import coint
    assert isinstance(x, pd.Series) and isinstance(y, pd.Series)
    assert len(x) == len(y)

    score, pvalue, _ = coint(x, y)
    return score, pvalue

def rolling_coint_test(x,y, period=dt.timedelta(days=30), delta=dt.timedelta(days=10)):
    import pandas as pd
    import numpy as np
    from statsmodels.tsa.stattools import coint
    assert isinstance(x, pd.Series) and isinstance(y, pd.Series)
    assert isinstance(period, dt.timedelta)
    assert len(x) == len(y)

    score = pd.Series()
    pvalue = pd.Series()

    startDay = x.index[0]
    while startDay < x.index[-1]-period:
        score[startDay + period], pvalue[startDay + period], _ = coint(x[startDay:startDay + period], y[startDay:startDay + period])
        startDay += delta
    coint_percentage = np.sum(pvalue<=0.05)/ len(pvalue)
    return score, pvalue, coint_percentage

