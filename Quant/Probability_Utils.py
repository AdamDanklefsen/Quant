def moments(x,y):
    import numpy as np
    
    Area = np.trapezoid(y, x)
    mu = np.trapezoid(x * y/Area, x)
    stddev = np.sqrt(np.trapezoid((x - mu)**2 * y/Area, x))
    skew = np.trapezoid((x - mu)**3 * y/Area, x) / (stddev**3 + 1e-8)
    kurt = np.trapezoid((x - mu)**4 * y/Area, x) / (stddev**4 + 1e-8)
    return Area, mu, stddev, skew, kurt