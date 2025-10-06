# k(x,x') = ln(1+exp( -(x-x') / l))
def fit_option_price(strike_data, price_data, current_price):
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LinearRegression

    assert isinstance(strike_data, pd.Series) and isinstance(price_data, pd.Series)
    assert len(strike_data) == len(price_data)

    # compute prior
    model = LinearRegression().fit(strike_data[strike_data < current_price].values.reshape(-1, 1),
                                    price_data[strike_data < current_price].values)
    f0 = np.maximum(0, model.predict(strike_data.values.reshape(-1, 1)))
    f0 = np.zeros_like(f0)
    sigman = 1e-3
    def kernel(x1, x2, length_scale):
        # return np.log(1 + np.exp(-np.abs(x1 - x2) / length_scale))
        return np.exp(-0.5 * ((x1 - x2) / length_scale) ** 2)
    def kernel_xx(x1, x2, length_scale):
        return kernel(x1, x2, length_scale) * ((x1-x2)**2/length_scale**4 - 1/length_scale**2)
    Kxx = np.array([[kernel(x1, x2, 15) for x2 in strike_data] for x1 in strike_data])
    A = np.linalg.inv(Kxx + sigman * np.eye(len(strike_data)))
    mu = f0 + Kxx @ A @ (price_data - f0)
    COV = Kxx - Kxx @ A @ Kxx
    Kxx_xx = np.array([[kernel_xx(x1, x2, 50) for x2 in strike_data] for x1 in strike_data])
    mu_xx = Kxx_xx @ A @ (price_data - f0)
    return mu, COV, mu_xx
