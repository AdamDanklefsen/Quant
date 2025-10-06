def SpreadFit(x,y):
    import numpy as np
    import pandas as pd
    assert len(x) == len(y)

    # Linear regression initialization
    from sklearn.linear_model import LinearRegression
    model = LinearRegression().fit(x.values.reshape(-1, 1), y.values)
    theta_0 = np.array([model.coef_[0], model.intercept_])

    # Initial Parameters
    theta_hat_0 = theta_0
    Q = np.eye(2) / 1e5  # Process Noise Covariance
    R = 10
    N = len(x)
    S = np.empty(N)
    theta_hat = np.empty((2,N))
    theta_hat[:,0] = theta_hat_0
    P_tn1_tn1 = np.identity(2)
    F = np.identity(2) # Identity State Transtion Matrix

    # Log = pd.DataFrame(index=data.index, columns=['',''])

    for i in range(1,len(x)):
        Ht = np.array([x.iat[i], 1]).T
        theta_hat_t_tn1 = F @ theta_hat[:,i-1]
        nu_t = y.iat[i] - Ht.T @ theta_hat_t_tn1
        P_t_tn1 = P_tn1_tn1 + Q
        Kt = (P_t_tn1 @ Ht) / (Ht.T @ P_t_tn1 @ Ht + R)
        A = np.identity(2) - np.outer(Kt, Ht)
        theta_hat[:,i] = theta_hat_t_tn1 + Kt * nu_t
        P_tn1_tn1 = A @ P_t_tn1
        S[i] = y.iat[i] - Ht.T @ theta_hat[:,i]
    S = pd.Series(S, index=x.index)
    theta = pd.DataFrame(theta_hat.T, index=x.index, columns=['Hedge Ratio', 'Intercept'])
    return S, theta