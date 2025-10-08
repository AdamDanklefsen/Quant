from datetime import datetime
import pandas as pd
def trading_days_between(start_date, end_date):
    from pandas.tseries.holiday import USFederalHolidayCalendar
    from pandas.tseries.offsets import CustomBusinessDay
    # Define a custom business day calendar excluding U.S. federal holidays
    cal = USFederalHolidayCalendar()
    cbd = CustomBusinessDay(calendar=cal)

    # Create a date index using that business day frequency
    dates = pd.date_range(start=start_date, end=end_date, freq=cbd)
    return len(dates)

# let f''(x) = p(x) = g(x)^2, where g(x) is a weighted sum of hermite functions
# f(x) = integral from x to inf of (K-x) * p(K) dK
# L = (y - wT M(x) w)^2 + lambda * (wT P w - 1)^2
# L'=0 -> lambda P w = 2(y-sigma wT M(x) w) M(x) w
# -> lambda P w = 2( sum(yM) - sigma sum((wTMw)M)) w
def fit_option_price_hermite(strike_data, price_data, forward_price, N = 3, w0=None):
    # Imports and Assertions
    import pandas as pd
    import torch
    assert isinstance(strike_data, pd.Series) and isinstance(price_data, pd.Series)
    assert len(strike_data) == len(price_data)
    if w0 is not None and N is not None:
        assert len(w0) == N

    # Initializations
    mu = torch.tensor(forward_price, dtype=torch.float32)
    sigma = torch.tensor(forward_price * 0.02, dtype=torch.float32)
    strike_data = torch.tensor(strike_data.values, dtype=torch.float32)
    price_data = torch.tensor(price_data.values, dtype=torch.float32)
    X = torch.arange(strike_data.min()/2, strike_data.max()*1.5, 1, dtype=torch.float32)
    if w0 is not None:
        w = torch.tensor(w0, dtype=torch.float32)
    else:
        w = torch.ones(N, dtype=torch.float32) / torch.tensor(N, dtype=torch.float32)
    print(f"# Strike Data: {len(strike_data)}, # Price Data: {len(price_data)}, # X: {len(X)}, # Weights: {len(w)}")
    print(f"Strike Data: {type(strike_data)}, Price Data: {type(price_data)}, X: {type(X)}, Weights: {type(w)}")
    # print(f"Strike Data: {strike_data}, Price Data: {price_data}, X: {X}, Weights: {w}")
    
    # Hermite Polynomial Basis
    # N generally small
    # returns probabilist's hermite functions of degree n
    # Only evaluate at z = (x-mu)/sigma
    def hermite(z, n):
        # z = (x-mu)/sigma
        if n == 0:
            return torch.ones_like(z) * torch.exp(-0.5 * z**2) / torch.sqrt(torch.tensor(2 * torch.pi))
        elif n == 1:
            return z * torch.exp(-0.5 * z**2) / torch.sqrt(torch.tensor(2 * torch.pi))
        else:
            return z * hermite(z, n-1) - hermite(z, n-2)
    # A(K) = integral of outer product of hermite functions from (k-mu)/sigma to inf
    def Hermite_A(x, mu, sigma, N, strike_data):
        A = torch.zeros((N, N, len(strike_data)))
        z = (x - mu) / sigma
        # iterate over diagonal and upper triangular then flip
        for i in range(N):
            for j in range(i, N):
                #integrate from (k-mu)/sigma to inf
                for k, ki in zip(strike_data, range(len(strike_data))):
                    mask = (z >= (k - mu) / sigma)
                    A[i,j,ki] = torch.trapezoid(hermite(z[mask], j) * hermite(z[mask], i), z[mask])
                A[j,i,:] = A[i,j,:]
        return A
    # B(K) = integral of outer product of hermite functions * z from (k-mu)/sigma to inf
    def Hermite_B(x, mu, sigma, N, strike_data):
        B = torch.zeros((N, N, len(strike_data)))
        z = (x - mu) / sigma
        # iterate over diagonal and upper triangular then flip
        for i in range(N):
            for j in range(i, N):
                #integrate from (k-mu)/sigma to inf
                for k, ki in zip(strike_data, range(len(strike_data))):
                    mask = (z >= (k - mu) / sigma)
                    B[i,j,ki] = torch.trapezoid(z[mask] * hermite(z[mask], j) * hermite(z[mask], i), z[mask])
                B[j,i,:] = B[i,j,:]
        return B
    # P(x) = integral of outer product of hermite functions from -inf to inf
    def Hermite_P(x, mu, sigma, N):
        P = torch.zeros((N, N))
        z = (x - mu) / sigma
        # iterate over diagonal and upper triangular then flip
        for i in range(N):
            for j in range(i, N):
                #integrate from -inf to inf
                P[i,j] = torch.trapezoid(hermite(z, j) * hermite(z, i), z)
                P[j,i] = P[i,j]
        return P
    # M(x) = (mu-k) * A(x) + sigma * B(x)
    def Hermite_M(x, mu, sigma, N, strike_data):
        return (mu - strike_data) * Hermite_A(x, mu, sigma, N, strike_data) + sigma * Hermite_B(x, mu, sigma, N, strike_data)
    def qForm(w, M):
        return torch.einsum('i,ijk,j->k', w, M, w)
    
    # Objective Function
    M = Hermite_M(X, mu, sigma, N, strike_data)
    My = torch.sum(M * price_data, dim=2) / len(strike_data)
    P = Hermite_P(X, mu, sigma, N)

    residual = torch.linalg.norm(price_data - qForm(w, M))
    iter = 0
    while residual > 1 and iter < 20:
        Q = torch.sum(qForm(w, M) * M, dim=2) / len(strike_data)
 
        eigVal, eigVec = torch.linalg.eig(2 * torch.linalg.pinv(P) @ (My - sigma * Q))
        # print(eigVal)
        # print(eigVec)

        resid = torch.zeros(N, dtype=torch.float32)
        for i in range(N):
            if not eigVec[:,i].isreal:
                print(f"Warning: Eigenvector {i} has imaginary component: {eigVec[:,i]}")
            resid[i] = torch.linalg.norm(price_data - qForm(eigVec[:,i].real, M)) + 500*(eigVec[:,i].real.T @ P @ eigVec[:,i].real - 1)
        
        w = eigVec[:, resid.argmin()].real
        w = w / torch.sqrt(w.T @ P @ w)

        print(f"Iter: {iter}, residual: {resid}, Selected weights: {w}, I: {w.T @ P @ w}")
        residual = torch.min(resid)
        iter +=1
    f = qForm(w, M).detach().numpy()
    h = torch.stack([torch.dot(w, torch.stack([hermite((k - mu) / sigma, n) for n in range(N)])) for k in strike_data]).detach().numpy()
    w = w.detach().numpy()
    return f, h, w

# let f''(x) = g(x) = h(x)^2, where h(x) is a weighted sum of B-splines
def fit_option_price_BSpline(strike_data, price_data, current_price, N = 3, w0=None):
    import numpy as np
    import pandas as pd
    import torch
    import torch.optim
    assert isinstance(strike_data, pd.Series) and isinstance(price_data, pd.Series)
    assert len(strike_data) == len(price_data)
    current_price = torch.tensor(current_price)


    X = torch.arange(strike_data.min()/2, strike_data.max()*1.5, 1)
    K_basis = torch.tensor([.5, .75, .85, .9, 1, 1.1,1.15, 1.25, 1.5])*current_price
    K_basis = torch.tensor([.5, .6, .7 ,.8 ,.9, .95, 1, 1.05, 1.1, 1.2, 1.3, 1.4, 1.5])*current_price
    knots = torch.tensor([.5, .7, .9, 1, 1.1, 1.3, 1.5])*current_price
    degree = 3
    knots = torch.cat((torch.tensor([knots[0]]*degree), knots, torch.tensor([knots[-1]]*degree)))
    observation_variance = 1e-3
    weights_covariance_INV = torch.eye(len(K_basis)-degree-1) * 1e-3
    if w0 is not None:
        weights = torch.tensor(w0)
    else:
        weights = torch.ones(len(K_basis)-degree-1) / 100
    ZERO = torch.tensor(0.)
    length_scale = torch.tensor(50.)

    def kernel(x1,x2, length_scale):
        return torch.exp(-0.5 *((x1 - x2) / length_scale) ** 2) / torch.sqrt(torch.tensor(2 * torch.pi * length_scale**2))
    


    def BSpline_A(x, knots, i):
        if knots[i+1] == knots[i]:
            return torch.zeros_like(x)
        else:
            return (x - knots[i]) / (knots[i+1] - knots[i])
    def BSpline_B(x, knots, i, degree):
        if knots[i+degree+1] == knots[i+1]:
            return torch.zeros_like(x)
        else:
            return (knots[i+degree+1] - x) / (knots[i+degree+1] - knots[i+1])
    def BSpline(x, knots, degree=3):
        print(f"# Knots: {len(knots)}, Degree: {degree}, # Basis: {len(knots)-degree-1}")
        n_basis = len(knots) - degree - 1 + degree
        deg_0 = torch.zeros((len(x), n_basis))
        deg_1 = torch.zeros((len(x), n_basis-1))
        deg_2 = torch.zeros((len(x), n_basis-2))
        deg_3 = torch.zeros((len(x), n_basis-3))

        for i in range(n_basis):
            deg_0[:, i] = torch.where(
                (knots[i] <= x) & (x <= knots[i+1]),
                torch.ones(len(x)),
                torch.zeros(len(x))
            )
        for i in range(n_basis - 1):
            deg_1[:, i] = BSpline_A(x, knots, i) * deg_0[:, i] + BSpline_B(x, knots, i, 1) * deg_0[:, i+1]
        for i in range(n_basis - 2):
            deg_2[:, i] = BSpline_A(x, knots, i)* deg_1[:, i] + BSpline_B(x, knots, i, 2) * deg_1[:, i+1]
        for i in range(n_basis - 3):
            deg_3[:, i] = BSpline_A(x, knots, i)* deg_2[:, i] + BSpline_B(x, knots, i, 3) * deg_2[:, i+1]
        return deg_3




    Spline = BSpline(X, knots, degree)
    print(Spline)
    print(Spline.shape)
    print(weights)
    print(weights.shape)

    l = 5e4
    W = torch.exp(torch.tensor(-.5) * (torch.tensor(strike_data.values) - current_price)**2/length_scale**2)/torch.sqrt(torch.tensor(2) * torch.pi * length_scale**2)
    # Posterior
    def Posterior(w):
        # h = torch.stack([torch.dot(w, torch.stack([kernel(x, kb, length_scale) for kb in K_basis])) for x in K])
        h = Spline @ w
        h2 = h**2 / (torch.trapezoid(h**2, X) + 1e-8)
        I = torch.stack([
            torch.trapz(torch.maximum(ZERO, X - k) * h2, X)
            for k in strike_data
        ])
        return torch.sum( W * (I - torch.tensor(price_data.values))**2 )/observation_variance + w.T @ weights_covariance_INV @ w + l * (torch.trapezoid(h**2, X)-1)**2

    w = weights.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([w], lr=0.1)
    for i in range(1000):
        optimizer.zero_grad()
        loss = Posterior(w)
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            print(f"Iteration {i}: Weights: {w}, Loss = {loss.item()}, I = {torch.trapezoid((Spline @ w)**2, X)}")

    h = Spline @ w
    f = torch.stack([torch.trapezoid(torch.maximum(ZERO, X - k) * h**2, X) for k in strike_data])
    # h = BSpline(torch.tensor(strike_data.values), knots, degree) @ w

    w = w.detach().numpy()
    h = h.detach().numpy()
    f = f.detach().numpy()

    return w, h, f, X
