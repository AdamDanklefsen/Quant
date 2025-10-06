# let f''(x) = g(x) = h(x)^2, where h(x) is a weighted sum of B-splines
def fit_option_price(strike_data, price_data, current_price):
    import numpy as np
    import pandas as pd
    import torch
    import torch.optim
    assert isinstance(strike_data, pd.Series) and isinstance(price_data, pd.Series)
    assert len(strike_data) == len(price_data)


    K = strike_data.min() + torch.arange(0, 1000, 1)
    K_basis = torch.tensor([.5, .75, .85, .9, 1, 1.1,1.15, 1.25, 1.5])*current_price
    K_basis = torch.tensor([.5, .6, .7 ,.8 ,.9, .95, 1, 1.05, 1.1, 1.2, 1.3, 1.4, 1.5])*current_price
    knots = torch.tensor([.5, .7, .9, 1, 1.1, 1.3, 1.5])*current_price
    degree = 3
    knots = torch.cat((torch.tensor([knots[0]]*degree), knots, torch.tensor([knots[-1]]*degree)))
    observation_variance = 1e-3
    weights_covariance_INV = torch.eye(len(K_basis)) * 1e-3
    weights = torch.ones(len(K_basis)) * 1
    ZERO = torch.tensor(0.)
    length_scale = 50

    def kernel(x1,x2, length_scale):
        return torch.exp(-0.5 *((x1 - x2) / length_scale) ** 2) / torch.sqrt(torch.tensor(2 * torch.pi * length_scale**2))
    def BSpline(x, knots, degree=3):
        n_basis = len(knots) - degree - 1
        deg_0 = torch.zeros((len(x), n_basis))
        deg_1 = torch.zeros((len(x), n_basis-1))
        deg_2 = torch.zeros((len(x), n_basis-2))
        deg_3 = torch.zeros((len(x), n_basis-3))

        for i in range(n_basis):
            deg_0[:, i] = torch.where(
                (knots[i] <= x) & (x < knots[i+1]),
                torch.ones(len(x)),
                torch.zeros(len(x))
            )
        for i in range(n_basis - 1):
            deg_1[:, i] = (x - knots[i]) / (knots[i+1] - knots[i]) * deg_0[:, i] + (knots[i+degree+1] - x) / (knots[i+degree+1] - knots[i+1]) * deg_0[:, i+1]
        for i in range(n_basis - 2):
            deg_2[:, i] = (x - knots[i]) / (knots[i+degree] - knots[i]) * deg_1[:, i] + (knots[i+degree+2] - x) / (knots[i+degree+2] - knots[i+1]) * deg_1[:, i+1]
        for i in range(n_basis - 3):
            deg_3[:, i] = (x - knots[i]) / (knots[i+degree+1] - knots[i]) * deg_2[:, i] + (knots[i+degree+3] - x) / (knots[i+degree+3] - knots[i+1]) * deg_2[:, i+1]
        return deg_3




    Spline = BSpline(K, knots, degree)
    print(Spline)
    
    # Posterior
    def Posterior(w):
        # h = torch.stack([torch.dot(w, torch.stack([kernel(x, kb, length_scale) for kb in K_basis])) for x in K])
        h = torch.dot(Spline, w)
        I = torch.stack([
            torch.trapz(torch.maximum(ZERO, K - k) * h**2, K)
            for k in strike_data
        ])
        return torch.sum( (I - torch.tensor(price_data.values))**2 )/observation_variance + w.T @ weights_covariance_INV @ w
    
    w = weights.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([w], lr=0.1)
    for i in range(1000):
        optimizer.zero_grad()
        loss = Posterior(w)
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            print(f"Iteration {i}: Weights: {w}, Loss = {loss.item()}")
    
    h = torch.dot(BSpline(K, knots, degree), w)
    f = torch.stack([torch.trapezoid(torch.maximum(ZERO, K-x) * h**2, K) for x in strike_data])
    h = torch.dot(BSpline(strike_data, knots, degree), w)

    w = w.detach().numpy()
    h = h.detach().numpy()
    f = f.detach().numpy()

    return w, h, f
