# def strat(prices: pd.Series, signals: pd.Series, budget: float = 10000) -> pd.Series, pd.Series:

import pandas as pd

def buy_and_hold(prices: pd.Series, signals: pd.Series = None, budget: float = 10000.0, risk_free_rate: float = 0.00) -> tuple[pd.Series, pd.Series, float]:
    shares = budget / prices.iloc[51]
    bank = pd.Series(0.0, index=prices.index)
    bank.iloc[0] = budget % prices.iloc[0]

    for i in range(1, len(prices)):
        bank.iloc[i] = (1 + risk_free_rate / 252) * bank.iloc[i - 1]

    bank.iloc[0:50] = budget

    Equity = shares * prices
    Equity.iloc[0:50] = 0
    Value = Equity + bank
    ROI = (Value - budget) / budget
    return Value, ROI, Sharpe_ratio(Value, risk_free_rate)

def crossover_10_50(prices: pd.Series, signals: pd.Series = None, budget: float = 10000, risk_free_rate: float = 0.00) -> tuple[pd.Series, pd.Series, float]:
    from Quant.Signals import crossover_signal_edge

    signals = crossover_signal_edge(prices, fast_window=10, slow_window=50)
    shares = pd.Series(index=prices.index)
    bank = pd.Series(0.0, index=prices.index)
    shares.iloc[0] = 0
    bank.iloc[0] = budget

    for i in range(1, len(prices)):
        bank.iloc[i] = (1 + risk_free_rate / 252) * bank.iloc[i - 1]
        if signals.iloc[i] == 1 and bank.iloc[i] >= prices.iloc[i]:
            shares.iloc[i] = bank.iloc[i] // prices.iloc[i]
            bank.iloc[i] -= shares.iloc[i] * prices.iloc[i]
        elif signals.iloc[i] == -1 and shares.iloc[i - 1] > 0:
            bank.iloc[i] += shares.iloc[i - 1] * prices.iloc[i]
            shares.iloc[i] = 0
        else:
            shares.iloc[i] = shares.iloc[i - 1]

    Value = shares * prices + bank
    ROI = (Value - budget) / budget
    return Value, ROI, Sharpe_ratio(Value, risk_free_rate)

def Zscore_strategy_discrete(prices: pd.Series, signals: pd.Series,
                             buystd: float = -1, sellstd: float = 1, budget: float = 10000,
                             risk_free_rate: float = 0.00) -> tuple[pd.Series, pd.Series, float, pd.Series, pd.Series]:
    shares = pd.Series(0, index=prices.index)
    bank = pd.Series(0.0, index=prices.index)
    shares.iloc[0] = 0
    bank.iloc[0] = budget

    for i in range(1, len(prices)):
        bank.iloc[i] = (1 + risk_free_rate / 252) * bank.iloc[i - 1]
        if signals.iloc[i] <= buystd and bank.iloc[i] >= prices.iloc[i]:
            newShares = bank.iloc[i] // prices.iloc[i]
            shares.iloc[i] = newShares + shares.iloc[i - 1]
            bank.iloc[i] -= newShares * prices.iloc[i]
            # print(f"Buying {newShares} shares at {prices.iloc[i]} on {prices.index[i]}, total shares: {shares.iloc[i]}, bank balance: {bank.iloc[i]}")
        elif signals.iloc[i] >= sellstd and shares.iloc[i - 1] > 0:
            bank.iloc[i] += shares.iloc[i - 1] * prices.iloc[i]
            shares.iloc[i] = 0
            # print(f"Selling {shares.iloc[i - 1]} shares at {prices.iloc[i]} on {prices.index[i]}, total shares: {shares.iloc[i]}, bank balance: {bank.iloc[i]}")
        else:
            shares.iloc[i] = shares.iloc[i - 1]

    Value = shares * prices + bank
    ROI = (Value - budget) / budget
    return Value, ROI, Sharpe_ratio(Value, risk_free_rate), shares, bank

def Zscore_strategy_atan(prices: pd.Series, signals: pd.Series, w: float=1, b: float=0, budget: float = 10000,
                         risk_free_rate: float = 0.00) -> tuple[pd.Series, pd.Series, float, pd.Series, pd.Series, pd.Series]:
    import numpy as np
    shares = pd.Series(0, index=prices.index)
    bank = pd.Series(0.0, index=prices.index)
    bank.iloc[0] = budget

    weights = np.atan(w * signals + b) * (2 / np.pi)
    weights.fillna(0, inplace=True)
    # print(weights)
    
    for i in range(1, len(prices)):
        bank.iloc[i] = (1 + risk_free_rate / 252) * bank.iloc[i - 1]

        # compute total equity (cash + holdings)
        equity = bank.iloc[i] + shares.iloc[i - 1] * prices.iloc[i]

        # target position based on *total equity* (not cash!)
        target_value = weights.iloc[i] * equity
        current_value = shares.iloc[i - 1] * prices.iloc[i]
        change_value = target_value - current_value

        # value difference we need to adjust
        change_value = target_value - current_value

        # number of shares to buy/sell
        delta_shares = change_value / prices.iloc[i]

        #  # enforce whole shares
        #  if delta_shares > 0:
        #      delta_shares = np.floor(delta_shares)     # buy fewer to not exceed target
        #  else:
        #      delta_shares = np.ceil(delta_shares)      # sell fewer to not exceed target

        # update positions
        shares.iloc[i] = shares.iloc[i - 1] + delta_shares

        # update cash (buying reduces, selling/shorting increases)
        bank.iloc[i] -= delta_shares * prices.iloc[i]

        # debug info (optional)
        if weights.iloc[i] == 0 or weights.iloc[i] == weights.iloc[i-1] or delta_shares == 0:
            continue
        # print(
        #     f"{prices.index[i]} | Weight: {weights.iloc[i]:.2f} | ΔShares: {delta_shares:.0f} | "
        #     f"Total Shares: {shares.iloc[i]:.0f} | Bank: {bank.iloc[i]:.2f}"
        # )


    
    Value = shares * prices + bank
    ROI = (Value - budget) / budget
    return Value, ROI, Sharpe_ratio(Value, risk_free_rate), shares, bank, weights

def Sharpe_ratio(Value: pd.Series, risk_free_rate: float = 0.00) -> float:
    import numpy as np
    ret = np.log(Value / Value.shift(1))
    total_ret = Value.iloc[-1] / Value.iloc[0] - 1
    annualized_ret = (1 + total_ret) ** (252 / (len(Value)-1)) - 1
    # print(f"Daily net return mean: {ret.mean() - risk_free_rate / 252}, std: {ret.std()}")
    print(f"Annualized return: {annualized_ret - risk_free_rate}, std: {ret.std() * (252 ** 0.5)}")
    return (annualized_ret - risk_free_rate) / (ret.std() * (252 ** 0.5) + 1e-8)


import torch
import torch.nn as nn

# Linear Atan Weighting Strategy
# w (-1, 1)
# x := vector of normalized signals 
# w = atan(ax+b)
class LATW(nn.Module):
    def __init__(self, Close, Features, is_real=None, N_hidden: int = 4,
                 w_hidden: torch.Tensor = None, b_hidden: torch.Tensor = None,
                 w0= -3.0, b0=6.0, risk_free_rate: float = 0.0, device: str = 'cuda'):
        super(LATW, self).__init__()
        # self.w = nn.Parameter(torch.zeros(N_features))
        # self.b = nn.Parameter(torch.zeros(1))
        N_features = Features.shape[1]
        self.w = nn.Parameter(torch.ones(N_hidden, dtype=torch.float32, device=device))
        self.b = nn.Parameter(torch.ones(1, device=device))
        self.w_hidden = nn.Parameter(torch.randn(N_features, N_hidden, device=device))
        self.b_hidden = nn.Parameter(torch.zeros(N_hidden, device=device))

        if is_real is None:
            self.is_real = torch.ones(Close.shape[0], dtype=torch.float32, device=device)
        else:
            self.is_real = is_real
        self.log_returns = torch.diff(torch.log(Close), prepend=torch.tensor([0.0], device=device))
        self.Features = Features
        self.Close = Close
        self.risk_free_rate = risk_free_rate

    def forward(self):
        return self.forward_second_order()
    def forward_second_order(self):
        z = self.Features @ self.w_hidden + self.b_hidden
        a_hidden = torch.relu(z)
        z = a_hidden @ self.w + self.b
        a = (2 / torch.pi) * torch.atan(z) * self.is_real
        return a.squeeze()
    def forward_first_order(self):
        z = self.Features @ self.w + self.b
        a = (2 / torch.pi) * torch.atan(z) * self.is_real
        return a.squeeze()

    def Sharpe_ratio(self, Value: torch.Tensor) -> float:
        log_returns = torch.log(Value[1:] / Value[:-1])
        total_ret = Value[-1] / Value[0] - 1
        annualized_ret = (1 + total_ret) ** (252 / (Value.shape[0]-1)) - 1
        # mean_ret = log_returns.mean() * 252 - self.risk_free_rate
        std_ret = (252**0.5) * log_returns.std(unbiased=False) + 1e-8
        sharpe = (annualized_ret - self.risk_free_rate) / std_ret
        return sharpe
    def torch_scan(self, price: torch.Tensor, X: torch.Tensor, budget: float = 10000,
                     ) -> tuple[torch.Tensor, torch.Tensor, float]:
        weights = self.forward(X)
        R = (1 + self.risk_free_rate / 252)

        def step(carry, inputs):
            shares_prev, bank_prev = carry
            price_i, weight_i, shares_i_minus1 = inputs

            shares_i = shares_prev + weight_i * R * bank_prev / price_i + (weight_i - 1) * shares_prev
            bank_i = R * bank_prev - (weight_i * R * bank_prev + (weight_i - 1) * shares_prev * price_i) * price_i

            return (shares_i, bank_i), (shares_i, bank_i)
        
        shares_0 = torch.tensor(0.0)
        bank_0 = torch.tensor(budget)
        carry_0 = (shares_0, bank_0)
        inputs = torch.stack((price[1:], weights[1:]),dim=1).T
        
        (final_state, outputs) = torch.scan(step, carry=carry_0, inputs=inputs)
        
        shares, bank = outputs
        shares = torch.cat((shares_0.unsqueeze(0), shares))
        bank = torch.cat((bank_0.unsqueeze(0), bank))

        Value = shares * price + bank
        ROI = (Value - budget) / budget
        return Value, ROI, self.Sharpe_ratio(Value), shares, bank, weights
    

    def Zscore_strategy_atan(self, budget: float = 10000,
                             ) -> tuple[torch.Tensor, torch.Tensor, float]:
        shares = torch.zeros(self.Features.shape[0], device=self.Features.device)
        bank = torch.zeros(self.Features.shape[0], device=self.Features.device)
        bank[0] = budget
        weights = self.forward()
        R = (1 + self.risk_free_rate / 252)

        for i in range(1, self.Features.shape[0]):
            bank[i] = R * bank[i - 1]

            equity = bank[i] + shares[i - 1] * self.Close[i]
            target_value = weights[i] * equity
            current_value = shares[i - 1] * self.Close[i]
            change_value = target_value - current_value
            delta_shares = change_value / self.Close[i]

            # # # delta_shares = weights[i] * R * bank[i-1] / price[i] + (weights[i] - 1) * shares[i - 1]

            # if delta_shares > 0:
            #     delta_shares = torch.floor(delta_shares)
            # else:
            #     delta_shares = torch.ceil(delta_shares)

            shares[i] = shares[i - 1] + delta_shares
            bank[i] -= delta_shares * self.Close[i]

            # shares[i] = shares[i - 1] + weights[i] * R * bank[i-1] / price[i] + (weights[i] - 1) * shares[i - 1]
            # bank[i] = R * bank[i - 1] - (weights[i] * R * bank[i-1] + (weights[i] - 1) * shares[i - 1] * price[i]) * price[i]

        Value = shares * self.Close + bank
        ROI = (Value - budget) / budget
        return Value, ROI, self.Sharpe_ratio(Value), shares, bank, weights

    def sharpe_loss(self):
        a = self.forward()  # weights in (-1,1)
        # Approximate cumulative return path
        strat_rets = a[:-1] * self.log_returns[1:]  # returns on position from previous step
        Value = torch.exp(torch.cumsum(strat_rets, dim=0))
        
        sharpe = self.Sharpe_ratio(Value)
        return -sharpe

    def atan_sharpe_loss(self, X: torch.Tensor, price: torch.Tensor, log_returns: torch.Tensor):
        return -self.Zscore_strategy_atan(price, X)[2]

    def fit(self, lr=0.01, epochs=500, verbose=True):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-4)
        for epoch in range(epochs):
            optimizer.zero_grad()
            loss = self.sharpe_loss()
            loss.backward()
            optimizer.step()


            if verbose and epoch % min(epochs // 10, min(50, epochs)) == 0:
                print(f"Epoch {epoch:3d} | Sharpe: {-loss.item():.4f}, "
                      f"w: {[f"{x:.4f}" for x in self.w.cpu().detach().numpy()]}, "
                        f"b: {self.b.cpu().detach().numpy()[0]:.4f}")
                
    def save(self, filepath: str):
        torch.save(self.state_dict(), filepath)
    def load(self, filepath: str):
        import os
        if os.path.exists(filepath):
            self.load_state_dict(torch.load(filepath))
            self.eval()