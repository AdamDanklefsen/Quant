import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import cvxpy as cvxpy

class ValueController(nn.Module):
    def __init__(self, Nx: int, Ny: int,
                 Wxy: torch.Tensor = None,
                 bxy: torch.Tensor = None,
                 device='cuda'):
        super(ValueController, self).__init__()
        self.device = device
        
        self.Nx = Nx
        self.Ny = Ny

        if Wxy is not None:
            self.Wxy = nn.Parameter(Wxy)
        else:
            self.Wxy = nn.Parameter(torch.ones(Nx, Ny, device=self.device) * 0.01)
        if bxy is not None:
            self.bxy = nn.Parameter(bxy)
        else:
            self.bxy = nn.Parameter(torch.zeros(Ny, device=self.device))

        self.features = None
        self.weights = None
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [time, tickers, features] * [features, output] -> [time, tickers]
        # print(f"Wxy: {self.Wxy.data}, bxy: {self.bxy.data}")
        # print(f"x: {x}")
        return torch.tanh(torch.matmul(x, self.Wxy) + self.bxy).squeeze()
    
    def train(self,
              mScore: pd.DataFrame,
              p2bScore: pd.DataFrame,
              roeScore: pd.DataFrame,
              incGrScore: pd.DataFrame,
              price_data: pd.DataFrame,
              start_date: str,
              rebalance_period: int = 252,
              epochs: int = 2000,
              lr: float = 0.001,
              verbose: bool = False):
        
        dates = price_data.index
        tickers = price_data.columns.tolist()

        # Pre-calculate initial weights using data before start_date
        # Using equal weights for training
        # pre_calc_log_rets = torch.tensor(np.log(price_data[price_data.index < start_date] / price_data[price_data.index < start_date].shift(1)).fillna(0).values, dtype=torch.float32, device=self.device)
        # weights_init = calcWeights(pre_calc_log_rets)
        weights_init = torch.ones(len(tickers), device=self.device) / len(tickers)


        log_rets = torch.tensor(np.log(price_data / price_data.shift(1)).fillna(0).values, dtype=torch.float32, device=self.device)

        self.features = torch.stack([
            torch.tensor(crossSectionNorm(mScore).values, dtype=torch.float32, device=self.device),
            torch.tensor(crossSectionNorm(p2bScore).values, dtype=torch.float32, device=self.device),
            torch.tensor(crossSectionNorm(roeScore).values, dtype=torch.float32, device=self.device),
            torch.tensor(crossSectionNorm(incGrScore).values, dtype=torch.float32, device=self.device)
        ], dim=2)
        # time, tickers, features

        weights = weights_init
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        for epoch in range(epochs):
            total_loss = 0.0
            pnl_history = []
            # print(dates.searchsorted(start_date), len(dates) - rebalance_period)
            for t in range(dates.searchsorted(start_date), len(dates) - rebalance_period, rebalance_period):
                x_batch = self.features[t:t+rebalance_period]
                r_batch = log_rets[t:t+rebalance_period]
                
                score = self.forward(x_batch)
                pnl = score * r_batch
                

                weighted_pnl = pnl @ weights
                pnl_history.extend(weighted_pnl.detach().cpu().numpy())
                IR = torch.mean(weighted_pnl, dim=0) / torch.std(weighted_pnl, dim=0).clamp(min=1e-6)
                # IR = torch.mean(weighted_pnl - r_batch @ weights, dim=0) / torch.std(weighted_pnl - r_batch @ weights, dim=0).clamp(min=1e-6)
                loss = -IR

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # Sticking with equal weights for training
                # weights = calcWeights(pnl)

                total_loss += loss.item()
                # print(f"Epoch {epoch+1}, Time {t}, Loss: {loss.item():.6f}, Sharpe: {sharpe.item():.6f}, W: {self.Wxy.cpu()}, B: {self.bxy.item():.6f}")
            IR = torch.mean(weighted_pnl, dim=0) / torch.std(weighted_pnl, dim=0).clamp(min=1e-6)
            annualized_return = np.exp(np.sum(pnl_history) * (365 / (dates[-1]-dates[dates.searchsorted(start_date)]).days)) - 1
            annualized_vol = np.std(pnl_history) * np.sqrt(252)
            total_sharpe = annualized_return / (annualized_vol + 1e-6)
            if epoch % 250 == 0:
                print(f"Epoch {epoch} completed. Total Loss: {total_loss:.6f}, Total Sharpe: {total_sharpe:.6f}, "
                    f"Total Return: {annualized_return:.6f}, Volatility: {annualized_vol:.6f}, "
                    f"Weights: {self.Wxy.detach().cpu().numpy().T}, Bias: {self.bxy.item():.6f}")

        raw_pnl = self.forward(self.features) * log_rets
        self.weights = calcWeights(raw_pnl).detach().cpu().numpy()
        print(f"Final portfolio weights: {self.weights}")




def crossSectionNorm(df: pd.DataFrame) -> pd.DataFrame:
    means = df.mean(axis=1)
    stds = df.std(axis=1)
    return df.sub(means, axis=0).div(stds, axis=0).fillna(0)

def normSoftmax(x: torch.Tensor) -> torch.Tensor:
    x_mean = torch.mean(x)
    x_std = torch.std(x)
    x_norm = (x - x_mean) / (x_std + 1e-6)
    return torch.softmax(x_norm, dim=0)

def calcWeights(rets: torch.Tensor) -> torch.Tensor:
    cov = torch.cov(rets.T).cpu().detach().numpy()
    w = cvxpy.Variable(rets.shape[1])
    mu = rets.mean(dim=0).cpu().detach().numpy()
    problem = cvxpy.Problem(
        cvxpy.Maximize(mu @ w - 10 * cvxpy.quad_form(w, cov)),
        [cvxpy.sum(w) == 1, w >= 0]
    )
    problem.solve()
    weights = torch.tensor(w.value, device=rets.device, dtype=rets.dtype)
    return weights