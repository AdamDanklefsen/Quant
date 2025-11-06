import torch
import torch.nn as nn
import Quant.TS_utils as TS

class RNNCell(nn.Module):
    def __init__(self, Nx: int, Nh: int, Ny: int,
                 kappa: float = 2, signslope: float = 20, tradingcost: float = 0.0001, lambda_var: float = 0.1,
                 nntype: str = 'RNN', device: str = 'cuda'):
        super(RNNCell, self).__init__()
        self.Nx = Nx
        self.Nh = Nh
        self.Ny = Ny
        self.T = 0
        # RNN Weights
        self.Wxh = nn.Parameter(torch.randn(self.Nh, self.Nx) * 0.01)
        self.Uhh = nn.Parameter(torch.randn(self.Nh, self.Nh) * 0.01)
        self.bh = nn.Parameter(torch.zeros(self.Nh, 1))
        # LSTM Weights
        self.Wf = nn.Parameter(torch.randn(self.Nh, self.Nx) * 0.01)
        self.Uf = nn.Parameter(torch.randn(self.Nh, self.Nh) * 0.01)
        self.bf = nn.Parameter(torch.zeros(self.Nh, 1))
        self.Wi = nn.Parameter(torch.randn(self.Nh, self.Nx) * 0.01)
        self.Ui = nn.Parameter(torch.randn(self.Nh, self.Nh) * 0.01)
        self.bi = nn.Parameter(torch.zeros(self.Nh, 1))
        self.Wo = nn.Parameter(torch.randn(self.Nh, self.Nx) * 0.01)
        self.Uo = nn.Parameter(torch.randn(self.Nh, self.Nh) * 0.01)
        self.bo = nn.Parameter(torch.zeros(self.Nh, 1))
        self.Wc = nn.Parameter(torch.randn(self.Nh, self.Nx) * 0.01)
        self.Uc = nn.Parameter(torch.randn(self.Nh, self.Nh) * 0.01)
        self.bc = nn.Parameter(torch.zeros(self.Nh, 1))

        self.Ahy = nn.Parameter(torch.randn(self.Ny, self.Nh) * 0.01)
        self.by = nn.Parameter(torch.zeros(self.Ny, 1))
        self.garch_omega = nn.Parameter(torch.tensor(0.000709))
        self.garch_alpha = nn.Parameter(torch.tensor(0.013078))
        self.garch_beta = nn.Parameter(torch.tensor(0.421968))
        self.kappa = kappa
        self.signslope = signslope
        self.tradingcost = tradingcost
        self.lambda_var = lambda_var
        self.nntype = nntype
        self.device = device
        self.epsilon = torch.tensor(1e-10, dtype=torch.float32).to(self.device)  # Small constant to prevent division by zero
        self.x_t_sharpe = torch.tensor([]).to(self.device)
        self.x_t_return = torch.tensor([]).to(self.device)

    # returns garch var(t) with var=0 for the first Nx steps
    def garch_forward(self, x_t: torch.tensor):
        var_list = []
        for i in range(self.T - self.Nx):
            prev_var = var_list[-1] if var_list else torch.zeros(1, device=self.device)
            x_prev = x_t[i - 1]  # scalar tensor
            new_var = (
                self.garch_omega
                + self.garch_alpha * x_prev ** 2
                + self.garch_beta * prev_var
            )
            var_list.append(new_var)

        var_t = torch.stack(var_list)  # Shape: (T, 1)
        return var_t
    
    def fit_garch(self, x_t: torch.tensor):
        optimizer = torch.optim.Adam([self.garch_omega, self.garch_alpha, self.garch_beta], lr=1e-2)
        self.train()
        for epoch in range(50):
            self.T = len(x_t)
            if self.T <= self.Nx:
                print(f"Input sequence length T={self.T} must be greater than lookback window Nx={self.Nx}")
                return
            optimizer.zero_grad()
            garch_var = self.garch_forward(x_t)
            garch_std = torch.sqrt(garch_var + self.epsilon)
            # print(f"garch_std: {garch_std.shape}, x_t slice: {x_t[self.Nx:].shape}")
            loss = torch.mean((garch_std - torch.abs(x_t[self.Nx:]))**2)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                self.garch_omega.clamp_(min=1e-6)
                self.garch_alpha.clamp_(min=0.0, max=1.0)
                self.garch_beta.clamp_(min=0.0, max=1.0)
                if self.garch_alpha + self.garch_beta >= 1.0:
                    total = self.garch_alpha + self.garch_beta
                    self.garch_alpha /= total
                    self.garch_beta /= total
            if epoch % 10 == 0:
                print(f"GARCH Fit Epoch {epoch+1}/100, Loss: {loss.item():.6f}, omega: {self.garch_omega.item():.6f}, alpha: {self.garch_alpha.item():.6f}, beta: {self.garch_beta.item():.6f}")


    def forward(self, x_t: torch.tensor):
        if self.nntype == 'RNN':
            h_t1 = torch.zeros(self.Nh, 1).to(self.device)
            mu_t_list = []
            for i in range(self.T - self.Nx):
                i_x = range(i, self.Nx+i)
                # print(f"i: {i}, x_t slice: {x_t[i_x].shape}, h_t1: {h_t1.shape}")
                # print(f"Wxh: {self.Wxh.shape}, Uhh: {self.Uhh.shape}, bh: {self.bh.shape}")
                h = torch.tanh(self.Wxh @ x_t[i_x] + self.Uhh @ h_t1 + self.bh)
                # print(f"h: {h.shape}, Ahy: {self.Ahy.shape}, by: {self.by.shape}")
                mu_t_list.append((self.Ahy @ h + self.by).squeeze())
                h_t1 = h
            mu_t = torch.stack(mu_t_list).unsqueeze(1)
            # print(f"mu_t final shape: {mu_t.shape}")
            return mu_t
        elif self.nntype == 'LSTM':
            h_t1 = torch.zeros(self.Nh, 1).to(self.device)
            c_t1 = torch.zeros(self.Nh, 1).to(self.device)
            mu_t_list = []
            for i in range(self.T - self.Nx):
                i_x = range(i, self.Nx+i)
                f_t = torch.sigmoid(self.Wf @ x_t[i_x] + self.Uf @ h_t1 + self.bf)
                i_t = torch.sigmoid(self.Wi @ x_t[i_x] + self.Ui @ h_t1 + self.bi)
                o_t = torch.sigmoid(self.Wo @ x_t[i_x] + self.Uo @ h_t1 + self.bo)
                c_hat_t = torch.tanh(self.Wc @ x_t[i_x] + self.Uc @ h_t1 + self.bc)
                c_t = f_t * c_t1 + i_t * c_hat_t
                h = o_t * torch.tanh(c_t)
                mu_t_list.append((self.Ahy @ h + self.by).squeeze())
                h_t1 = h
                c_t1 = c_t
            mu_t = torch.stack(mu_t_list).unsqueeze(1)
            return mu_t
        else:
            raise ValueError("nntype must be 'RNN' or 'LSTM'")
        
    def compute_return(self, logret: torch.tensor, normlogret: torch.tensor):
        w_t = self.compute_wt(normlogret)
        w_lag = torch.cat((torch.zeros(1, 1).to(self.device), w_t[:-1]))
        r_t = w_t * logret[self.Nx:] - self.tradingcost * torch.abs(w_t - w_lag)
        return r_t
    def compute_wt(self, x_t: torch.tensor):
        f = self.forward(x_t)
        # g = self.garch_forward(x_t)
        # w_t = torch.tanh(self.kappa * f / (g + self.epsilon))
        w_t = torch.tanh(self.kappa * f / .15)  # Using fixed stddev of 0.25 for normalization
        # zero w_t smoothly unless its magnitude exceeds .1
        w_t = torch.where(torch.abs(w_t) < .1, torch.zeros_like(w_t), w_t)
        return w_t
    

    def compute_cumulative_return(self, logret: torch.tensor, normlogret: torch.tensor):
        r_t = self.compute_return(logret, normlogret)
        cumulative_return = TS.compute_cumulative_return_from_log(r_t)
        return cumulative_return

    def compute_sharpe(self, logret: torch.tensor, normlogret: torch.tensor):
        r_t = self.compute_return(logret, normlogret)
        return TS.compute_series_sharpe(r_t)
    

    def compute_ema_sharpe(self, r_t, alpha=0.1):
        ema_mean = 0
        ema_var = 0
        sharpe_list = []

        for r in r_t:
            ema_mean = alpha * r + (1 - alpha) * ema_mean
            ema_var = alpha * (r - ema_mean).pow(2) + (1 - alpha) * ema_var
            sharpe = ema_mean / (torch.sqrt(ema_var) + self.epsilon)
            sharpe_list.append(sharpe)

        return torch.mean(torch.stack(sharpe_list))
    

    def sharpe_meanwt_loss(self, logret: torch.tensor, normlogret: torch.tensor):
        r_t = self.compute_return(logret, normlogret)
        sharpe = self.compute_ema_sharpe(r_t)
        w_t = self.compute_wt(logret)

        return -sharpe/self.x_t_sharpe - 10 * (TS.compute_final_cumulative_return_from_log(r_t)/self.x_t_return) + 10 * torch.mean(w_t)**2 # + 1000 * torch.mean((torch.abs(w_t)-.3)**2)
    
    def loss(self, logret: torch.tensor, normlogret: torch.tensor):
        return self.sharpe_meanwt_loss(logret, normlogret)

    def train_RNN(self, dataloader, epochs=50, lr=1e-3):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for logret, normlogret in dataloader:
                self.T = len(normlogret)
                self.x_t_sharpe = TS.compute_series_sharpe(logret[self.Nx:])
                self.x_t_return = TS.compute_final_cumulative_return_from_log(logret[self.Nx:])
                # print(f"Nx: {self.Nx}, Nh: {self.Nh}, T: {self.T}")
                if self.T <= self.Nx:
                    print(f"Input sequence length T={self.T} must be greater than lookback window Nx={self.Nx}")
                    continue
                optimizer.zero_grad()
                loss = self.loss(logret, normlogret)
                # print(f"Batch Sharpe: {self.compute_sharpe(logret, normlogret).item():.4f}, Loss: {loss.item():.4f}, total_loss: {total_loss:.4f}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=10)
                optimizer.step()
                total_loss += loss.item()
                # raise ValueError("Stop Here")
            
            logret = dataloader.dataset.tensors[0].to(self.device)
            normlogret = dataloader.dataset.tensors[1].to(self.device)
            self.T = len(logret)
            m, vol = TS.compute_meanvol(self.compute_return(logret, normlogret))
            wmean, wvol = TS.compute_meanvol(self.compute_wt(normlogret))
            print(f"{self.nntype}: Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}, "
                  f"Sharpe: {self.compute_sharpe(logret, normlogret).item():.4f}, "
                  f"Cumulative Return: {TS.compute_final_cumulative_return_from_log(self.compute_return(logret, normlogret)).item():.4%}, "
                  f"Mean Return: {m.item():.8%}, Volatility: {vol.item():.8f}, "
                  f"W Mean: {wmean.item():.8f}, W std: {wvol.item():.8f}")
            # Negate Ahy and by if Sharpe is negative
            if self.compute_sharpe(logret, normlogret) < 0.0:
                self.Ahy.data = -self.Ahy.data
                self.by.data = -self.by.data
                
            if torch.isnan(torch.tensor(total_loss)):
                print("Loss is NaN, stopping training.")
                break

    def load(self, filepath: str):
        self.load_state_dict(torch.load(filepath, map_location=self.device))

    def save(self, filepath: str):
        torch.save(self.state_dict(), filepath)
        print(f"Model saved to {filepath}")