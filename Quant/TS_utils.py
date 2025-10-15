import torch

def compute_cumulative_return_from_log(t: torch.tensor):
        return torch.exp(torch.cumsum(t, dim=0)) - 1
def compute_final_cumulative_return_from_log(r_t: torch.tensor):
        return torch.exp(torch.sum(r_t)) - 1


# Compute sharpe from given log return series
def compute_series_sharpe(y_t: torch.tensor, epsilon=1e-10):
    total_return = compute_final_cumulative_return_from_log(y_t)
    total_vol = torch.std(y_t) * torch.sqrt(torch.tensor(len(y_t), dtype=torch.float32).to(y_t.device))
    return total_return / (total_vol + epsilon)

def compute_meanvol(y_t: torch.tensor):
        total_return = torch.mean(y_t)
        total_vol = torch.std(y_t) * torch.sqrt(torch.tensor(len(y_t), dtype=torch.float32).to(y_t.device))
        return total_return, total_vol