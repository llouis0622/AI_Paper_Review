import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionModel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, 128),
            nn.ReLU(),
            nn.Linear(128, dim)
        )

    def forward(self, x, t):
        t = t.float().unsqueeze(1)
        x = torch.cat([x, t], dim=1)
        return self.net(x)


def q_sample(x0, t, alpha_bar):
    eps = torch.randn_like(x0)
    return (
        torch.sqrt(alpha_bar[t]) * x0 +
        torch.sqrt(1 - alpha_bar[t]) * eps,
        eps
    )


def loss_fn(model, x0, t, alpha_bar):
    xt, eps = q_sample(x0, t, alpha_bar)
    eps_pred = model(xt, t)
    return F.mse_loss(eps_pred, eps)


T = 1000
beta = torch.linspace(1e-4, 0.02, T)
alpha = 1 - beta
alpha_bar = torch.cumprod(alpha, dim=0)
