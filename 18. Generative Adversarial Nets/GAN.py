import torch
import torch.nn as nn
import torch.optim as optim


class Generator(nn.Module):
    def __init__(self, z_dim, x_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.ReLU(),
            nn.Linear(128, x_dim),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, x_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


z_dim = 32
x_dim = 784

G = Generator(z_dim, x_dim)
D = Discriminator(x_dim)

opt_G = optim.Adam(G.parameters(), lr=1e-4)
opt_D = optim.Adam(D.parameters(), lr=1e-4)

bce = nn.BCELoss()

for _ in range(100):
    real = torch.randn(64, x_dim)
    z = torch.randn(64, z_dim)
    fake = G(z).detach()

    D_real = D(real)
    D_fake = D(fake)
    loss_D = bce(D_real, torch.ones_like(D_real)) + bce(D_fake, torch.zeros_like(D_fake))
    opt_D.zero_grad()
    loss_D.backward()
    opt_D.step()

    z = torch.randn(64, z_dim)
    fake = G(z)
    loss_G = bce(D(fake), torch.ones(64, 1))
    opt_G.zero_grad()
    loss_G.backward()
    opt_G.step()
