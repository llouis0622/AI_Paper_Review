import torch
import torch.nn as nn
import torch.optim as optim
import random


class Policy(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Linear(dim, dim)

    def forward(self, x):
        return self.net(x)


class RewardModel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Linear(dim, 1)

    def forward(self, x):
        return self.net(x)


def supervised_finetune(policy, data, optimizer):
    for x, y in data:
        pred = policy(x)
        loss = ((pred - y) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def train_reward_model(rm, comparisons, optimizer):
    for x, y_w, y_l in comparisons:
        rw = rm(y_w)
        rl = rm(y_l)
        loss = -torch.log(torch.sigmoid(rw - rl))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def ppo_step(policy, ref_policy, rm, x, optimizer, beta):
    y = policy(x)
    reward = rm(y)
    kl = ((policy(x) - ref_policy(x)) ** 2).mean()
    loss = -reward + beta * kl
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


dim = 8
policy = Policy(dim)
ref_policy = Policy(dim)
rm = RewardModel(dim)

opt_policy = optim.Adam(policy.parameters(), lr=1e-3)
opt_rm = optim.Adam(rm.parameters(), lr=1e-3)

dummy_data = [(torch.randn(dim), torch.randn(dim)) for _ in range(10)]
comparisons = [(torch.randn(dim), torch.randn(dim), torch.randn(dim)) for _ in range(10)]

supervised_finetune(policy, dummy_data, opt_policy)
train_reward_model(rm, comparisons, opt_rm)

for _ in range(5):
    x = torch.randn(dim)
    ppo_step(policy, ref_policy, rm, x, opt_policy, beta=0.1)
